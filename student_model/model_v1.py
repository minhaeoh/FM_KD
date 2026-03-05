# /model/model.py
import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaConfig

from configs.model_config import ModelConfig, default_teacher_layer_map


# ---------------------------
# AdaLN utilities (from flow_matching style)
# ---------------------------

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # x: [B,S,D], shift/scale: [B,1,D] or [B,D]
    if shift.dim() == 2:
        shift = shift[:, None, :]
    if scale.dim() == 2:
        scale = scale[:, None, :]
    return x * (1.0 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    flow_matching style timestep embedder:
      t -> sinusoidal (frequency_embedding_size) -> MLP -> cond_dim
    """
    def __init__(self, cond_dim: int, frequency_embedding_size: int = 256, max_period: int = 10000):
        super().__init__()
        self.cond_dim = cond_dim
        self.freq_dim = frequency_embedding_size
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, cond_dim, bias=True),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim, bias=True),
        )

    @staticmethod
    def timestep_embedding(time: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        # time: [B] float
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=time.device) / half
        )
        args = time[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.dim() == 2 and time.size(-1) == 1:
            time = time.squeeze(-1)
        t_freq = self.timestep_embedding(time=time, dim=self.freq_dim, max_period=self.max_period)
        return self.mlp(t_freq)  # [B, cond_dim]


class AdaLNModulation(nn.Module):
    """
    Produces (shift, scale, gate) for two sites (msa, mlp) like DiT:
      (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
    We'll use it conservatively:
      - apply shift/scale to inputs (pre-layer) for Y tokens
      - gate the layer output for Y tokens
    """
    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 6 * hidden_dim, bias=True)
        # zero-init to start near identity
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, c: torch.Tensor):
        # c: [B, cond_dim] -> [B, 6D] -> chunk
        out = self.proj(c)[:, None, :]  # [B,1,6D]
        return out.chunk(6, dim=-1)


def build_4d_attention_mask(attn_2d: torch.Tensor, q_len: int, kv_len: int, dtype: torch.dtype, causal: bool):
    B = attn_2d.size(0)
    neg_inf = torch.finfo(dtype).min
    pad = (1.0 - attn_2d.to(dtype)) * neg_inf
    pad = pad[:, None, None, :].expand(B, 1, q_len, kv_len)
    if not causal:
        return pad
    causal_mask = torch.full((q_len, kv_len), neg_inf, dtype=dtype, device=attn_2d.device)
    causal_mask = torch.triu(causal_mask, diagonal=1)[None, None, :, :].expand(B, 1, q_len, kv_len)
    return pad + causal_mask


class StudentCFD(nn.Module):
    """
    Llama backbone (16L) + AdaLN time conditioning + velocity head.
    """
    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.model_cfg = model_cfg

        teacher_cfg = AutoConfig.from_pretrained(model_cfg.teacher_model_id)
        if not isinstance(teacher_cfg, LlamaConfig):
            raise ValueError("Expected a Llama-family teacher config.")

        d = teacher_cfg.to_dict()
        d["num_hidden_layers"] = model_cfg.student_num_layers
        d["output_hidden_states"] = False
        student_cfg = LlamaConfig(**d)

        self.backbone: LlamaForCausalLM = LlamaForCausalLM(student_cfg)
        self.hidden_size = student_cfg.hidden_size
        self.num_layers = model_cfg.student_num_layers

        # ---- time conditioning (AdaLN) ----
        # use hidden_size as cond_dim for simplicity
        self.time_embed = TimestepEmbedder(cond_dim=self.hidden_size, frequency_embedding_size=256)
        self.adaln = nn.ModuleList([AdaLNModulation(cond_dim=self.hidden_size, hidden_dim=self.hidden_size)
                                   for _ in range(self.num_layers)])

        # ---- velocity head ----
        self.velocity_head = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        if model_cfg.velocity_head_init == "identity":
            nn.init.eye_(self.velocity_head.weight)
        else:
            nn.init.xavier_uniform_(self.velocity_head.weight)

        # attention mode
        self.use_bidirectional_attention = model_cfg.use_bidirectional_attention

    @torch.no_grad()
    def init_from_teacher(self, device: Optional[torch.device] = None):
        teacher = AutoModelForCausalLM.from_pretrained(
            self.model_cfg.teacher_model_id,
            torch_dtype=torch.float16,
            device_map=None,
        )
        if device is not None:
            teacher = teacher.to(device)
            self.to(device)

        teacher.eval()

        # copy embeddings / lm head / final norm
        self.backbone.model.embed_tokens.weight.copy_(teacher.model.embed_tokens.weight)
        self.backbone.lm_head.weight.copy_(teacher.lm_head.weight)
        if hasattr(self.backbone.model, "norm") and hasattr(teacher.model, "norm"):
            self.backbone.model.norm.weight.copy_(teacher.model.norm.weight)

        # copy layers
        t_layers = teacher.model.layers
        s_layers = self.backbone.model.layers

        T = len(t_layers)
        S = len(s_layers)

        if self.model_cfg.teacher_layer_indices is not None:
            idxs = self.model_cfg.teacher_layer_indices
        else:
            idxs = default_teacher_layer_map(T, S, self.model_cfg.layer_copy_mode)

        for i, t_idx in enumerate(idxs):
            s_layers[i].load_state_dict(t_layers[t_idx].state_dict())

        del teacher
        torch.cuda.empty_cache()
        print(f"[init] teacher->student layer map: {idxs}")

    def encode_x(self, x_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone.model.embed_tokens(x_ids)

    def decode_logits(self, h_y: torch.Tensor) -> torch.Tensor:
        return self.backbone.lm_head(h_y)

    def forward(
        self,
        z_y: torch.Tensor,               # [B,y_len,D]
        t: torch.Tensor,                 # [B] or [B,1]
        x_ids: torch.Tensor,             # [B,x_len]
        x_mask: torch.Tensor,            # [B,x_len]
        y_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = z_y.device
        B, y_len, D = z_y.shape

        x_emb = self.encode_x(x_ids)
        x_len = x_emb.size(1)

        if y_mask is None:
            y_mask = torch.ones((B, y_len), device=device, dtype=x_mask.dtype)

        # concat
        hidden_states = torch.cat([x_emb, z_y], dim=1)  # [B, x+y, D]
        attn_2d = torch.cat([x_mask, y_mask], dim=1)

        attn_4d = build_4d_attention_mask(
            attn_2d=attn_2d,
            q_len=x_len + y_len,
            kv_len=x_len + y_len,
            dtype=hidden_states.dtype,
            causal=(not self.use_bidirectional_attention),
        )

        position_ids = torch.arange(0, x_len + y_len, device=device).unsqueeze(0).expand(B, -1)

        # time conditioning vector c
        c = F.silu(self.time_embed(t.to(device)))  # [B, D]

        # run layers with AdaLN modulation (conservative)
        for li, layer in enumerate(self.backbone.model.layers):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln[li](c)

            # apply modulation ONLY to Y tokens (recommended)
            hx, hy = hidden_states[:, :x_len, :], hidden_states[:, x_len:, :]

            # we don't have explicit access to MSA/MLP internals of Llama layer,
            # so we do a single modulation before the whole layer:
            hy_mod = modulate(hy, shift=shift_msa, scale=scale_msa)

            hidden_in = torch.cat([hx, hy_mod], dim=1)

            out = layer(
                hidden_states=hidden_in,
                attention_mask=attn_4d,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )[0]

            # gate only Y output to keep X stable
            ox, oy = out[:, :x_len, :], out[:, x_len:, :]
            oy = oy * gate_msa  # [B,1,D] broadcast
            hidden_states = torch.cat([ox, oy], dim=1)

        hidden_states = self.backbone.model.norm(hidden_states)
        h_y = hidden_states[:, x_len:, :]
        v_y = self.velocity_head(h_y)
        return v_y


def build_student_from_config(model_cfg: ModelConfig, device: Optional[torch.device] = None) -> StudentCFD:
    student = StudentCFD(model_cfg)
    if device is not None:
        student = student.to(device)
    student.init_from_teacher(device=device)
    student.train()
    return student