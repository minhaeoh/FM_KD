import inspect
import math
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM

from configs.model_config import ModelConfig, default_teacher_layer_map


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if shift.dim() == 2:
        shift = shift[:, None, :]
    if scale.dim() == 2:
        scale = scale[:, None, :]
    return x * (1.0 + scale) + shift


def _prepare_timesteps(t: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=device)
    t = t.to(device=device)

    if t.ndim == 0:
        t = t.expand(batch_size)
    elif t.ndim == 2 and t.size(-1) == 1:
        t = t.squeeze(-1)
    elif t.ndim == 1 and t.size(0) == 1 and batch_size > 1:
        t = t.expand(batch_size)

    if t.ndim != 1 or t.size(0) != batch_size:
        raise ValueError(f"Expected timesteps broadcastable to [B], got {tuple(t.shape)} for B={batch_size}.")
    return t


def _prepare_2d_mask(mask: Optional[torch.Tensor], batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)

    mask = mask.to(device=device)
    if mask.ndim != 2 or mask.shape != (batch_size, seq_len):
        raise ValueError(f"Expected 2D mask shape {(batch_size, seq_len)}, got {tuple(mask.shape)}.")
    if mask.dtype == torch.bool:
        return mask
    return mask > 0


def build_asymmetric_4d_attention_mask(
    attn_2d: torch.Tensor,
    x_len: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Asymmetric bidirectional attention mask:
      - X queries cannot attend Y keys.
      - Y queries can attend X and Y keys.
    Padding keys are always masked out via attn_2d.
    """
    bsz, total_len = attn_2d.shape

    # Structural rule: disallow only X->Y block.
    structural = torch.ones((total_len, total_len), device=attn_2d.device, dtype=torch.bool)
    structural[:x_len, x_len:] = False
    structural = structural[None, None, :, :].expand(bsz, 1, total_len, total_len)

    # Key padding mask.
    valid_k = attn_2d.bool()[:, None, None, :].expand(bsz, 1, total_len, total_len)
    valid = structural & valid_k

    neg_inf = torch.finfo(dtype).min
    additive = torch.zeros((bsz, 1, total_len, total_len), device=attn_2d.device, dtype=dtype)
    return additive.masked_fill(~valid, neg_inf)


class TimestepEmbedder(nn.Module):
    """
    t -> sinusoidal embedding -> MLP -> conditioning vector.
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
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t=t, dim=self.freq_dim, max_period=self.max_period)
        # Keep timestep features in the same dtype as MLP weights (important for bf16/FSDP).
        t_freq = t_freq.to(dtype=self.mlp[0].weight.dtype)
        return self.mlp(t_freq)


class AdaLNModulation(nn.Module):
    """
    Produces (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp).
    Zero-init keeps the residual path near identity at startup.
    """

    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 6 * hidden_dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, c: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.proj(c)[:, None, :].chunk(6, dim=-1)


class StudentCFD(nn.Module):
    """
    Llama-compatible student model for hidden-state flow matching.

    Inputs:
      - x_ids, x_mask: conditioning tokens
      - z_y(t), y_mask: continuous y-latents at time t
      - t: scalar timestep per batch item in [0, 1]

    Conventions:
      - Concatenate [embed(x), z_y(t)] and run a shared transformer.
      - Apply time conditioning only on y states (x branch remains fixed).
      - Predict velocity only for y positions.
    """

    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.model_cfg = model_cfg

        teacher_cfg = AutoConfig.from_pretrained(model_cfg.teacher_model_id)
        if not isinstance(teacher_cfg, LlamaConfig):
            raise ValueError("Expected a Llama-family teacher config.")

        cfg_dict = teacher_cfg.to_dict()
        cfg_dict["num_hidden_layers"] = model_cfg.student_num_layers
        cfg_dict["output_hidden_states"] = False
        student_cfg = LlamaConfig(**cfg_dict)

        self.backbone: LlamaForCausalLM = LlamaForCausalLM(student_cfg)
        self.hidden_size = student_cfg.hidden_size
        self.num_layers = model_cfg.student_num_layers

        num_freq = max(1, int(model_cfg.time_num_frequencies))
        self.cond_dim = max(1, int(model_cfg.time_embedding_dim))
        self.time_embed = TimestepEmbedder(
            cond_dim=self.cond_dim,
            frequency_embedding_size=2 * num_freq,
        )
        self.adaln = nn.ModuleList(
            [AdaLNModulation(cond_dim=self.cond_dim, hidden_dim=self.hidden_size) for _ in range(self.num_layers)]
        )

        self.velocity_head = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        if model_cfg.velocity_head_init == "identity":
            nn.init.eye_(self.velocity_head.weight)
        else:
            nn.init.xavier_uniform_(self.velocity_head.weight)

        self.use_bidirectional_attention = model_cfg.use_bidirectional_attention
        layer_params = set(inspect.signature(self.backbone.model.layers[0].forward).parameters.keys())
        self._layer_params = layer_params
        self._supports_position_embeddings = "position_embeddings" in layer_params
        self._supports_cache_position = "cache_position" in layer_params
        self._supports_past_key_values = "past_key_values" in layer_params
        self._supports_past_key_value = "past_key_value" in layer_params

    def _teacher_load_dtype(self, device: Optional[torch.device]) -> torch.dtype:
        if device is not None:
            return torch.float16 if device.type == "cuda" else torch.float32
        param_device = next(self.parameters()).device
        return torch.float16 if param_device.type == "cuda" else torch.float32

    @torch.no_grad()
    def init_from_teacher(self, device: Optional[torch.device] = None) -> None:
        load_dtype = self._teacher_load_dtype(device)
        teacher = AutoModelForCausalLM.from_pretrained(
            self.model_cfg.teacher_model_id,
            torch_dtype=load_dtype,
            device_map=None,
        )
        if device is not None:
            teacher = teacher.to(device)
            self.to(device)

        teacher.eval()

        # Copy shared token/decoder heads.
        self.backbone.model.embed_tokens.weight.copy_(teacher.model.embed_tokens.weight)
        self.backbone.lm_head.weight.copy_(teacher.lm_head.weight)
        if hasattr(self.backbone.model, "norm") and hasattr(teacher.model, "norm"):
            self.backbone.model.norm.weight.copy_(teacher.model.norm.weight)

        t_layers = teacher.model.layers
        s_layers = self.backbone.model.layers
        num_teacher_layers = len(t_layers)
        num_student_layers = len(s_layers)

        if self.model_cfg.teacher_layer_indices is not None:
            idxs = list(self.model_cfg.teacher_layer_indices)
        else:
            idxs = default_teacher_layer_map(
                num_teacher_layers,
                num_student_layers,
                self.model_cfg.layer_copy_mode,
            )

        if len(idxs) != num_student_layers:
            raise ValueError(
                f"Teacher layer map length {len(idxs)} != number of student layers {num_student_layers}."
            )
        if min(idxs) < 0 or max(idxs) >= num_teacher_layers:
            raise ValueError(f"Teacher layer indices out of range: {idxs}.")

        for i, t_idx in enumerate(idxs):
            s_layers[i].load_state_dict(t_layers[t_idx].state_dict())

        del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[init] teacher->student layer map: {idxs}")

    def encode_x(self, x_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone.model.embed_tokens(x_ids)

    def decode_logits(self, h_y: torch.Tensor) -> torch.Tensor:
        return self.backbone.lm_head(h_y)

    def _forward_impl(
        self,
        z_y: torch.Tensor,
        t: torch.Tensor,
        x_ids: Optional[torch.Tensor],
        x_mask: torch.Tensor,
        y_mask: Optional[torch.Tensor] = None,
        x_states: Optional[torch.Tensor] = None,
        capture_layers: Optional[Set[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
        device = z_y.device
        batch_size, y_len, hidden_size = z_y.shape
        if hidden_size != self.hidden_size:
            raise ValueError(f"z_y last dim must match hidden_size={self.hidden_size}, got {hidden_size}.")

        if x_states is not None:
            if x_states.ndim != 3 or x_states.size(0) != batch_size or x_states.size(2) != self.hidden_size:
                raise ValueError(
                    f"x_states must have shape [B, x_len, {self.hidden_size}], got {tuple(x_states.shape)}."
                )
            x_emb = x_states.to(device=device, dtype=z_y.dtype)
        else:
            if x_ids is None:
                raise ValueError("Either x_ids or x_states must be provided.")
            x_emb = self.encode_x(x_ids)
        x_len = x_emb.size(1)

        x_mask_2d = _prepare_2d_mask(x_mask, batch_size, x_len, device=device)
        y_mask_2d = _prepare_2d_mask(y_mask, batch_size, y_len, device=device)

        hidden_states = torch.cat([x_emb, z_y], dim=1)
        attn_2d = torch.cat([x_mask_2d, y_mask_2d], dim=1)
        total_len = x_len + y_len

        attn_4d = build_asymmetric_4d_attention_mask(
            attn_2d=attn_2d,
            x_len=x_len,
            dtype=hidden_states.dtype,
        )

        cache_position = torch.arange(total_len, device=device)
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = None
        if self._supports_position_embeddings:
            rotary = self.backbone.model.rotary_emb
            try:
                position_embeddings = rotary(hidden_states, position_ids=position_ids)
            except TypeError:
                # Backward-compat path for older callable signatures.
                position_embeddings = rotary(hidden_states, position_ids)

        t = _prepare_timesteps(t, batch_size=batch_size, device=device)
        cond = F.silu(self.time_embed(t))

        layer_states: Dict[int, torch.Tensor] = {}

        for layer_idx, layer in enumerate(self.backbone.model.layers):
            (
                shift_attn,
                scale_attn,
                gate_attn,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = self.adaln[layer_idx](cond)

            h_x = hidden_states[:, :x_len, :]
            h_y = hidden_states[:, x_len:, :]

            # Y-only modulation: keep x branch stable.
            h_y_mod = modulate(h_y, shift=shift_attn, scale=scale_attn)
            hidden_in = torch.cat([h_x, h_y_mod], dim=1)

            layer_kwargs = {
                "hidden_states": hidden_in,
                "attention_mask": attn_4d,
                "position_ids": position_ids,
                "use_cache": False,
            }
            if "output_attentions" in self._layer_params:
                layer_kwargs["output_attentions"] = False
            if self._supports_position_embeddings and position_embeddings is not None:
                layer_kwargs["position_embeddings"] = position_embeddings
            if self._supports_cache_position:
                layer_kwargs["cache_position"] = cache_position
            if self._supports_past_key_values:
                layer_kwargs["past_key_values"] = None
            elif self._supports_past_key_value:
                layer_kwargs["past_key_value"] = None

            layer_out = layer(**layer_kwargs)
            out = layer_out[0] if isinstance(layer_out, tuple) else layer_out

            out_y = out[:, x_len:, :]

            # Identity-safe gating: zero gate means "keep previous y state".
            y_after_attn_gate = h_y + gate_attn * (out_y - h_y)
            y_mlp_mod = modulate(y_after_attn_gate, shift=shift_mlp, scale=scale_mlp)
            y_next = y_after_attn_gate + gate_mlp * (y_mlp_mod - y_after_attn_gate)

            # Keep x fixed across layers by design.
            hidden_states = torch.cat([h_x, y_next], dim=1)

            if capture_layers is not None and layer_idx in capture_layers:
                layer_states[layer_idx] = y_next

        hidden_states = self.backbone.model.norm(hidden_states)
        h_y_final = hidden_states[:, x_len:, :]
        v_y = self.velocity_head(h_y_final)
        return v_y, h_y_final, layer_states

    def forward(
        self,
        z_y: torch.Tensor,
        t: torch.Tensor,
        x_ids: Optional[torch.Tensor],
        x_mask: torch.Tensor,
        y_mask: Optional[torch.Tensor] = None,
        x_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        v_y, _, _ = self._forward_impl(
            z_y=z_y,
            t=t,
            x_ids=x_ids,
            x_mask=x_mask,
            y_mask=y_mask,
            x_states=x_states,
            capture_layers=None,
        )
        return v_y

    def forward_with_anchor_states(
        self,
        z_y: torch.Tensor,
        t: torch.Tensor,
        x_ids: Optional[torch.Tensor],
        x_mask: torch.Tensor,
        y_mask: Optional[torch.Tensor] = None,
        x_states: Optional[torch.Tensor] = None,
        anchor_layers: Optional[List[int]] = None,
        include_final: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if anchor_layers is None:
            capture_layers: Set[int] = set(range(self.num_layers))
        else:
            capture_layers = set(anchor_layers)
            invalid = [idx for idx in capture_layers if idx < 0 or idx >= self.num_layers]
            if invalid:
                raise ValueError(f"Anchor layer indices out of range: {invalid}")

        v_y, h_y_final, layer_states = self._forward_impl(
            z_y=z_y,
            t=t,
            x_ids=x_ids,
            x_mask=x_mask,
            y_mask=y_mask,
            x_states=x_states,
            capture_layers=capture_layers,
        )

        out: Dict[str, torch.Tensor] = {"v_y": v_y}
        for idx in sorted(layer_states):
            out[f"layer_{idx}"] = layer_states[idx]
        if include_final:
            out["h_y_final"] = h_y_final
        return out


def build_student_from_config(model_cfg: ModelConfig, device: Optional[torch.device] = None) -> StudentCFD:
    student = StudentCFD(model_cfg)
    if device is not None:
        student = student.to(device)
    student.init_from_teacher(device=device)
    student.train()
    return student
