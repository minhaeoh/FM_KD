from __future__ import annotations

import math
from dataclasses import asdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from torch import Tensor, nn

from config import DiffusionLlamaConfig


def _to_config(config: DiffusionLlamaConfig | Dict) -> DiffusionLlamaConfig:
    if isinstance(config, DiffusionLlamaConfig):
        return config
    if isinstance(config, dict):
        return DiffusionLlamaConfig.from_dict(config)
    # Support OmegaConf.DictConfig-like objects
    if hasattr(config, "items"):
        return DiffusionLlamaConfig.from_dict(dict(config.items()))
    raise TypeError(f"Unsupported config type: {type(config)}")


def rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    # q, k: [B, H, S, D], cos/sin: [S, D]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, S, D]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    # x: [B, H_kv, S, D] -> [B, H, S, D]
    if n_rep == 1:
        return x
    b, h_kv, s, d = x.shape
    x = x[:, :, None, :, :].expand(b, h_kv, n_rep, s, d)
    return x.reshape(b, h_kv * n_rep, s, d)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # Keep norm computation in fp32 for stability.
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_fp32 = x_fp32 * torch.rsqrt(variance + self.eps)
        x_fp32 = x_fp32 * self.weight.float()
        return x_fp32.to(dtype=x.dtype)


class TimestepEmbedder(nn.Module):
    def __init__(self, out_dim: int, frequency_embedding_size: int, max_period: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, out_dim, bias=True),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

    def _timestep_embedding(self, timesteps: Tensor) -> Tensor:
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
            / half
        )
        args = timesteps[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, timesteps: Tensor) -> Tensor:
        return self.mlp(self._timestep_embedding(timesteps))


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        base: float,
        scaling: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.scaling = scaling or None

    def _scaled_positions(self, seq_len: int, device: torch.device) -> Tensor:
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        if self.scaling is None:
            return pos

        scaling_type = self.scaling.get("type", "linear")
        factor = float(self.scaling.get("factor", 1.0))
        if scaling_type == "linear" and factor > 0:
            return pos / factor
        raise ValueError(f"Unsupported rope scaling config: {self.scaling}")

    def forward(self, seq_len: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        pos = self._scaled_positions(seq_len=seq_len, device=device)
        freqs = torch.outer(pos, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def _prepare_timesteps(timesteps: Tensor, batch_size: int, device: torch.device) -> Tensor:
    if not torch.is_tensor(timesteps):
        timesteps = torch.tensor(timesteps, device=device)
    timesteps = timesteps.to(device=device)

    if timesteps.ndim == 0:
        timesteps = timesteps.expand(batch_size)
    elif timesteps.ndim == 2 and timesteps.shape[1] == 1:
        timesteps = timesteps[:, 0]
    elif timesteps.ndim == 1 and timesteps.shape[0] == 1 and batch_size > 1:
        timesteps = timesteps.expand(batch_size)

    if timesteps.ndim != 1 or timesteps.shape[0] != batch_size:
        raise ValueError(
            "timesteps must be broadcastable to shape [B]. "
            f"Got shape {tuple(timesteps.shape)} for batch size {batch_size}."
        )

    return timesteps


def _mask_from_bool(valid: Tensor, dtype: torch.dtype) -> Tensor:
    # True -> keep, False -> mask out
    mask = torch.zeros_like(valid, dtype=dtype)
    return mask.masked_fill(~valid, torch.finfo(dtype).min)


def _prepare_attention_mask(
    attention_mask: Optional[Tensor],
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Tensor]:
    """
    Convert common padding masks to SDPA-compatible additive masks.
    Supported:
      - [B, S] with 1 valid / 0 pad
      - [B, S, S]
      - [B, 1, 1, S] or [B, 1, S, S]
    """
    if attention_mask is None:
        return None

    attention_mask = attention_mask.to(device=device)

    if attention_mask.ndim == 2:
        if attention_mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"2D attention_mask must have shape {(batch_size, seq_len)}, "
                f"got {tuple(attention_mask.shape)}."
            )
        valid = attention_mask.bool()
        return _mask_from_bool(valid[:, None, None, :], dtype=dtype)

    if attention_mask.ndim == 3:
        if attention_mask.shape[0] != batch_size or attention_mask.shape[1:] != (seq_len, seq_len):
            raise ValueError(
                f"3D attention_mask must have shape {(batch_size, seq_len, seq_len)}, "
                f"got {tuple(attention_mask.shape)}."
            )
        valid = attention_mask.bool()
        return _mask_from_bool(valid[:, None, :, :], dtype=dtype)

    if attention_mask.ndim == 4:
        if attention_mask.shape[0] != batch_size:
            raise ValueError(
                f"4D attention_mask batch dim mismatch: expected {batch_size}, "
                f"got {attention_mask.shape[0]}."
            )
        if attention_mask.dtype == torch.bool:
            return _mask_from_bool(attention_mask, dtype=dtype)

        # If user already passes additive mask (e.g. containing negative values), preserve it.
        if torch.is_floating_point(attention_mask) and torch.any(attention_mask < 0):
            return attention_mask.to(dtype=dtype)

        valid = attention_mask.bool()
        return _mask_from_bool(valid, dtype=dtype)

    raise ValueError(
        "attention_mask must be None, 2D [B,S], 3D [B,S,S], or 4D [B,1,1,S]/[B,1,S,S]. "
        f"Got {attention_mask.ndim}D tensor."
    )


class LlamaDiffusionAttention(nn.Module):
    def __init__(self, config: DiffusionLlamaConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor],
    ) -> Tensor:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).view(
            bsz, seq_len, self.num_key_value_heads, self.head_dim
        )

        q = q.transpose(1, 2)  # [B, H, S, D]
        k = k.transpose(1, 2)  # [B, H_kv, S, D]
        v = v.transpose(1, 2)

        # Apply RoPE in fp32 for numerical stability.
        with torch.amp.autocast("cuda", enabled=False):
            q_fp32, k_fp32 = apply_rotary_pos_emb(q.float(), k.float(), cos.float(), sin.float())
        q = q_fp32.to(dtype=q.dtype)
        k = k_fp32.to(dtype=k.dtype)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(attn_out)


class LlamaDiffusionMLP(nn.Module):
    def __init__(self, config: DiffusionLlamaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDiffusionDecoderLayer(nn.Module):
    def __init__(self, config: DiffusionLlamaConfig) -> None:
        super().__init__()
        self.self_attn = LlamaDiffusionAttention(config)
        self.mlp = LlamaDiffusionMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.residual_dropout = nn.Dropout(config.residual_dropout)

        # DiT/DDiT-style conditioning: shift/scale/gate for attention and MLP streams.
        self.adaLN_modulation = nn.Linear(config.cond_dim, 6 * config.hidden_size, bias=True)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def _modulate(self, x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
        return x * (1 + scale) + shift

    def forward(
        self,
        hidden_states: Tensor,
        cond: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Optional[Tensor],
    ) -> Tensor:
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(cond).unsqueeze(1).chunk(6, dim=-1)

        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x = self._modulate(x, shift_attn, scale_attn)
        attn_out = self.self_attn(x, cos=cos, sin=sin, attention_mask=attention_mask)
        hidden_states = residual + gate_attn * self.residual_dropout(attn_out)

        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states)
        x = self._modulate(x, shift_mlp, scale_mlp)
        mlp_out = self.mlp(x)
        hidden_states = residual + gate_mlp * self.residual_dropout(mlp_out)

        return hidden_states


class LlamaDiffusionBackbone(nn.Module):
    """
    Llama-like bidirectional backbone used for continuous hidden-state diffusion.

    Teacher-compatible names (where possible):
      - model.embed_tokens
      - model.layers[i].self_attn.{q_proj,k_proj,v_proj,o_proj}
      - model.layers[i].mlp.{gate_proj,up_proj,down_proj}
      - model.layers[i].input_layernorm / post_attention_layernorm
      - model.norm
    """

    def __init__(self, config: DiffusionLlamaConfig) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [LlamaDiffusionDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_emb = LlamaRotaryEmbedding(
            dim=head_dim,
            base=config.rope_theta,
            scaling=config.rope_scaling,
        )
        self.time_embed = TimestepEmbedder(
            out_dim=config.cond_dim,
            frequency_embedding_size=config.time_embed_dim,
            max_period=config.time_max_period,
        )

    def _forward_hidden(
        self,
        hidden_states: Tensor,
        timesteps: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                f"Expected hidden states shape [B, S, H], got {tuple(hidden_states.shape)}."
            )

        batch_size, seq_len, hidden_size = hidden_states.shape
        if hidden_size != self.config.hidden_size:
            raise ValueError(
                f"Last dim of hidden states must be {self.config.hidden_size}, got {hidden_size}."
            )

        timesteps = _prepare_timesteps(timesteps, batch_size=batch_size, device=hidden_states.device)
        attn_mask = _prepare_attention_mask(
            attention_mask=attention_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        cos, sin = self.rotary_emb(seq_len=seq_len, device=hidden_states.device)
        cond = F.silu(self.time_embed(timesteps))

        # Use bf16 autocast on CUDA; keep critical ops fp32 inside their modules.
        amp_enabled = hidden_states.is_cuda
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            for layer in self.layers:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    cond=cond,
                    cos=cos,
                    sin=sin,
                    attention_mask=attn_mask,
                )
            hidden_states = self.norm(hidden_states)

        return hidden_states

    def forward(
        self,
        input_ids: Tensor,
        timesteps: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        hidden_states = self.embed_tokens(input_ids)
        return self._forward_hidden(hidden_states, timesteps=timesteps, attention_mask=attention_mask)

    def forward_hidden(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self._forward_hidden(x_t, timesteps=timesteps, attention_mask=attention_mask)


class LlamaFlowMatchingModel(nn.Module):
    """
    Top-level model used for hidden-state flow matching.

    Primary diffusion output: v_pred (continuous velocity prediction).
    Optional output: logits from lm_head for auxiliary/token-level objectives.
    """

    def __init__(self, config: DiffusionLlamaConfig | Dict) -> None:
        super().__init__()
        self.config = _to_config(config)

        self.model = LlamaDiffusionBackbone(self.config)
        self.v_head = nn.Linear(self.config.hidden_size, self.config.out_dim, bias=True)

        # Keep this trainable and teacher-compatible for weight copy.
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def _build_outputs(
        self,
        hidden_states: Tensor,
        return_logits: bool,
        return_hidden: bool,
    ) -> Dict[str, Tensor]:
        pred = self.v_head(hidden_states)
        outputs: Dict[str, Tensor] = {"v_pred": pred}

        # Optional alias if the caller tracks non-velocity prediction naming.
        if self.config.pred_type != "velocity":
            outputs[f"{self.config.pred_type}_pred"] = pred

        if return_logits:
            outputs["logits"] = self.lm_head(hidden_states)
        if return_hidden:
            outputs["hidden_states"] = hidden_states
        return outputs

    def forward(
        self,
        input_ids: Tensor,
        timesteps: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_logits: bool = False,
        return_hidden: bool = False,
    ) -> Dict[str, Tensor]:
        hidden_states = self.model(
            input_ids=input_ids,
            timesteps=timesteps,
            attention_mask=attention_mask,
        )
        return self._build_outputs(
            hidden_states=hidden_states,
            return_logits=return_logits,
            return_hidden=return_hidden,
        )

    def forward_hidden(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_logits: bool = False,
    ) -> Dict[str, Tensor]:
        hidden_states = self.model.forward_hidden(
            x_t=x_t,
            timesteps=timesteps,
            attention_mask=attention_mask,
        )
        return self._build_outputs(
            hidden_states=hidden_states,
            return_logits=return_logits,
            return_hidden=False,
        )


def _sanity_check() -> None:
    torch.manual_seed(0)
    cfg = DiffusionLlamaConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=192,
        cond_dim=64,
        out_dim=64,
        attention_dropout=0.0,
        residual_dropout=0.0,
    )
    model = LlamaFlowMatchingModel(cfg)
    model.eval()

    bsz, seq_len = 2, 7
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq_len))
    timesteps = torch.rand(bsz)
    pad_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    # forward() with padding mask + logits + hidden output.
    out = model(
        input_ids=input_ids,
        timesteps=timesteps,
        attention_mask=pad_mask,
        return_logits=True,
        return_hidden=True,
    )
    assert out["v_pred"].shape == (bsz, seq_len, cfg.out_dim)
    assert out["logits"].shape == (bsz, seq_len, cfg.vocab_size)
    assert out["hidden_states"].shape == (bsz, seq_len, cfg.hidden_size)

    # forward() without mask should also work (bidirectional attention, no causal mask).
    out_nomask = model(input_ids=input_ids, timesteps=timesteps)
    assert out_nomask["v_pred"].shape == (bsz, seq_len, cfg.out_dim)

    # forward_hidden() path for continuous hidden-state diffusion.
    x_t = torch.randn(bsz, seq_len, cfg.hidden_size)
    out_hidden = model.forward_hidden(
        x_t=x_t,
        timesteps=timesteps,
        attention_mask=pad_mask[:, None, None, :],
        return_logits=True,
    )
    assert out_hidden["v_pred"].shape == (bsz, seq_len, cfg.out_dim)
    assert out_hidden["logits"].shape == (bsz, seq_len, cfg.vocab_size)

    print("Sanity check passed.", asdict(cfg))


if __name__ == "__main__":
    _sanity_check()
