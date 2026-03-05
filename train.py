import os
import math
import time
import random
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from configs.train_config import TrainConfig
from configs.model_config import ModelConfig
from dataset.hidden_state_dataset import FlowBatchCollator, FlowHiddenStateDataset
from student_model.model import StudentCFD

# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_checkpoint(
    cfg: TrainConfig,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> None:
    ensure_dir(cfg.output_dir)
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": cfg.__dict__,
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    path = os.path.join(cfg.output_dir, f"{cfg.run_name}_step{step}.pt")
    torch.save(ckpt, path)
    print(f"[ckpt] saved: {path}")


def _extract_student_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint payload must be a dict, got {type(obj)}.")

    # Case 1: raw state_dict payload.
    if obj and all(torch.is_tensor(v) for v in obj.values()):
        state = obj
    else:
        # Case 2: wrapped checkpoints.
        state = None
        for key in ("model", "student", "state_dict"):
            value = obj.get(key, None)
            if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
                state = value
                break
        if state is None:
            raise KeyError(
                "Could not find student state_dict in checkpoint. "
                "Expected raw state_dict or one of keys: model/student/state_dict."
            )

    # Handle DataParallel/DDP prefix if present.
    if state and all(k.startswith("module.") for k in state.keys()):
        state = {k[len("module.") :]: v for k, v in state.items()}
    return state


def load_student_init_checkpoint(student: nn.Module, ckpt_path: str) -> None:
    payload = torch.load(ckpt_path, map_location="cpu")
    state = _extract_student_state_dict(payload)
    student.load_state_dict(state, strict=True)
    print(f"[init] loaded student_init_ckpt: {ckpt_path}")

# ============================================================
# Expected dataset batch format
# ============================================================
"""
Your DataLoader should yield a batch dict with at least:

batch = {
  # conditioning tokens (X)
  "x_ids": LongTensor [B, x_len]
  "x_mask": Bool/LongTensor [B, x_len]  (1 for valid tokens)

  # target tokens (Y) for optional CE/KL
  "y_ids": LongTensor [B, y_len]
  "y_mask": Bool/LongTensor [B, y_len]  (1 for valid tokens)

  # teacher hidden anchors for Y positions only
  # e.g. 12 anchors from teacher layers -> (B, M, y_len, D)
  "teacher_anchors": FloatTensor [B, M, y_len, D]

  # teacher X hidden states for all sampled layers (M+1 endpoints)
  "teacher_x_layers": FloatTensor [B, M+1, x_len, D]

  # fixed-length full-sequence placeholders + mask
  "input_ids_fixed": LongTensor [B, max_length]
  "attention_mask": LongTensor [B, max_length]

  # optional: teacher logits for Y positions (if cached)
  # "teacher_logits": FloatTensor [B, y_len, vocab]

  # optional: non-uniform times (length M+1)
  # "times": FloatTensor [M+1]  (global for batch)
}
"""

# ============================================================
# Core: sample interval k and time t inside the interval
# ============================================================

def sample_interval_and_time(
    cfg: TrainConfig,
    times: torch.Tensor,  # [M+1]
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      k:    LongTensor [B] interval index in {0..M-1}
      s:    FloatTensor [B] in [0,1] local coordinate within interval
      t0:   FloatTensor [B]
      t1:   FloatTensor [B]
    """
    M = cfg.num_intervals
    # interval index
    k = torch.randint(low=0, high=M, size=(batch_size,), device=device)

    # local coordinate s in [0,1]
    s = torch.rand(batch_size, device=device)

    # gather t0, t1
    t0 = times[k]          # [B]
    t1 = times[k + 1]      # [B]
    return k, s, t0, t1


# ============================================================
# Main training step skeleton
# ============================================================

class MaskInit(nn.Module):
    """
    Learnable mask embedding used to initialize z0(Y) as:
      z0 = mask_embed + noise
    """
    def __init__(self, hidden_dim: int, learnable: bool = True):
        super().__init__()
        if learnable:
            self.mask = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.register_buffer("mask", torch.zeros(hidden_dim), persistent=False)

    def forward(self, batch_size: int, y_len: int, device: torch.device) -> torch.Tensor:
        # [B, y_len, D]
        return self.mask.view(1, 1, -1).to(device).expand(batch_size, y_len, -1)


def compute_losses_skeleton(
    cfg: TrainConfig,
    student: nn.Module,
    mask_init: MaskInit,
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    This function produces:
      - loss_total
      - loss_fm, loss_anchor
      - optional loss_ce, loss_kl

    NOTE:
      CE/KL are optional and activated via cfg.lambda_ce / cfg.lambda_kl.
    """
    device = next(student.parameters()).device

    x_mask = batch["x_mask"].to(device)

    y_mask = batch["y_mask"].to(device)  # [B, y_len] (0/1)

    teacher_anchors = batch["teacher_anchors"].to(device)  # [B, M, y_len, D]
    teacher_x_layers = batch["teacher_x_layers"].to(device)  # [B, M+1, x_len, D]
    B, M, y_len, D = teacher_anchors.shape
    _, M_x, x_len, D_x = teacher_x_layers.shape

    assert M == cfg.num_intervals, f"Expected {cfg.num_intervals} anchors, got {M}"
    assert M_x == (M + 1), f"Expected teacher_x_layers dim1={M+1}, got {M_x}"
    assert D_x == D, f"X/Y hidden dim mismatch: D_x={D_x}, D_y={D}"

    # times: [M+1]
    if "times" in batch:
        times = batch["times"].to(device)
    else:
        # default uniform grid
        times = torch.linspace(0.0, 1.0, steps=M + 1, device=device)

    # ------------------------------------------------------------
    # Initialize z0(Y)
    # ------------------------------------------------------------
    z0 = mask_init(batch_size=B, y_len=y_len, device=device)
    if cfg.z0_noise_std > 0:
        z0 = z0 + cfg.z0_noise_std * torch.randn_like(z0)

    # ------------------------------------------------------------
    # Sample interval k and local time s
    # ------------------------------------------------------------
    # We will sample per-sample (k,s). If you want R>1 samples per pair,
    # you can repeat/expand batch by factor R (see note below).
    k, s, t0, t1 = sample_interval_and_time(cfg, times, B, device)

    # compute actual t in [t0, t1]
    t = (1.0 - s) * t0 + s * t1  # [B]

    # ------------------------------------------------------------
    # Select endpoints a, b for interval k
    # Convention:
    #   interval 0: a = z0,                 b = teacher_anchors[:, 0]
    #   interval i: a = teacher_anchors[:, i-1], b = teacher_anchors[:, i]
    # ------------------------------------------------------------
    # Build a and b (shape [B, y_len, D])
    a = torch.empty((B, y_len, D), device=device, dtype=teacher_anchors.dtype)
    b = torch.empty((B, y_len, D), device=device, dtype=teacher_anchors.dtype)

    # mask for interval 0
    is0 = (k == 0)
    if is0.any():
        a[is0] = z0[is0]
        b[is0] = teacher_anchors[is0, 0]

    # intervals 1..M-1
    if (~is0).any():
        kk = k[~is0]
        # a = anchor[k-1], b = anchor[k]
        a[~is0] = teacher_anchors[~is0, kk - 1]
        b[~is0] = teacher_anchors[~is0, kk]

    dt = (t1 - t0).clamp_min(1e-6)  # [B]

    # ------------------------------------------------------------
    # Construct z_t via a bridge (start with linear interpolation)
    # z_t = (1-s)*a + s*b
    # ------------------------------------------------------------
    z_t = (1.0 - s).view(B, 1, 1) * a + s.view(B, 1, 1) * b

    # Optional: add small noise (Y only) to improve robustness
    # TODO: if you want, add something like:
    # z_t = z_t + sigma(t) * torch.randn_like(z_t)

    # ------------------------------------------------------------
    # Construct X_t from the sampled interval left endpoint.
    # Interval k corresponds to teacher layer transition L_k -> L_{k+1}.
    # We set x_t = x0 = hidden(L_k) (not a single global x_0).
    # ------------------------------------------------------------
    x_idx = k.view(B, 1, 1, 1).expand(-1, 1, x_len, D)
    x_t = torch.gather(teacher_x_layers, dim=1, index=x_idx).squeeze(1)  # [B, x_len, D]

    # ------------------------------------------------------------
    # Student predicts velocity v_theta(z_t, t, X)
    # ------------------------------------------------------------
    v = student(
        z_y=z_t,
        t=t,              # [B]
        x_ids=None,
        x_mask=x_mask,
        y_mask=y_mask,
        x_states=x_t,
    )  # [B, y_len, D]

    # ------------------------------------------------------------
    # Target velocity (for linear bridge):
    #   u* = (b - a) / dt
    # ------------------------------------------------------------
    u_star = (b - a) / dt.view(B, 1, 1)

    # ------------------------------------------------------------
    # Flow Matching loss (masked MSE)
    # ------------------------------------------------------------
    if "y_len_real" in batch:
        y_len_real = batch["y_len_real"].to(device)  # derived from fixed-length attention_mask
        y_pos = torch.arange(y_len, device=device).unsqueeze(0)
        attn_valid_y = y_pos < y_len_real.unsqueeze(1)
        y_valid = y_mask.bool() & attn_valid_y
    else:
        y_valid = y_mask.bool()

    loss_mask = y_valid.unsqueeze(-1).to(v.dtype)  # [B, y_len, 1], PAD=0
    denom = loss_mask.sum().clamp_min(1.0)
    loss_fm = ((v - u_star).pow(2) * loss_mask).sum() / denom

    # ------------------------------------------------------------
    # Anchor / endpoint consistency loss:
    # predict endpoint b_hat from (z_t, v) using Euler to t1:
    #   b_hat = z_t + (t1 - t) * v
    # ------------------------------------------------------------
    b_hat = z_t + (t1 - t).view(B, 1, 1) * v

    loss_anchor = ((b_hat - b).pow(2) * loss_mask).sum() / denom

    # ------------------------------------------------------------
    # Optional CE/KL at final interval only
    # If k == M-1 then b_hat approximates the t=1 endpoint.
    # ------------------------------------------------------------
    loss_ce = loss_anchor.new_zeros(())
    loss_kl = loss_anchor.new_zeros(())
    use_ce = float(cfg.lambda_ce) != 0.0
    use_kl = float(cfg.lambda_kl) != 0.0
    is_last = (k == (M - 1))
    ignore_index = -100

    if (use_ce or use_kl) and is_last.any():
        decode_logits = getattr(student, "decode_logits", None)
        if not callable(decode_logits):
            raise AttributeError(
                "Student model must implement `decode_logits(h_y)` when lambda_ce/lambda_kl is non-zero."
            )
        student_logits = decode_logits(b_hat[is_last]).to(dtype=torch.float32)

        if use_ce:
            if "y_ids" not in batch:
                raise KeyError("Batch must include `y_ids` when lambda_ce is non-zero.")
            y_ids = batch["y_ids"].to(device=device, dtype=torch.long)
            y_ids_last = y_ids[is_last]
            if y_ids_last.ne(ignore_index).any():
                loss_ce = F.cross_entropy(
                    student_logits.view(-1, student_logits.size(-1)),
                    y_ids_last.view(-1),
                    ignore_index=ignore_index,
                )

        if use_kl:
            if "teacher_logits" not in batch:
                raise KeyError("Batch must include `teacher_logits` when lambda_kl is non-zero.")
            teacher_logits = batch["teacher_logits"].to(device=device, dtype=torch.float32)
            if teacher_logits.ndim != 3:
                raise ValueError(
                    f"teacher_logits must have shape [B, y_len, vocab], got {tuple(teacher_logits.shape)}."
                )
            if teacher_logits.size(0) != B or teacher_logits.size(1) != y_len:
                raise ValueError(
                    "teacher_logits shape mismatch: "
                    f"expected first dims {(B, y_len)}, got {tuple(teacher_logits.shape[:2])}."
                )

            teacher_logits_last = teacher_logits[is_last]
            if teacher_logits_last.size(-1) != student_logits.size(-1):
                raise ValueError(
                    "Vocab mismatch between teacher and student logits: "
                    f"{teacher_logits_last.size(-1)} vs {student_logits.size(-1)}."
                )

            valid_tokens = y_valid[is_last]
            if valid_tokens.any():
                student_log_probs = F.log_softmax(student_logits, dim=-1)
                teacher_probs = F.softmax(teacher_logits_last, dim=-1)
                token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
                loss_kl = token_kl.masked_select(valid_tokens).mean()

    # ------------------------------------------------------------
    # Total
    # ------------------------------------------------------------
    loss_total = (
        cfg.lambda_fm * loss_fm
        + cfg.lambda_anchor * loss_anchor
        + cfg.lambda_ce * loss_ce
        + cfg.lambda_kl * loss_kl
    )

    return {
        "loss_total": loss_total,
        "loss_fm": loss_fm.detach(),
        "loss_anchor": loss_anchor.detach(),
        "loss_ce": loss_ce.detach(),
        "loss_kl": loss_kl.detach(),
        "t_mean": t.detach().mean(),
        "k_mean": k.detach().float().mean(),
    }


# ============================================================
# Main training loop
# ============================================================

def train_main(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------
    # Load dataset + collate_fn
    # ------------------------------------------------------------
    dataset = FlowHiddenStateDataset(
        data_root=cfg.data_root,
        pad_token_id=cfg.pad_token_id,
        fixed_total_length=cfg.max_length,
        include_padded_y_in_loss=cfg.include_padded_y_in_loss,
    )
    if dataset.num_anchor_layers != cfg.num_intervals:
        raise ValueError(
            f"cfg.num_intervals={cfg.num_intervals} does not match dataset anchor count={dataset.num_anchor_layers}."
        )

    collate_fn = FlowBatchCollator(pad_token_id=dataset.pad_token_id, ignore_index=dataset.ignore_index)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=(cfg.num_workers > 0),
    )
    print(
        f"[data] samples={len(dataset)} chunks={len(dataset.chunk_paths)} "
        f"anchors={dataset.num_anchor_layers} hidden={dataset.hidden_size}"
    )

    # ------------------------------------------------------------
    # Load student model
    # ------------------------------------------------------------
    model_cfg = ModelConfig(
        teacher_model_id=cfg.teacher_model_id,
        student_num_layers=cfg.student_num_layers,
        use_bidirectional_attention=cfg.use_bidirectional_attention,
        time_inject="film",
        layer_copy_mode=cfg.layer_copy_mode,
    )
    student = StudentCFD(model_cfg).to(device)

    init_ckpt = cfg.student_init_ckpt.strip()
    if init_ckpt:
        init_ckpt = os.path.expanduser(init_ckpt)
        if not os.path.isfile(init_ckpt):
            raise FileNotFoundError(f"student_init_ckpt not found: {init_ckpt}")
        load_student_init_checkpoint(student, init_ckpt)
    else:
        if cfg.require_student_init_ckpt:
            raise ValueError(
                "require_student_init_ckpt=True but student_init_ckpt is empty. "
                "Generate init checkpoint first and set cfg.student_init_ckpt."
            )
        print("[init] student_init_ckpt not provided. Falling back to teacher->student initialization.")
        student.init_from_teacher(device=device)
    student.train()

    # ------------------------------------------------------------
    # Infer hidden dim D for mask_init from student or from dataset sample
    # ------------------------------------------------------------
    # Option 1: student exposes hidden_dim attribute
    # D = student.hidden_dim
    D = dataset.hidden_size

    mask_init = MaskInit(hidden_dim=D, learnable=cfg.learnable_mask_init).to(device)

    # Optim: include mask_init params too
    params = list(mask_init.parameters())
    params += list(student.parameters())  # type: ignore[union-attr]

    optimizer = torch.optim.AdamW(
        params,
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    if getattr(cfg, "detect_anomaly", False):
        torch.autograd.set_detect_anomaly(True)

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    student.train()  # type: ignore[union-attr]
    mask_init.train()

    step = 0
    t_start = time.time()

    optimizer.zero_grad(set_to_none=True)

    for batch in loader:
        step += 1

        with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
            losses = compute_losses_skeleton(cfg, student, mask_init, batch)
            loss = losses["loss_total"] / cfg.grad_accum_steps

        scaler.scale(loss).backward()

        # grad accumulation
        if step % cfg.grad_accum_steps == 0:
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # logging
        if step % cfg.log_every == 0:
            elapsed = time.time() - t_start
            it_s = step / max(elapsed, 1e-6)
            msg = (
                f"[step {step:>6}] "
                f"loss={losses['loss_total'].item():.4f} "
                f"fm={losses['loss_fm'].item():.4f} "
                f"anc={losses['loss_anchor'].item():.4f} "
                f"ce={losses['loss_ce'].item():.4f} "
                f"kl={losses['loss_kl'].item():.4f} "
                f"t_mean={losses['t_mean'].item():.3f} "
                f"k_mean={losses['k_mean'].item():.2f} "
                f"it/s={it_s:.2f}"
            )
            print(msg)

        # save
        if step % cfg.save_every == 0:
            save_checkpoint(cfg, step, student, optimizer, scaler)  # type: ignore[arg-type]

        if step >= cfg.max_steps:
            break

    # final save
    save_checkpoint(cfg, step, student, optimizer, scaler)  # type: ignore[arg-type]


if __name__ == "__main__":
    cfg = TrainConfig(
        output_dir="./checkpoints_cfd",
        run_name="cfd_poc",
        max_steps=2000,
        log_every=20,
        save_every=200,
        amp=True,
        num_intervals=12,
        lambda_fm=1.0,
        lambda_anchor=1.0,
        lambda_ce=0.0,   # start with 0 for stability; enable later
        lambda_kl=0.0,
        z0_noise_std=1.0,
        learnable_mask_init=True,
        num_samples_per_pair=1,
    )
    train_main(cfg)
