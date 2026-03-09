import argparse
import os
import random
import time
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from configs.model_config import ModelConfig
from configs.train_config import TrainConfig
from dataset.hidden_state_dataset import FlowBatchCollator, FlowHiddenStateDataset
from student_model.model import StudentCFD, build_student_from_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug training loop: verify dataset + student forward/backward wiring."
    )
    parser.add_argument("--data-root", type=str, default=TrainConfig.data_root)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=TrainConfig.max_length)
    parser.add_argument("--pad-token-id", type=int, default=TrainConfig.pad_token_id)
    parser.add_argument(
        "--include-padded-y-in-loss",
        action="store_true",
        default=TrainConfig.include_padded_y_in_loss,
    )
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--loss-mode", type=str, default="l2", choices=["l2", "zero"])
    parser.add_argument("--max-samples", type=int, default=0, help="If >0, only use first N samples.")

    parser.add_argument("--teacher-model-id", type=str, default=ModelConfig.teacher_model_id)
    parser.add_argument("--student-num-layers", type=int, default=ModelConfig.student_num_layers)
    parser.add_argument(
        "--use-bidirectional-attention",
        dest="use_bidirectional_attention",
        action="store_true",
        default=ModelConfig.use_bidirectional_attention,
    )
    parser.add_argument(
        "--causal-attention",
        dest="use_bidirectional_attention",
        action="store_false",
    )
    parser.add_argument(
        "--skip-teacher-init",
        dest="skip_teacher_init",
        action="store_true",
        default=False,
        help="Skip teacher load/copy and build StudentCFD only.",
    )
    parser.add_argument(
        "--student-init-ckpt",
        type=str,
        default="",
        help="Path to pre-initialized student checkpoint (teacher-free startup).",
    )
    return parser.parse_args()


def _extract_student_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint payload must be a dict, got {type(obj)}.")

    if obj and all(torch.is_tensor(v) for v in obj.values()):
        state = obj
    else:
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

    if state and all(k.startswith("module.") for k in state.keys()):
        state = {k[len("module.") :]: v for k, v in state.items()}

    if state:
        student_state = {k[len("student.") :]: v for k, v in state.items() if k.startswith("student.")}
        if student_state:
            state = student_state
    return state


def load_student_init_checkpoint(model: StudentCFD, ckpt_path: str) -> None:
    payload = torch.load(ckpt_path, map_location="cpu")
    state = _extract_student_state_dict(payload)
    model.load_state_dict(state, strict=True)
    print(f"[init] loaded student_init_ckpt: {ckpt_path}")


def build_debug_batch(
    batch: Dict[str, torch.Tensor],
    model: StudentCFD,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    x_mask = batch["x_mask"].to(device)
    y_mask = batch["y_mask"].to(device)
    teacher_anchors = batch["teacher_anchors"].to(device)  # [B, M, y_len, D]
    teacher_x_layers = batch["teacher_x_layers"].to(device)  # [B, M+1, x_len, D]

    B, M, y_len, D = teacher_anchors.shape
    _, M_x, x_len, D_x = teacher_x_layers.shape
    if M_x != M + 1:
        raise ValueError(f"Expected teacher_x_layers dim1=M+1={M+1}, got {M_x}.")
    if D != D_x:
        raise ValueError(f"Hidden mismatch: anchors D={D}, x_layers D={D_x}.")
    if model.hidden_size != D:
        raise ValueError(
            f"Model hidden_size={model.hidden_size} does not match dataset hidden_size={D}. "
            "Use matching teacher_model_id / dataset."
        )

    k = torch.randint(low=0, high=M, size=(B,), device=device)
    t = torch.rand(B, device=device, dtype=torch.float32)

    anchor_idx = k.view(B, 1, 1, 1).expand(-1, 1, y_len, D)
    z_t = torch.gather(teacher_anchors, dim=1, index=anchor_idx).squeeze(1)

    x_idx = k.view(B, 1, 1, 1).expand(-1, 1, x_len, D)
    x_t = torch.gather(teacher_x_layers, dim=1, index=x_idx).squeeze(1)

    model_dtype = next(model.parameters()).dtype
    z_t = z_t.to(dtype=model_dtype)
    x_t = x_t.to(dtype=model_dtype)

    return {
        "z_t": z_t,
        "t": t,
        "x_t": x_t,
        "x_mask": x_mask,
        "y_mask": y_mask,
        "k": k,
    }


def compute_debug_loss(v: torch.Tensor, y_mask: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "zero":
        # Graph-attached zero loss for pure plumbing checks.
        return (v * 0.0).sum()

    mask = y_mask.unsqueeze(-1).to(v.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (v.pow(2) * mask).sum() / denom


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = choose_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    print("=" * 80)
    print("Debug Train Config")
    print(f"  device: {device}")
    print(f"  data_root: {args.data_root}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  max_steps: {args.max_steps}")
    print(f"  loss_mode: {args.loss_mode}")
    print(f"  skip_teacher_init: {args.skip_teacher_init}")
    print(f"  student_init_ckpt: {args.student_init_ckpt if args.student_init_ckpt else '(none)'}")
    print("=" * 80)

    dataset = FlowHiddenStateDataset(
        data_root=args.data_root,
        pad_token_id=args.pad_token_id,
        fixed_total_length=args.max_length,
        include_padded_y_in_loss=args.include_padded_y_in_loss,
    )
    if args.max_samples > 0:
        max_n = min(args.max_samples, len(dataset))
        dataset = torch.utils.data.Subset(dataset, list(range(max_n)))

    collate_fn = FlowBatchCollator(pad_token_id=args.pad_token_id, ignore_index=-100)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    model_cfg = ModelConfig(
        teacher_model_id=args.teacher_model_id,
        student_num_layers=args.student_num_layers,
        use_bidirectional_attention=args.use_bidirectional_attention,
    )
    build_t0 = time.time()
    init_ckpt = args.student_init_ckpt.strip()
    if init_ckpt:
        init_ckpt = os.path.expanduser(init_ckpt)
        if not os.path.isfile(init_ckpt):
            raise FileNotFoundError(f"student_init_ckpt not found: {init_ckpt}")
        model = StudentCFD(model_cfg).to(device)
        load_student_init_checkpoint(model, init_ckpt)
    elif args.skip_teacher_init:
        model = StudentCFD(model_cfg).to(device)
    else:
        model = build_student_from_config(model_cfg, device=device)
    model.train()
    build_t1 = time.time()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(
        f"[data] samples={len(dataset)} "
        f"anchors={getattr(dataset, 'num_anchor_layers', 'n/a')} "
        f"hidden={getattr(dataset, 'hidden_size', 'n/a')}"
    )
    print(
        f"[model] hidden={model.hidden_size} layers={model.num_layers} "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )
    print(f"[model] build_sec={build_t1 - build_t0:.2f}")

    t0 = time.time()
    step = 0
    for batch in loader:
        step += 1
        debug_inputs = build_debug_batch(batch=batch, model=model, device=device)
        v = model(
            z_y=debug_inputs["z_t"],
            t=debug_inputs["t"],
            x_ids=None,
            x_mask=debug_inputs["x_mask"],
            y_mask=debug_inputs["y_mask"],
            x_states=debug_inputs["x_t"],
        )

        if not torch.isfinite(v).all():
            raise RuntimeError(f"Non-finite velocity output at step {step}.")

        loss = compute_debug_loss(v=v, y_mask=debug_inputs["y_mask"], mode=args.loss_mode)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = 0.0
        if args.grad_clip > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item())
        optimizer.step()

        with torch.no_grad():
            v_abs = float(v.abs().mean().item())
            y_valid = int(debug_inputs["y_mask"].sum().item())
            k_mean = float(debug_inputs["k"].float().mean().item())
            print(
                f"[step {step:03d}] loss={float(loss.item()):.6f} "
                f"v_abs_mean={v_abs:.6f} y_valid={y_valid} k_mean={k_mean:.2f} "
                f"grad_norm={grad_norm:.6f} v_shape={tuple(v.shape)}"
            )

        if step >= args.max_steps:
            break

    elapsed = time.time() - t0
    print("=" * 80)
    print(f"Completed debug run: steps={step}, elapsed_sec={elapsed:.2f}")
    print("Dataset/model wiring is functional if no exception occurred.")
    print("=" * 80)


if __name__ == "__main__":
    main()
