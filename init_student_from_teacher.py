import argparse
import os
import time
from dataclasses import asdict
from typing import List, Optional

import torch

from configs.model_config import ModelConfig
from student_model.model import build_student_from_config


def parse_indices(text: Optional[str]) -> Optional[List[int]]:
    if text is None or text.strip() == "":
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize student weights from teacher once and save checkpoint."
    )
    parser.add_argument("--output", type=str, required=True, help="Output path for student init checkpoint.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--teacher-model-id", type=str, default=ModelConfig.teacher_model_id)
    parser.add_argument("--student-num-layers", type=int, default=ModelConfig.student_num_layers)
    parser.add_argument("--layer-copy-mode", type=str, default=ModelConfig.layer_copy_mode, choices=["odd", "even", "uniform"])
    parser.add_argument("--teacher-layer-indices", type=str, default=None, help="Comma-separated layer indices.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    teacher_layer_indices = parse_indices(args.teacher_layer_indices)
    model_cfg = ModelConfig(
        teacher_model_id=args.teacher_model_id,
        student_num_layers=args.student_num_layers,
        use_bidirectional_attention=args.use_bidirectional_attention,
        layer_copy_mode=args.layer_copy_mode,
        teacher_layer_indices=teacher_layer_indices,
    )

    print("=" * 80)
    print("Init Student From Teacher")
    print(f"  device: {device}")
    print(f"  teacher_model_id: {model_cfg.teacher_model_id}")
    print(f"  student_num_layers: {model_cfg.student_num_layers}")
    print(f"  layer_copy_mode: {model_cfg.layer_copy_mode}")
    print(f"  teacher_layer_indices: {model_cfg.teacher_layer_indices}")
    print(f"  output: {args.output}")
    print("=" * 80)

    t0 = time.time()
    student = build_student_from_config(model_cfg, device=device)
    build_sec = time.time() - t0

    # Save CPU state dict so later training can load without teacher.
    student_state_cpu = {k: v.detach().cpu() for k, v in student.state_dict().items()}
    payload = {
        "model": student_state_cpu,
        "model_cfg": asdict(model_cfg),
        "meta": {
            "created_from_teacher": model_cfg.teacher_model_id,
            "build_sec": build_sec,
            "device_used": str(device),
        },
    }

    out_path = os.path.expanduser(args.output)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(payload, out_path)

    print(f"[save] student init checkpoint: {out_path}")
    print(f"[save] build_sec={build_sec:.2f}")


if __name__ == "__main__":
    main()
