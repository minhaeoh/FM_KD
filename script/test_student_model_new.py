#!/usr/bin/env python
import argparse
import time
import sys
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_config import ModelConfig
from student_model.model import StudentCFD, build_student_from_config


def parse_indices(text: Optional[str]) -> Optional[List[int]]:
    if text is None or text.strip() == "":
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test student_model/model_new.py construction and forward pass."
    )
    parser.add_argument("--teacher-model-id", type=str, default=ModelConfig.teacher_model_id)
    parser.add_argument("--student-num-layers", type=int, default=ModelConfig.student_num_layers)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--x-len", type=int, default=16)
    parser.add_argument("--y-len", type=int, default=32)
    parser.add_argument("--pad-y", type=int, default=0, help="Number of padded y tokens at sequence end.")

    parser.add_argument("--time-embedding-dim", type=int, default=ModelConfig.time_embedding_dim)
    parser.add_argument("--time-num-frequencies", type=int, default=ModelConfig.time_num_frequencies)
    parser.add_argument("--time-inject", type=str, default=ModelConfig.time_inject, choices=["film", "none"])

    parser.add_argument("--velocity-head-init", type=str, default=ModelConfig.velocity_head_init, choices=["identity", "xavier"])
    parser.add_argument("--layer-copy-mode", type=str, default=ModelConfig.layer_copy_mode, choices=["odd", "even", "uniform"])
    parser.add_argument("--teacher-layer-indices", type=str, default=None, help="Comma-separated indices, e.g. '1,3,5,...'.")
    parser.add_argument(
        "--use-bidirectional-attention",
        dest="use_bidirectional_attention",
        action="store_true",
        default=ModelConfig.use_bidirectional_attention,
        help="Enable bidirectional attention (default follows ModelConfig).",
    )
    parser.add_argument(
        "--causal-attention",
        dest="use_bidirectional_attention",
        action="store_false",
        help="Disable bidirectional attention and use causal masking.",
    )
    parser.add_argument("--skip-teacher-init", action="store_true", help="Build StudentCFD only, skip init_from_teacher copy.")

    parser.add_argument("--anchor-layers", type=str, default=None, help="Comma-separated layers for anchor-state check.")
    parser.add_argument("--include-final", action="store_true", help="Return final y hidden states in anchor output.")

    parser.add_argument("--run-text-demo", action="store_true", help="Run a simple text generation demo after model checks.")
    parser.add_argument("--prompt", type=str, default="Explain why the sky looks blue in one sentence.")
    parser.add_argument("--gen-len", type=int, default=32, help="Generated y-token length for demo.")
    parser.add_argument("--ode-steps", type=int, default=32, help="Euler steps for t: 1 -> 0 integration.")
    parser.add_argument("--temperature", type=float, default=0.0, help="0.0 means greedy argmax.")
    parser.add_argument("--top-k", type=int, default=0, help="If >0, sample only from top-k logits.")
    return parser.parse_args()

def sample_tokens_from_logits(logits: torch.Tensor, temperature: float = 0.0, top_k: int = 0) -> torch.Tensor:
    # logits: [B, S, V]
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1)

    scaled = logits / temperature
    if top_k > 0:
        k = min(top_k, scaled.size(-1))
        topk_vals, topk_idx = torch.topk(scaled, k=k, dim=-1)
        probs = torch.softmax(topk_vals, dim=-1)
        sampled_local = torch.multinomial(probs.view(-1, k), num_samples=1).view(*probs.shape[:-1], 1)
        sampled = torch.gather(topk_idx, dim=-1, index=sampled_local).squeeze(-1)
        return sampled

    probs = torch.softmax(scaled, dim=-1)
    sampled = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(*probs.shape[:-1])
    return sampled


@torch.no_grad()
def generate_demo_text(
    model: StudentCFD,
    teacher_model_id: str,
    prompt: str,
    gen_len: int,
    ode_steps: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> str:
    if gen_len <= 0:
        raise ValueError(f"gen_len must be > 0, got {gen_len}.")
    if ode_steps <= 0:
        raise ValueError(f"ode_steps must be > 0, got {ode_steps}.")

    tokenizer = AutoTokenizer.from_pretrained(teacher_model_id)
    encoded = tokenizer(prompt, return_tensors="pt")
    x_ids = encoded["input_ids"].to(device)
    if "attention_mask" in encoded:
        x_mask = encoded["attention_mask"].to(device)
    else:
        x_mask = torch.ones_like(x_ids, dtype=torch.long, device=device)

    bsz = 1
    y_mask = torch.ones((bsz, gen_len), dtype=torch.long, device=device)
    dtype = next(model.parameters()).dtype
    z = torch.randn((bsz, gen_len, model.hidden_size), device=device, dtype=dtype)

    # Rectified-flow style reverse-time Euler integration: t=1 -> 0.
    times = torch.linspace(1.0, 0.0, steps=ode_steps + 1, device=device, dtype=torch.float32)
    for i in range(ode_steps):
        t_cur = times[i].expand(bsz)
        t_next = times[i + 1].expand(bsz)
        dt = (t_next - t_cur).view(bsz, 1, 1).to(dtype)
        v = model(z_y=z, t=t_cur, x_ids=x_ids, x_mask=x_mask, y_mask=y_mask)
        z = z + dt * v

    logits = model.decode_logits(z)
    y_ids = sample_tokens_from_logits(logits, temperature=temperature, top_k=top_k)
    gen_text = tokenizer.batch_decode(y_ids, skip_special_tokens=True)[0]
    return gen_text


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    print("=" * 80)
    print("Environment")
    print(f"  torch: {torch.__version__}")
    print(f"  cuda_available: {torch.cuda.is_available()}")
    print(f"  device: {device}")
    if device.type == "cuda":
        print(f"  gpu: {torch.cuda.get_device_name(device)}")
    print("=" * 80)

    layer_indices = parse_indices(args.teacher_layer_indices)
    model_cfg = ModelConfig(
        teacher_model_id=args.teacher_model_id,
        student_num_layers=args.student_num_layers,
        time_embedding_dim=args.time_embedding_dim,
        time_num_frequencies=args.time_num_frequencies,
        time_inject=args.time_inject,
        use_bidirectional_attention=args.use_bidirectional_attention,
        velocity_head_init=args.velocity_head_init,
        layer_copy_mode=args.layer_copy_mode,
        teacher_layer_indices=layer_indices,
    )

    print("ModelConfig")
    print(f"  teacher_model_id: {model_cfg.teacher_model_id}")
    print(f"  student_num_layers: {model_cfg.student_num_layers}")
    print(f"  use_bidirectional_attention: {model_cfg.use_bidirectional_attention}")
    print(f"  layer_copy_mode: {model_cfg.layer_copy_mode}")
    print(f"  teacher_layer_indices: {model_cfg.teacher_layer_indices}")
    print("=" * 80)

    t0 = time.time()
    if args.skip_teacher_init:
        print("Building StudentCFD (skip teacher weight copy)")
        model = StudentCFD(model_cfg).to(device)
        model.train()
    else:
        print("Building StudentCFD with teacher initialization")
        model = build_student_from_config(model_cfg, device=device)
    t1 = time.time()

    total_params = count_params(model)
    trainable_params = count_trainable_params(model)
    model_dtype = next(model.parameters()).dtype
    vocab_size = model.backbone.config.vocab_size

    print("Model Stats")
    print(f"  build_time_sec: {t1 - t0:.2f}")
    print(f"  hidden_size: {model.hidden_size}")
    print(f"  num_layers: {model.num_layers}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  param_dtype: {model_dtype}")
    print(f"  total_params: {total_params:,}")
    print(f"  trainable_params: {trainable_params:,}")
    print("=" * 80)

    model.eval()

    batch_size = args.batch_size
    x_len = args.x_len
    y_len = args.y_len
    pad_y = max(0, min(args.pad_y, y_len))

    x_ids = torch.randint(0, vocab_size, (batch_size, x_len), device=device)
    x_mask = torch.ones((batch_size, x_len), dtype=torch.long, device=device)
    y_mask = torch.ones((batch_size, y_len), dtype=torch.long, device=device)
    if pad_y > 0:
        y_mask[:, y_len - pad_y :] = 0

    z_y = torch.randn(batch_size, y_len, model.hidden_size, device=device, dtype=model_dtype)
    t = torch.rand(batch_size, device=device, dtype=torch.float32)

    with torch.no_grad():
        v_y = model(
            z_y=z_y,
            t=t,
            x_ids=x_ids,
            x_mask=x_mask,
            y_mask=y_mask,
        )

    expected_shape = (batch_size, y_len, model.hidden_size)
    if tuple(v_y.shape) != expected_shape:
        raise AssertionError(f"v_y shape mismatch: got {tuple(v_y.shape)}, expected {expected_shape}.")
    if not torch.isfinite(v_y).all():
        raise AssertionError("v_y contains non-finite values.")

    print("Forward Check")
    print(f"  v_y shape: {tuple(v_y.shape)}")
    print(f"  v_y dtype: {v_y.dtype}")
    print(f"  v_y finite: {bool(torch.isfinite(v_y).all().item())}")
    print("=" * 80)

    anchor_layers = parse_indices(args.anchor_layers)
    with torch.no_grad():
        out = model.forward_with_anchor_states(
            z_y=z_y,
            t=t,
            x_ids=x_ids,
            x_mask=x_mask,
            y_mask=y_mask,
            anchor_layers=anchor_layers,
            include_final=args.include_final,
        )

    if "v_y" not in out:
        raise AssertionError("forward_with_anchor_states output is missing key: 'v_y'.")
    if tuple(out["v_y"].shape) != expected_shape:
        raise AssertionError(f"anchor v_y shape mismatch: got {tuple(out['v_y'].shape)}, expected {expected_shape}.")

    print("Anchor-State Check")
    print(f"  returned_keys: {sorted(out.keys())}")
    if anchor_layers is not None:
        for idx in anchor_layers:
            key = f"layer_{idx}"
            if key not in out:
                raise AssertionError(f"Missing anchor key: {key}")
            if tuple(out[key].shape) != expected_shape:
                raise AssertionError(
                    f"{key} shape mismatch: got {tuple(out[key].shape)}, expected {expected_shape}."
                )
            if not torch.isfinite(out[key]).all():
                raise AssertionError(f"{key} contains non-finite values.")
            print(f"  {key} shape: {tuple(out[key].shape)}")
    if args.include_final:
        if "h_y_final" not in out:
            raise AssertionError("include_final=True but 'h_y_final' key is missing.")
        if tuple(out["h_y_final"].shape) != expected_shape:
            raise AssertionError(
                f"h_y_final shape mismatch: got {tuple(out['h_y_final'].shape)}, expected {expected_shape}."
            )
        print(f"  h_y_final shape: {tuple(out['h_y_final'].shape)}")
    print("=" * 80)

    if args.run_text_demo:
        print("Text Demo")
        demo_text = generate_demo_text(
            model=model,
            teacher_model_id=model_cfg.teacher_model_id,
            prompt=args.prompt,
            gen_len=args.gen_len,
            ode_steps=args.ode_steps,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
        print(f"  prompt: {args.prompt}")
        print(f"  output: {demo_text}")
        print("=" * 80)

    print("All checks passed.")


if __name__ == "__main__":
    main()
