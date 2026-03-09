import argparse
import glob
import json
import os
import random
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from configs.generate_config import GenerateConfig
from configs.model_config import ModelConfig
from student_model.model import StudentCFD


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def select_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "fp32":
        return torch.float32
    if dtype_arg == "fp16":
        return torch.float16
    if dtype_arg == "bf16":
        return torch.bfloat16

    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def parse_teacher_layer_indices(text: str) -> Optional[List[int]]:
    text = text.strip()
    if not text:
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


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

    # Normalize wrapper prefixes from DDP/FSDP containers.
    if state and all(k.startswith("module.") for k in state.keys()):
        state = {k[len("module.") :]: v for k, v in state.items()}

    # FSDP checkpoints can store train-module state (e.g., student.* + mask_init.*).
    # Keep only student params when present and strip the wrapper prefix.
    if state:
        student_state = {k[len("student.") :]: v for k, v in state.items() if k.startswith("student.")}
        if student_state:
            state = student_state

    return state


def _extract_mask_init_vector(obj: Any) -> Optional[torch.Tensor]:
    if not isinstance(obj, dict):
        return None

    model_state = obj.get("model", None)
    if isinstance(model_state, dict):
        mask = model_state.get("mask_init.mask", None)
        if torch.is_tensor(mask):
            return mask.detach().cpu()

    direct_mask = obj.get("mask_init.mask", None)
    if torch.is_tensor(direct_mask):
        return direct_mask.detach().cpu()

    return None


def _has_zip_eocd(path: str) -> bool:
    # Torch's default checkpoint format is zip. Missing EOCD usually means truncated save.
    size = os.path.getsize(path)
    if size < 22:
        return False
    read_size = min(size, 70000)
    with open(path, "rb") as f:
        f.seek(-read_size, os.SEEK_END)
        tail = f.read(read_size)
    return b"PK\x05\x06" in tail


def _is_probably_corrupted_torch_zip(path: str) -> bool:
    with open(path, "rb") as f:
        magic = f.read(4)
    is_zip_like = magic == b"PK\x03\x04"
    if not is_zip_like:
        return False
    return not _has_zip_eocd(path)


def _resolve_checkpoint_path(path_or_dir: str) -> str:
    expanded = os.path.expanduser(path_or_dir)
    if os.path.isfile(expanded):
        return expanded
    if os.path.isdir(expanded):
        candidates = sorted(
            glob.glob(os.path.join(expanded, "step_*", "checkpoint.pt")),
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"No checkpoint.pt found under directory: {expanded}")
        return candidates[0]
    raise FileNotFoundError(f"checkpoint path not found: {expanded}")


def load_student_checkpoint(student: StudentCFD, ckpt_path: str) -> Tuple[Optional[torch.Tensor], str]:
    if _is_probably_corrupted_torch_zip(ckpt_path):
        raise RuntimeError(
            "Checkpoint appears corrupted/truncated (zip central directory missing). "
            f"path={ckpt_path} size={os.path.getsize(ckpt_path)} bytes. "
            "Try another step checkpoint."
        )

    try:
        payload = torch.load(ckpt_path, map_location="cpu")
    except RuntimeError as e:
        msg = str(e)
        if "failed finding central directory" in msg:
            run_dir = os.path.dirname(ckpt_path)
            candidates = sorted(glob.glob(os.path.join(run_dir, "step_*", "checkpoint.pt")), reverse=True)[:5]
            raise RuntimeError(
                "Failed to read checkpoint archive (likely incomplete/corrupted file). "
                f"path={ckpt_path}. "
                f"Nearby candidates={candidates if candidates else '[]'}"
            ) from e
        raise

    state = _extract_student_state_dict(payload)
    missing, unexpected = student.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch detected. "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    mask_init = _extract_mask_init_vector(payload)
    mask_msg = "found" if mask_init is not None else "not_found"
    return mask_init, mask_msg



def sample_tokens_from_logits(logits: torch.Tensor, temperature: float = 0.0, top_k: int = 0) -> torch.Tensor:
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1)

    scaled = logits / temperature
    if top_k > 0:
        k = min(top_k, scaled.size(-1))
        topk_vals, topk_idx = torch.topk(scaled, k=k, dim=-1)
        probs = torch.softmax(topk_vals, dim=-1)
        sampled_local = torch.multinomial(probs.view(-1, k), num_samples=1).view(*probs.shape[:-1], 1)
        return torch.gather(topk_idx, dim=-1, index=sampled_local).squeeze(-1)

    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(*probs.shape[:-1])


@torch.no_grad()
def generate_with_flow(
    model: StudentCFD,
    tokenizer: AutoTokenizer,
    prompt: str,
    gen_len: int,
    ode_steps: int,
    temperature: float,
    top_k: int,
    device: torch.device,
    param_dtype: torch.dtype,
    mask_init: Optional[torch.Tensor] = None,
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt")
    x_ids = encoded["input_ids"].to(device)
    x_mask = encoded.get("attention_mask", torch.ones_like(x_ids)).to(device)

    bsz = 1
    y_mask = torch.ones((bsz, gen_len), dtype=torch.long, device=device)
    z = torch.randn((bsz, gen_len, model.hidden_size), device=device, dtype=param_dtype)
    if mask_init is not None:
        z = z + mask_init.to(device=device, dtype=param_dtype).view(1, 1, -1)

    times = torch.linspace(0.0, 1.0, steps=ode_steps + 1, device=device, dtype=torch.float32)
    for i in range(ode_steps):
        t_cur = times[i].expand(bsz)
        t_next = times[i + 1].expand(bsz)
        dt = (t_next - t_cur).view(bsz, 1, 1).to(param_dtype)
        v = model(z_y=z, t=t_cur, x_ids=x_ids, x_mask=x_mask, y_mask=y_mask)
        z = z + dt * v

    logits = model.decode_logits(z).to(torch.float32)
    y_ids = sample_tokens_from_logits(logits, temperature=temperature, top_k=top_k)
    return tokenizer.batch_decode(y_ids, skip_special_tokens=True)[0]


def build_prompt(question: str, add_cot_trigger: bool) -> str:
    base = (
        "You are solving a grade-school math problem. "
        "Provide a concise final answer after your reasoning.\n\n"
        f"Question: {question}\n"
    )
    if add_cot_trigger:
        return base + "Answer: Let's think step by step."
    return base + "Answer:"


def parse_args() -> argparse.Namespace:
    defaults = GenerateConfig()
    parser = argparse.ArgumentParser(description="Run one-sample GSM8K generation from a saved student checkpoint.")

    parser.add_argument("--checkpoint-path", type=str, default=defaults.checkpoint_path)
    parser.add_argument("--teacher-model-id", type=str, default=defaults.teacher_model_id)
    parser.add_argument("--student-num-layers", type=int, default=defaults.student_num_layers)
    parser.add_argument("--use-bidirectional-attention", dest="use_bidirectional_attention", action="store_true", default=defaults.use_bidirectional_attention)
    parser.add_argument("--causal-attention", dest="use_bidirectional_attention", action="store_false")
    parser.add_argument("--layer-copy-mode", type=str, default=defaults.layer_copy_mode, choices=["odd", "even", "uniform"])

    parser.add_argument("--time-embedding-dim", type=int, default=defaults.time_embedding_dim)
    parser.add_argument("--time-num-frequencies", type=int, default=defaults.time_num_frequencies)
    parser.add_argument("--time-inject", type=str, default=defaults.time_inject, choices=["film", "none"])
    parser.add_argument("--velocity-head-init", type=str, default=defaults.velocity_head_init, choices=["identity", "xavier"])
    parser.add_argument("--teacher-layer-indices", type=str, default=defaults.teacher_layer_indices)

    parser.add_argument("--device", type=str, default=defaults.device, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--dtype", type=str, default=defaults.dtype, choices=["auto", "fp32", "fp16", "bf16"])

    parser.add_argument("--gen-len", type=int, default=defaults.gen_len)
    parser.add_argument("--ode-steps", type=int, default=defaults.ode_steps)
    parser.add_argument("--temperature", type=float, default=defaults.temperature)
    parser.add_argument("--top-k", type=int, default=defaults.top_k)

    parser.add_argument("--gsm8k-split", type=str, default=defaults.gsm8k_split)
    parser.add_argument("--sample-index", type=int, default=defaults.sample_index)
    parser.add_argument("--add-cot-trigger", dest="add_cot_trigger", action="store_true", default=defaults.add_cot_trigger)
    parser.add_argument("--no-cot-trigger", dest="add_cot_trigger", action="store_false")

    parser.add_argument("--output-path", type=str, default=defaults.output_path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = GenerateConfig(**vars(args))

    if not cfg.checkpoint_path:
        raise ValueError("--checkpoint-path is required.")

    ckpt_path = _resolve_checkpoint_path(cfg.checkpoint_path)

    set_seed(cfg.seed)
    device = choose_device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    model_dtype = select_dtype(cfg.dtype, device)

    teacher_layer_indices = parse_teacher_layer_indices(cfg.teacher_layer_indices)
    model_cfg = ModelConfig(
        teacher_model_id=cfg.teacher_model_id,
        student_num_layers=cfg.student_num_layers,
        time_embedding_dim=cfg.time_embedding_dim,
        time_num_frequencies=cfg.time_num_frequencies,
        time_inject=cfg.time_inject,
        use_bidirectional_attention=cfg.use_bidirectional_attention,
        velocity_head_init=cfg.velocity_head_init,
        layer_copy_mode=cfg.layer_copy_mode,
        teacher_layer_indices=teacher_layer_indices,
    )

    print("[load] building student model")
    print(f"[load] checkpoint={ckpt_path}")
    student = StudentCFD(model_cfg).to(device=device, dtype=model_dtype)
    mask_init, mask_msg = load_student_checkpoint(student, ckpt_path)
    print(f"[load] mask_init={mask_msg}")
    student.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[data] loading gsm8k split={cfg.gsm8k_split}")
    ds = load_dataset("openai/gsm8k", "main")
    split = ds[cfg.gsm8k_split]

    if cfg.sample_index < 0 or cfg.sample_index >= len(split):
        raise IndexError(f"sample_index={cfg.sample_index} is out of range for split size={len(split)}")

    sample = split[cfg.sample_index]
    question = sample["question"]
    gold_answer = sample["answer"]
    prompt = build_prompt(question, cfg.add_cot_trigger)

    print(f"[run] sample_index={cfg.sample_index}")
    generated = generate_with_flow(
        model=student,
        tokenizer=tokenizer,
        prompt=prompt,
        gen_len=cfg.gen_len,
        ode_steps=cfg.ode_steps,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        device=device,
        param_dtype=model_dtype,
        mask_init=mask_init,
    )

    result = {
        "config": asdict(cfg),
        "question": question,
        "gold_answer": gold_answer,
        "prompt": prompt,
        "generated": generated,
    }

    print("=" * 80)
    print("[question]")
    print(question)
    print("=" * 80)
    print("[gold_answer]")
    print(gold_answer)
    print("=" * 80)
    print("[generated]")
    print(generated)
    print("=" * 80)

    if cfg.output_path:
        out_path = os.path.expanduser(cfg.output_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[save] {out_path}")


if __name__ == "__main__":
    main()
