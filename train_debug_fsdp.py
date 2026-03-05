import argparse
import functools
import os
import random
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from configs.model_config import ModelConfig
from configs.train_config import TrainConfig
from dataset.hidden_state_dataset import FlowBatchCollator, FlowHiddenStateDataset
from student_model.model import StudentCFD
from train import load_student_init_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FSDP debug training entrypoint (student_init_ckpt only)."
    )
    parser.add_argument("--data-root", type=str, default=TrainConfig.data_root)
    parser.add_argument("--student-init-ckpt", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./checkpoints_debug_fsdp")
    parser.add_argument("--run-name", type=str, default="debug_fsdp")

    parser.add_argument("--teacher-model-id", type=str, default=TrainConfig.teacher_model_id)
    parser.add_argument("--student-num-layers", type=int, default=TrainConfig.student_num_layers)
    parser.add_argument(
        "--use-bidirectional-attention",
        dest="use_bidirectional_attention",
        action="store_true",
        default=TrainConfig.use_bidirectional_attention,
    )
    parser.add_argument(
        "--causal-attention",
        dest="use_bidirectional_attention",
        action="store_false",
    )

    parser.add_argument("--batch-size", type=int, default=1, help="Per-rank batch size.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=TrainConfig.max_length)
    parser.add_argument("--pad-token-id", type=int, default=TrainConfig.pad_token_id)
    parser.add_argument("--include-padded-y-in-loss", action="store_true", default=TrainConfig.include_padded_y_in_loss)
    parser.add_argument("--max-samples", type=int, default=0)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--loss-mode", type=str, default="zero", choices=["zero", "l2"])
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)

    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--cpu-offload", action="store_true", default=False, help="Enable FSDP CPU parameter offload.")
    parser.add_argument("--activation-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-activation-checkpointing", dest="activation_checkpointing", action="store_false")
    return parser.parse_args()


def is_main_process(rank: int) -> bool:
    return rank == 0


def rank_print(rank: int, msg: str) -> None:
    if is_main_process(rank):
        print(msg, flush=True)


def setup_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "Distributed env vars are missing. Launch with torchrun, e.g. "
            "`torchrun --nproc_per_node=2 train_debug_fsdp.py ...`"
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def set_seed(seed: int, rank: int) -> None:
    base = seed + rank
    random.seed(base)
    torch.manual_seed(base)
    torch.cuda.manual_seed_all(base)


def build_debug_batch(
    batch: Dict[str, torch.Tensor],
    model_hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    x_mask = batch["x_mask"].to(device)
    y_mask = batch["y_mask"].to(device)
    teacher_anchors = batch["teacher_anchors"].to(device)
    teacher_x_layers = batch["teacher_x_layers"].to(device)

    B, M, y_len, D = teacher_anchors.shape
    _, M_x, x_len, D_x = teacher_x_layers.shape
    if M_x != M + 1:
        raise ValueError(f"Expected teacher_x_layers dim1=M+1={M+1}, got {M_x}.")
    if D != D_x:
        raise ValueError(f"Hidden mismatch: anchors D={D}, x_layers D={D_x}.")
    if model_hidden_size != D:
        raise ValueError(
            f"Model hidden_size={model_hidden_size} does not match dataset hidden_size={D}. "
            "Use matching teacher_model_id / dataset."
        )

    k = torch.randint(low=0, high=M, size=(B,), device=device)
    t = torch.rand(B, device=device, dtype=torch.float32)

    anchor_idx = k.view(B, 1, 1, 1).expand(-1, 1, y_len, D)
    z_t = torch.gather(teacher_anchors, dim=1, index=anchor_idx).squeeze(1).to(dtype=dtype)

    x_idx = k.view(B, 1, 1, 1).expand(-1, 1, x_len, D)
    x_t = torch.gather(teacher_x_layers, dim=1, index=x_idx).squeeze(1).to(dtype=dtype)

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
        return (v * 0.0).sum()
    mask = y_mask.unsqueeze(-1).to(v.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (v.pow(2) * mask).sum() / denom


class FSDPDebugModule(nn.Module):
    def __init__(self, student: StudentCFD, loss_mode: str):
        super().__init__()
        self.student = student
        self.loss_mode = loss_mode

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = next(self.student.parameters()).device
        dtype = next(self.student.parameters()).dtype
        debug_inputs = build_debug_batch(
            batch=batch,
            model_hidden_size=self.student.hidden_size,
            dtype=dtype,
            device=device,
        )
        v = self.student(
            z_y=debug_inputs["z_t"],
            t=debug_inputs["t"],
            x_ids=None,
            x_mask=debug_inputs["x_mask"],
            y_mask=debug_inputs["y_mask"],
            x_states=debug_inputs["x_t"],
        )
        if not torch.isfinite(v).all():
            raise RuntimeError("Non-finite velocity output detected.")
        loss = compute_debug_loss(v=v, y_mask=debug_inputs["y_mask"], mode=self.loss_mode)
        return {
            "loss_total": loss,
            "v_abs_mean": v.abs().mean().detach(),
            "y_valid": debug_inputs["y_mask"].sum().detach().to(torch.float32),
            "k_mean": debug_inputs["k"].float().mean().detach(),
        }


def apply_ckpting_if_needed(model: nn.Module, enabled: bool) -> None:
    if not enabled:
        return
    wrapper = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=wrapper,
        check_fn=lambda m: isinstance(m, LlamaDecoderLayer),
    )


def save_fsdp_checkpoint(
    args: argparse.Namespace,
    step: int,
    fsdp_model: FSDP,
    optimizer: AdamW,
    rank: int,
) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    full_optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        fsdp_model,
        StateDictType.FULL_STATE_DICT,
        full_state_cfg,
        full_optim_cfg,
    ):
        model_state = fsdp_model.state_dict()
        optim_state = FSDP.optim_state_dict(fsdp_model, optimizer)

    if is_main_process(rank):
        payload = {
            "step": step,
            "model": model_state,
            "optimizer": optim_state,
            "args": vars(args),
        }
        ckpt_path = os.path.join(args.output_dir, f"{args.run_name}_step{step}.pt")
        torch.save(payload, ckpt_path)
        print(f"[ckpt] saved: {ckpt_path}", flush=True)


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    set_seed(args.seed, rank)

    device = torch.device(f"cuda:{local_rank}")

    rank_print(rank, "=" * 80)
    rank_print(rank, "FSDP Debug Config")
    rank_print(rank, f"  world_size: {world_size}")
    rank_print(rank, f"  local_rank: {local_rank}")
    rank_print(rank, f"  batch_size(per-rank): {args.batch_size}")
    rank_print(rank, f"  bf16: {args.bf16}")
    rank_print(rank, f"  activation_checkpointing: {args.activation_checkpointing}")
    rank_print(rank, f"  student_init_ckpt: {args.student_init_ckpt}")
    rank_print(rank, "=" * 80)

    dataset = FlowHiddenStateDataset(
        data_root=args.data_root,
        pad_token_id=args.pad_token_id,
        fixed_total_length=args.max_length,
        include_padded_y_in_loss=args.include_padded_y_in_loss,
    )
    if args.max_samples > 0:
        max_n = min(args.max_samples, len(dataset))
        dataset = torch.utils.data.Subset(dataset, list(range(max_n)))

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    )
    collate_fn = FlowBatchCollator(pad_token_id=args.pad_token_id, ignore_index=-100)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    rank_print(rank, f"[data] samples={len(dataset)}")

    model_cfg = ModelConfig(
        teacher_model_id=args.teacher_model_id,
        student_num_layers=args.student_num_layers,
        use_bidirectional_attention=args.use_bidirectional_attention,
        time_inject="film",
    )
    student = StudentCFD(model_cfg)
    ckpt_path = os.path.expanduser(args.student_init_ckpt)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"student_init_ckpt not found: {ckpt_path}")
    load_student_init_checkpoint(student, ckpt_path)
    if args.bf16:
        student = student.to(dtype=torch.bfloat16)

    debug_module = FSDPDebugModule(student=student, loss_mode=args.loss_mode)
    apply_ckpting_if_needed(debug_module, enabled=args.activation_checkpointing)

    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )
    mp_policy: Optional[MixedPrecision] = None
    if args.bf16:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    cpu_offload = CPUOffload(offload_params=True) if args.cpu_offload else None

    torch.cuda.empty_cache()
    fsdp_model = FSDP(
        debug_module,
        auto_wrap_policy=auto_wrap,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        cpu_offload=cpu_offload,
        device_id=device,
        limit_all_gathers=True,
        use_orig_params=False,
    )
    fsdp_model.train()

    optimizer = AdamW(fsdp_model.parameters(), lr=args.lr)

    step = 0
    epoch = 0
    t_start = time.time()
    optimizer.zero_grad(set_to_none=True)

    while step < args.max_steps:
        sampler.set_epoch(epoch)
        epoch += 1

        for batch in loader:
            step += 1
            out = fsdp_model(batch)
            loss = out["loss_total"]
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(fsdp_model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % args.log_every == 0:
                with torch.no_grad():
                    loss_val = out["loss_total"].detach()
                    v_abs = out["v_abs_mean"].detach()
                    y_valid = out["y_valid"].detach()
                    k_mean = out["k_mean"].detach()

                    dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                    dist.all_reduce(v_abs, op=dist.ReduceOp.SUM)
                    dist.all_reduce(y_valid, op=dist.ReduceOp.SUM)
                    dist.all_reduce(k_mean, op=dist.ReduceOp.SUM)

                    inv_w = 1.0 / world_size
                    loss_val = loss_val * inv_w
                    v_abs = v_abs * inv_w
                    y_valid = y_valid * inv_w
                    k_mean = k_mean * inv_w

                    if is_main_process(rank):
                        elapsed = time.time() - t_start
                        it_s = step / max(elapsed, 1e-6)
                        print(
                            f"[step {step:03d}] "
                            f"loss={loss_val.item():.6f} "
                            f"v_abs_mean={v_abs.item():.6f} "
                            f"y_valid={int(y_valid.item())} "
                            f"k_mean={k_mean.item():.2f} "
                            f"it/s={it_s:.2f}",
                            flush=True,
                        )

            if step % args.save_every == 0:
                save_fsdp_checkpoint(args, step, fsdp_model, optimizer, rank)
            if step >= args.max_steps:
                break

    save_fsdp_checkpoint(args, step, fsdp_model, optimizer, rank)
    cleanup_distributed()


if __name__ == "__main__":
    main()
