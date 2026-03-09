import argparse
import functools
import os
import random
import time
import json
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
from train import AdaptiveLossWeightBalancer, MaskInit, compute_losses, load_student_init_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FSDP training entrypoint (teacher init separated via student_init_ckpt)."
    )
    parser.add_argument("--data-root", type=str, default=TrainConfig.data_root)
    parser.add_argument("--student-init-ckpt", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=TrainConfig.output_dir)
    parser.add_argument("--run-name", type=str, default=TrainConfig.run_name)

    parser.add_argument("--teacher-model-id", type=str, default=TrainConfig.teacher_model_id)
    parser.add_argument("--student-num-layers", type=int, default=TrainConfig.student_num_layers)
    parser.add_argument("--layer-copy-mode", type=str, default=TrainConfig.layer_copy_mode, choices=["odd", "even", "uniform"])
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

    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size, help="Per-rank batch size.")
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--max-length", type=int, default=TrainConfig.max_length)
    parser.add_argument("--pad-token-id", type=int, default=TrainConfig.pad_token_id)
    parser.add_argument("--include-padded-y-in-loss", action="store_true", default=TrainConfig.include_padded_y_in_loss)

    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--beta1", type=float, default=TrainConfig.betas[0])
    parser.add_argument("--beta2", type=float, default=TrainConfig.betas[1])
    parser.add_argument("--grad-clip", type=float, default=TrainConfig.grad_clip)
    parser.add_argument("--grad-accum-steps", type=int, default=TrainConfig.grad_accum_steps)
    parser.add_argument("--max-steps", type=int, default=TrainConfig.max_steps)
    parser.add_argument("--log-every", type=int, default=TrainConfig.log_every)
    parser.add_argument("--save-every", type=int, default=TrainConfig.save_every)

    parser.add_argument("--num-intervals", type=int, default=TrainConfig.num_intervals)
    parser.add_argument(
        "--last-interval-sample-prob",
        type=float,
        default=TrainConfig.last_interval_sample_prob,
        help="Probability of forcing interval k=M-1 to densify CE/KL supervision.",
    )
    parser.add_argument("--lambda-fm", type=float, default=TrainConfig.lambda_fm)
    parser.add_argument("--lambda-anchor", type=float, default=TrainConfig.lambda_anchor)
    parser.add_argument("--lambda-ce", type=float, default=TrainConfig.lambda_ce)
    parser.add_argument("--lambda-kl", type=float, default=TrainConfig.lambda_kl)
    parser.add_argument("--z0-noise-std", type=float, default=TrainConfig.z0_noise_std)
    parser.add_argument("--learnable-mask-init", action="store_true", default=TrainConfig.learnable_mask_init)
    parser.add_argument("--auto-balance-losses", dest="auto_balance_losses", action="store_true", default=TrainConfig.auto_balance_losses)
    parser.add_argument("--no-auto-balance-losses", dest="auto_balance_losses", action="store_false")
    parser.add_argument("--balance-ema-decay", type=float, default=TrainConfig.balance_ema_decay)
    parser.add_argument("--balance-update-every", type=int, default=TrainConfig.balance_update_every)
    parser.add_argument("--balance-warmup-steps", type=int, default=TrainConfig.balance_warmup_steps)
    parser.add_argument("--balance-loss-eps", type=float, default=TrainConfig.balance_loss_eps)
    parser.add_argument("--balance-min-lambda", type=float, default=TrainConfig.balance_min_lambda)
    parser.add_argument("--balance-max-lambda", type=float, default=TrainConfig.balance_max_lambda)

    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--cpu-offload", action="store_true", default=False, help="Enable FSDP CPU parameter offload.")
    parser.add_argument("--activation-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-activation-checkpointing", dest="activation_checkpointing", action="store_false")
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.add_argument("--wandb-entity", type=str, default="cml_dit")
    parser.add_argument("--wandb-project", type=str, default="train")
    parser.add_argument("--wandb-run-name", type=str, default="")
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
            "`torchrun --nproc_per_node=2 train_fsdp.py ...`"
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


def build_train_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        max_length=args.max_length,
        pad_token_id=args.pad_token_id,
        include_padded_y_in_loss=args.include_padded_y_in_loss,
        num_intervals=args.num_intervals,
        last_interval_sample_prob=args.last_interval_sample_prob,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        grad_clip=args.grad_clip,
        seed=args.seed,
        device="cuda",
        amp=False,
        grad_accum_steps=args.grad_accum_steps,
        max_steps=args.max_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        lambda_fm=args.lambda_fm,
        lambda_anchor=args.lambda_anchor,
        lambda_ce=args.lambda_ce,
        lambda_kl=args.lambda_kl,
        auto_balance_losses=args.auto_balance_losses,
        balance_ema_decay=args.balance_ema_decay,
        balance_update_every=args.balance_update_every,
        balance_warmup_steps=args.balance_warmup_steps,
        balance_loss_eps=args.balance_loss_eps,
        balance_min_lambda=args.balance_min_lambda,
        balance_max_lambda=args.balance_max_lambda,
        z0_noise_std=args.z0_noise_std,
        learnable_mask_init=args.learnable_mask_init,
        output_dir=args.output_dir,
        run_name=args.run_name,
        teacher_model_id=args.teacher_model_id,
        student_num_layers=args.student_num_layers,
        use_bidirectional_attention=args.use_bidirectional_attention,
        layer_copy_mode=args.layer_copy_mode,
        student_init_ckpt=args.student_init_ckpt,
        require_student_init_ckpt=True,
    )


class FSDPTrainModule(nn.Module):
    def __init__(self, cfg: TrainConfig, student: StudentCFD, mask_init: MaskInit):
        super().__init__()
        self.cfg = cfg
        self.student = student
        self.mask_init = mask_init

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return compute_losses(self.cfg, self.student, self.mask_init, batch)


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
    cfg: TrainConfig,
    step: int,
    fsdp_model: FSDP,
    optimizer: AdamW,
    rank: int,
) -> None:
    ckpt_dir = os.path.join(cfg.output_dir, cfg.run_name, f"step_{step:07d}")
    os.makedirs(ckpt_dir, exist_ok=True)
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
        cfg_dict = asdict(cfg)
        payload = {
            "step": step,
            "model": model_state,
            "optimizer": optim_state,
            "cfg": cfg_dict,
        }
        ckpt_path = os.path.join(ckpt_dir, "checkpoint.pt")
        cfg_path = os.path.join(ckpt_dir, "config.json")
        torch.save(payload, ckpt_path)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, ensure_ascii=False, indent=2)
        print(f"[ckpt] saved: {ckpt_path}", flush=True)


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    set_seed(args.seed, rank)

    device = torch.device(f"cuda:{local_rank}")
    cfg = build_train_config(args)
    ts_obj = [time.strftime("%Y%m%d_%H%M") if is_main_process(rank) else ""]
    dist.broadcast_object_list(ts_obj, src=0)
    run_timestamp = ts_obj[0]
    cfg.output_dir = os.path.join(cfg.output_dir, run_timestamp)

    rank_print(rank, "=" * 80)
    rank_print(rank, "FSDP Train Config")
    rank_print(rank, f"  world_size: {world_size}")
    rank_print(rank, f"  local_rank: {local_rank}")
    rank_print(rank, f"  batch_size(per-rank): {cfg.batch_size}")
    rank_print(rank, f"  bf16: {args.bf16}")
    rank_print(rank, f"  activation_checkpointing: {args.activation_checkpointing}")
    rank_print(rank, f"  student_init_ckpt: {cfg.student_init_ckpt}")
    rank_print(rank, f"  output_root: {cfg.output_dir}")
    rank_print(rank, f"  wandb: {args.wandb}")
    if args.wandb:
        rank_print(rank, f"  wandb_entity: {args.wandb_entity}")
        rank_print(rank, f"  wandb_project: {args.wandb_project}")
    rank_print(rank, "=" * 80)

    wandb_run = None
    if args.wandb and is_main_process(rank):
        try:
            import wandb  # type: ignore
        except ImportError as e:
            raise ImportError(
                "wandb is enabled but package is not installed. "
                "Install with `pip install wandb` or pass --no-wandb."
            ) from e

        run_name = args.wandb_run_name.strip() or cfg.run_name
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            config={
                **asdict(cfg),
                "world_size": world_size,
                "bf16": args.bf16,
                "activation_checkpointing": args.activation_checkpointing,
            },
        )

    dataset = FlowHiddenStateDataset(
        data_root=cfg.data_root,
        pad_token_id=cfg.pad_token_id,
        fixed_total_length=cfg.max_length,
        include_padded_y_in_loss=cfg.include_padded_y_in_loss,
    )
    teacher_logits_mode = getattr(dataset, "teacher_logits_mode", "none")
    dense_logits_count = getattr(dataset, "teacher_logits_dense_count", 0)
    topk_logits_count = getattr(dataset, "teacher_logits_topk_count", 0)

    if cfg.lambda_kl != 0.0 and teacher_logits_mode not in ("dense", "topk"):
        rank_print(
            rank,
            "[warn] lambda_kl is non-zero but teacher_logits are not available in a supported format "
            f"(mode={teacher_logits_mode}, dense={dense_logits_count}, topk={topk_logits_count}, "
            f"total={len(dataset)}). Setting lambda_kl=0.0.",
        )
        cfg.lambda_kl = 0.0

    if dataset.num_anchor_layers != cfg.num_intervals:
        raise ValueError(
            f"cfg.num_intervals={cfg.num_intervals} does not match dataset anchor count={dataset.num_anchor_layers}."
        )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    collate_fn = FlowBatchCollator(pad_token_id=dataset.pad_token_id, ignore_index=dataset.ignore_index)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=(cfg.num_workers > 0),
    )
    rank_print(
        rank,
        f"[data] samples={len(dataset)} anchors={dataset.num_anchor_layers} hidden={dataset.hidden_size} "
        f"teacher_logits_mode={teacher_logits_mode} dense={dense_logits_count} topk={topk_logits_count}",
    )

    model_cfg = ModelConfig(
        teacher_model_id=cfg.teacher_model_id,
        student_num_layers=cfg.student_num_layers,
        use_bidirectional_attention=cfg.use_bidirectional_attention,
        time_inject="film",
        layer_copy_mode=cfg.layer_copy_mode,
    )
    student = StudentCFD(model_cfg)

    ckpt_path = os.path.expanduser(cfg.student_init_ckpt)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"student_init_ckpt not found: {ckpt_path}")
    load_student_init_checkpoint(student, ckpt_path)

    if args.bf16:
        student = student.to(dtype=torch.bfloat16)
    mask_init = MaskInit(hidden_dim=dataset.hidden_size, learnable=cfg.learnable_mask_init)
    if args.bf16:
        mask_init = mask_init.to(dtype=torch.bfloat16)

    train_module = FSDPTrainModule(cfg=cfg, student=student, mask_init=mask_init)
    apply_ckpting_if_needed(train_module, enabled=args.activation_checkpointing)

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
        train_module,
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

    optimizer = AdamW(
        fsdp_model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )
    loss_balancer = AdaptiveLossWeightBalancer(cfg)

    step = 0
    epoch = 0
    t_start = time.time()
    optimizer.zero_grad(set_to_none=True)

    use_bf16_autocast = bool(args.bf16 and device.type == "cuda")

    while step < cfg.max_steps:
        sampler.set_epoch(epoch)
        epoch += 1

        for batch in loader:
            step += 1
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16_autocast):
                losses = fsdp_model(batch)
            loss = losses["loss_total"] / cfg.grad_accum_steps
            loss.backward()

            if step % cfg.grad_accum_steps == 0:
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(fsdp_model.parameters(), cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                raw_fm = losses["loss_fm"].detach()
                raw_anchor = losses["loss_anchor"].detach()
                raw_ce = losses["loss_ce"].detach()
                raw_kl = losses["loss_kl"].detach()
                dist.all_reduce(raw_fm, op=dist.ReduceOp.SUM)
                dist.all_reduce(raw_anchor, op=dist.ReduceOp.SUM)
                dist.all_reduce(raw_ce, op=dist.ReduceOp.SUM)
                dist.all_reduce(raw_kl, op=dist.ReduceOp.SUM)
                inv_w = 1.0 / world_size
                raw_global = {
                    "fm": float((raw_fm * inv_w).item()),
                    "anchor": float((raw_anchor * inv_w).item()),
                    "ce": float((raw_ce * inv_w).item()),
                    "kl": float((raw_kl * inv_w).item()),
                }

            if step % cfg.log_every == 0:
                with torch.no_grad():
                    loss_total = losses["loss_total"].detach()
                    loss_fm = losses["loss_fm"].detach()
                    loss_anchor = losses["loss_anchor"].detach()
                    loss_ce = losses["loss_ce"].detach()
                    loss_kl = losses["loss_kl"].detach()
                    t_mean = losses["t_mean"].detach()
                    k_mean = losses["k_mean"].detach()

                    dist.all_reduce(loss_total, op=dist.ReduceOp.SUM)
                    dist.all_reduce(loss_fm, op=dist.ReduceOp.SUM)
                    dist.all_reduce(loss_anchor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(loss_ce, op=dist.ReduceOp.SUM)
                    dist.all_reduce(loss_kl, op=dist.ReduceOp.SUM)
                    dist.all_reduce(t_mean, op=dist.ReduceOp.SUM)
                    dist.all_reduce(k_mean, op=dist.ReduceOp.SUM)

                    inv_w = 1.0 / world_size
                    loss_total = loss_total * inv_w
                    loss_fm = loss_fm * inv_w
                    loss_anchor = loss_anchor * inv_w
                    loss_ce = loss_ce * inv_w
                    loss_kl = loss_kl * inv_w
                    t_mean = t_mean * inv_w
                    k_mean = k_mean * inv_w

                    if is_main_process(rank):
                        elapsed = time.time() - t_start
                        it_s = step / max(elapsed, 1e-6)
                        cur_epoch = epoch
                        weighted = loss_balancer.weighted(
                            {
                                "fm": float(loss_fm.item()),
                                "anchor": float(loss_anchor.item()),
                                "ce": float(loss_ce.item()),
                                "kl": float(loss_kl.item()),
                            }
                        )
                        log_payload = {
                            "train/loss_total": loss_total.item(),
                            "train/loss_fm": loss_fm.item(),
                            "train/loss_anchor": loss_anchor.item(),
                            "train/loss_ce": loss_ce.item(),
                            "train/loss_kl": loss_kl.item(),
                            "train/weighted_fm": weighted["fm"],
                            "train/weighted_anchor": weighted["anchor"],
                            "train/weighted_ce": weighted["ce"],
                            "train/weighted_kl": weighted["kl"],
                            "train/lambda_fm": float(cfg.lambda_fm),
                            "train/lambda_anchor": float(cfg.lambda_anchor),
                            "train/lambda_ce": float(cfg.lambda_ce),
                            "train/lambda_kl": float(cfg.lambda_kl),
                            "train/t_mean": t_mean.item(),
                            "train/k_mean": k_mean.item(),
                            "train/it_s": it_s,
                            "train/epoch": float(cur_epoch),
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/step": float(step),
                        }
                        print(
                            f"[step {step:>6}] "
                            f"epoch={cur_epoch} "
                            f"loss={log_payload['train/loss_total']:.4f} "
                            f"fm={log_payload['train/loss_fm']:.4f} "
                            f"anc={log_payload['train/loss_anchor']:.4f} "
                            f"ce={log_payload['train/loss_ce']:.4f} "
                            f"kl={log_payload['train/loss_kl']:.4f} "
                            f"wfm={log_payload['train/weighted_fm']:.4f} "
                            f"wanc={log_payload['train/weighted_anchor']:.4f} "
                            f"wce={log_payload['train/weighted_ce']:.4f} "
                            f"wkl={log_payload['train/weighted_kl']:.4f} "
                            f"lfm={log_payload['train/lambda_fm']:.3e} "
                            f"lanc={log_payload['train/lambda_anchor']:.3e} "
                            f"lce={log_payload['train/lambda_ce']:.3e} "
                            f"lkl={log_payload['train/lambda_kl']:.3e} "
                            f"t_mean={log_payload['train/t_mean']:.3f} "
                            f"k_mean={log_payload['train/k_mean']:.2f} "
                            f"it/s={log_payload['train/it_s']:.2f}",
                            flush=True,
                        )
                        if wandb_run is not None:
                            wandb_run.log(log_payload, step=step)

            loss_balancer.update(step, raw_global)

            if step % cfg.save_every == 0:
                save_fsdp_checkpoint(cfg, step, fsdp_model, optimizer, rank)
            if step >= cfg.max_steps:
                break

    save_fsdp_checkpoint(cfg, step, fsdp_model, optimizer, rank)
    if wandb_run is not None:
        wandb_run.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()
