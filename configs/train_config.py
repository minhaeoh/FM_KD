# /configs/train_config.py
from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrainConfig:
    # Data
    data_root: str = "/home/minhae/diffusion/FM_KD/data/collected_data"
    batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = True
    max_length: int = 1024
    pad_token_id: int = 0
    include_padded_y_in_loss: bool = False

    # CFD / FM
    # NOTE: this must match the number of anchor layers you saved (excluding embedding layer 0).
    num_intervals: int = 12
    num_samples_per_pair: int = 1  # R
    last_interval_sample_prob: float = 0.5

    # Optimization
    lr: float = 2e-5
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    # Runtime
    seed: int = 42
    device: str = "cuda"
    amp: bool = True
    grad_accum_steps: int = 1
    num_epochs: int = 5
    max_steps: int = 0
    log_every: int = 20
    save_every: int = 1000
    save_every_epochs: float = 0.5

    # Loss weights (POC: FM + Anchor only)
    lambda_fm: float = 1.0
    lambda_anchor: float = 1.0
    lambda_ce: float = 0.0
    lambda_kl: float = 0.0
    auto_balance_losses: bool = True
    balance_ema_decay: float = 0.95
    balance_update_every: int = 20
    balance_warmup_steps: int = 100
    balance_loss_eps: float = 1e-8
    balance_min_lambda: float = 1e-6
    balance_max_lambda: float = 1e3

    # z0 init
    z0_noise_std: float = 1.0
    learnable_mask_init: bool = True

    # Checkpointing
    output_dir: str = "./checkpoints_cfd"
    run_name: str = "cfd_poc"

    # Student model init
    teacher_model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    student_num_layers: int = 16
    use_bidirectional_attention: bool = True
    layer_copy_mode: str = "odd"
    student_init_ckpt: str = ""
    require_student_init_ckpt: bool = False
