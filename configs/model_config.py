# /configs/model_config.py
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    # Teacher
    teacher_model_id: str = "meta-llama/Llama-3.1-8B-Instruct"

    # Student backbone
    student_num_layers: int = 16

    # Time conditioning
    time_embedding_dim: int = 4096  # will be overridden by teacher hidden_size at runtime if needed
    time_num_frequencies: int = 64
    time_inject: str = "film"       # "film" or "none"

    # Attention behavior
    # Diffusion student default: bidirectional attention.
    use_bidirectional_attention: bool = True

    # Velocity head init
    velocity_head_init: str = "identity"  # "identity" or "xavier"

    # Layer-copy initialization
    # mapping: teacher 32L -> student 16L
    layer_copy_mode: str = "odd"          # "odd", "even", "uniform"
    teacher_layer_indices: Optional[List[int]] = None  # if set, overrides layer_copy_mode


def default_teacher_layer_map(num_teacher_layers: int, num_student_layers: int, mode: str) -> List[int]:
    """
    Returns teacher layer indices (0-based within transformer blocks, excluding embedding).
    Teacher blocks are typically 0..31 for 32-layer Llama.
    """
    if mode == "odd" and num_teacher_layers == 32 and num_student_layers == 16:
        return [2 * i + 1 for i in range(num_student_layers)]  # 1,3,...,31
    if mode == "even" and num_teacher_layers == 32 and num_student_layers == 16:
        return [2 * i for i in range(num_student_layers)]      # 0,2,...,30

    # generic "uniform-ish" mapping
    # spread student layers over teacher layers
    idxs = []
    for i in range(num_student_layers):
        t = int(round((i + 0.5) * num_teacher_layers / num_student_layers - 0.5))
        t = max(0, min(num_teacher_layers - 1, t))
        idxs.append(t)
    return idxs
