from dataclasses import dataclass


@dataclass
class GenerateConfig:
    # Model / checkpoint
    checkpoint_path: str = ""
    teacher_model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    student_num_layers: int = 16
    use_bidirectional_attention: bool = True
    layer_copy_mode: str = "odd"

    # Time-conditioning architecture (must match training-time model shape)
    time_embedding_dim: int = 4096
    time_num_frequencies: int = 64
    time_inject: str = "film"
    velocity_head_init: str = "identity"
    teacher_layer_indices: str = ""

    # Runtime
    device: str = "auto"
    seed: int = 42
    dtype: str = "auto"  # auto | fp32 | fp16 | bf16

    # Generation
    gen_len: int = 128
    ode_steps: int = 64
    temperature: float = 0.0
    top_k: int = 0

    # Dataset / prompt
    gsm8k_split: str = "test"
    sample_index: int = 0
    add_cot_trigger: bool = True

    # Optional output save path (json)
    output_path: str = ""
