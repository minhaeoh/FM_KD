import os
import math
import json
import random
import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file
from safetensors import safe_open
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(dotenv_path="/home/minhae/diffusion/FM_KD/.env")
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# ============================================================
# Config
# ============================================================

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DATASET_ID = "meta-math/MetaMathQA"

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = "/home/minhae/diffusion/FM_KD/data/collected_data"  # new output

NUM_GPUS = 2
BATCH_SIZE = 4
SAVE_INTERVAL_SAMPLES = 100        # 100 samples per safetensors file
TARGET_SAMPLE_COUNT = 50000
RNG_SEED = 42

# Optional: save teacher final-layer logits for Y tokens (distillation target).
# NOTE: full-vocab logits are extremely large; top-k is the practical default.
SAVE_TEACHER_LOGITS = True
TEACHER_LOGITS_TOPK = 64         # <=0 saves full-vocab logits
TEACHER_LOGITS_DTYPE = torch.float16

# Use true token-length filtering
MAX_LENGTH = 1024                  # can increase if too few samples pass
ALLOW_AUTO_INCREASE_MAXLEN = True  # if not enough <= MAX_LENGTH, automatically increase
MAXLEN_INCREASE_TO = 1536          # fallback
# You can set MAX_LENGTH=1536 directly if you prefer.

# Save which layers?
# IMPORTANT: HF hidden_states includes embedding output at index 0 in most models.
# For Llama, you'll typically get 33 tensors: embedding + 32 layers (may include final norm output).
# Here we store EXACTLY what HF returns, but you can subset later.
TARGET_LAYERS = None  # None means save all returned hidden_states
# Example: TARGET_LAYERS = [1,4,7,10,...] but I'd keep all for reusability.
ANCHOR_LAYERS = [2,5,8,11,14,17,20,23,26,29,31,32]  
TARGET_LAYERS = [0]+ANCHOR_LAYERS  # ensure embedding layer (0) is included for start point(X) for student model while training.


SYSTEM_PROMPT = (
    "You are an expert mathematician. "
    "Solve the following math problem step-by-step. "
    "Ensure your reasoning is clear and logical. "
    "Conclude your final answer in the exact format: 'The answer is: [number]'."
)

def format_prompt(query: str, response: str) -> str:
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{response}<|eot_id|>"
    )

def format_query_prefix(query: str) -> str:
    # text up to assistant header, to compute q_len precisely in tokens
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

# ============================================================
# Length-based sample selection (token-accurate)
# ============================================================

def compute_token_lengths(tokenizer, dataset, max_probe: int = None):
    """
    Computes token lengths for full prompt and query-prefix only.
    Returns lists: full_len[i], q_len[i]
    NOTE: This can take time on full dataset; but it's a one-time preprocessing step.
    """
    full_lens = []
    q_lens = []

    n = len(dataset) if max_probe is None else min(len(dataset), max_probe)

    for i in tqdm(range(n), desc="Computing token lengths"):
        q = dataset[i]["query"]
        a = dataset[i]["response"]

        full_text = format_prompt(q, a)
        prefix = format_query_prefix(q)

        # IMPORTANT: add_special_tokens=False to avoid double BOS/EOS
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        q_ids = tokenizer.encode(prefix, add_special_tokens=False)

        full_lens.append(len(full_ids))
        q_lens.append(len(q_ids))

    return full_lens, q_lens

def select_indices_bucketed(
    full_lens,
    max_len: int,
    target_count: int,
    seed: int,
):
    """
    Bucket-balanced selection over token lengths.
    Only consider samples with full_len <= max_len.
    Buckets: [1..256], [257..384], [385..512], [513..640], [641..768], [769..896], [897..max_len]
    You can tweak bins freely.
    """
    random.seed(seed)

    bins = [256, 384, 512, 640, 768, 896, max_len]
    bucket_ids = [[] for _ in bins]

    for idx, L in enumerate(full_lens):
        if L > max_len:
            continue
        # find bucket
        for b, upper in enumerate(bins):
            if L <= upper:
                bucket_ids[b].append(idx)
                break

    # Determine per-bucket quota
    nonempty = [b for b in range(len(bins)) if len(bucket_ids[b]) > 0]
    if not nonempty:
        return []

    quota = target_count // len(nonempty)
    remainder = target_count - quota * len(nonempty)

    selected = []
    for b in nonempty:
        ids = bucket_ids[b]
        random.shuffle(ids)
        take = min(quota, len(ids))
        selected.extend(ids[:take])

    # Fill remainder from all remaining eligible samples
    if len(selected) < target_count:
        remaining = []
        selected_set = set(selected)
        for b in nonempty:
            for idx in bucket_ids[b]:
                if idx not in selected_set:
                    remaining.append(idx)
        random.shuffle(remaining)
        need = target_count - len(selected)
        selected.extend(remaining[:need])

    random.shuffle(selected)
    return selected[:target_count]

# ============================================================
# Worker: extraction
# ============================================================

def run_extraction(rank: int, world_size: int, selected_indices, max_len: int):
    device = torch.device(f"cuda:{rank}")
    print(f"[GPU {rank}] init")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        torch_dtype=torch.float16,
        output_hidden_states=True,
        device_map=None,
    ).to(device)
    model.eval()

    dataset = load_dataset(DATASET_ID, split="train")

    # select the 50k indices and shard them across GPUs
    dataset = dataset.select(selected_indices)

    total_size = len(dataset)
    chunk_size = math.ceil(total_size / world_size)
    start_idx = rank * chunk_size
    end_idx = min(start_idx + chunk_size, total_size)
    sub_dataset = dataset.select(range(start_idx, end_idx))
    print(f"[GPU {rank}] samples: {len(sub_dataset)} ({start_idx}..{end_idx})")

    gpu_out = os.path.join(OUTPUT_DIR, f"shard_{rank}")
    os.makedirs(gpu_out, exist_ok=True)

    save_buffer = {}
    buffered_samples = 0
    chunk_counter = 0
    batch_texts = []
    batch_q_lens = []
    batch_global_ids = []

    # We'll assign a new global id 0..TARGET_SAMPLE_COUNT-1 AFTER selection
    # so sample ids are contiguous and consistent for training.
    # global_id = start_idx + i within selected dataset.
    for i, item in enumerate(tqdm(sub_dataset, desc=f"GPU {rank}", position=rank)):
        q = item["query"]
        a = item["response"]

        full_text = format_prompt(q, a)
        prefix = format_query_prefix(q)

        # exact token length checks (avoid truncation completely)
        # add_special_tokens=False is critical
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        q_ids = tokenizer.encode(prefix, add_special_tokens=False)

        if len(full_ids) > max_len:
            # should not happen if selection was correct, but keep safe
            continue
        if len(q_ids) >= len(full_ids):
            continue

        global_id = start_idx + i  # contiguous within the selected 50k set
        batch_texts.append(full_text)
        batch_q_lens.append(len(q_ids))
        batch_global_ids.append(global_id)

        if len(batch_texts) == BATCH_SIZE or i == (len(sub_dataset) - 1):
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=False,         # IMPORTANT: no truncation
                add_special_tokens=False, # IMPORTANT
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            # outputs.hidden_states: tuple length = (#layers+1 maybe)
            hs = outputs.hidden_states  # tuple: each [B, seq, D]
            # stack -> [L, B, seq, D]
            all_layers = torch.stack(hs, dim=0)

            if TARGET_LAYERS is not None:
                all_layers = all_layers[TARGET_LAYERS]  # [Lsel, B, seq, D]

            # [B, L, seq, D]
            all_layers = all_layers.permute(1, 0, 2, 3).contiguous()

            # per sample save
            for b in range(len(batch_texts)):
                seq_len = int(inputs.attention_mask[b].sum().item())
                q_len = int(batch_q_lens[b])
                sid = int(batch_global_ids[b])

                # Safety checks
                if q_len >= seq_len:
                    continue
                if seq_len > max_len:
                    # should not happen (no truncation + selection), but safe
                    continue

                # Save full-sequence teacher signals for this sample.
                # hidden shape: (L, seq_len, D), input_ids shape: (seq_len,)
                hidden = all_layers[b, :, :seq_len, :].cpu()
                input_ids = inputs.input_ids[b, :seq_len].to(torch.long).cpu()
                qlen_t = torch.tensor([q_len], dtype=torch.long)

                save_buffer[f"sample_{sid}_hidden"] = hidden
                save_buffer[f"sample_{sid}_qlen"] = qlen_t
                save_buffer[f"sample_{sid}_input_ids"] = input_ids

                # Teacher final-layer LM logits for Y positions only (last interval endpoint).
                if SAVE_TEACHER_LOGITS:
                    y_logits = outputs.logits[b, q_len:seq_len, :].contiguous()
                    if TEACHER_LOGITS_TOPK > 0:
                        topk = min(int(TEACHER_LOGITS_TOPK), y_logits.size(-1))
                        topk_vals, topk_idx = torch.topk(y_logits, k=topk, dim=-1)
                        save_buffer[f"sample_{sid}_teacher_logits_topk_vals"] = (
                            topk_vals.to(dtype=TEACHER_LOGITS_DTYPE).cpu()
                        )
                        save_buffer[f"sample_{sid}_teacher_logits_topk_idx"] = topk_idx.to(torch.int32).cpu()
                    else:
                        save_buffer[f"sample_{sid}_teacher_logits"] = (
                            y_logits.to(dtype=TEACHER_LOGITS_DTYPE).cpu()
                        )
                buffered_samples += 1

            # flush chunk every SAVE_INTERVAL_SAMPLES samples
            if buffered_samples >= SAVE_INTERVAL_SAMPLES:
                save_path = os.path.join(gpu_out, f"chunk_{chunk_counter}.safetensors")
                save_file(save_buffer, save_path)
                save_buffer = {}
                buffered_samples = 0
                chunk_counter += 1

            # clear batch
            batch_texts, batch_q_lens, batch_global_ids = [], [], []
            del outputs, hs, all_layers, inputs
            torch.cuda.empty_cache()

    if len(save_buffer) > 0:
        save_path = os.path.join(gpu_out, f"chunk_{chunk_counter}_final.safetensors")
        save_file(save_buffer, save_path)

    print(f"[GPU {rank}] done")


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(RNG_SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

    dataset = load_dataset(DATASET_ID, split="train")
    print(f"Loaded dataset: {len(dataset)} samples")

    # 1) token-accurate length computation
    full_lens, q_lens = compute_token_lengths(tokenizer, dataset)

    # 2) filter + select 50k
    selected = select_indices_bucketed(
        full_lens=full_lens,
        max_len=MAX_LENGTH,
        target_count=TARGET_SAMPLE_COUNT,
        seed=RNG_SEED,
    )

    if len(selected) < TARGET_SAMPLE_COUNT:
        print(f"[warn] Only {len(selected)} samples fit max_len={MAX_LENGTH}")
        if ALLOW_AUTO_INCREASE_MAXLEN:
            print(f"[info] Increasing MAX_LENGTH to {MAXLEN_INCREASE_TO} and re-selecting...")
            selected = select_indices_bucketed(
                full_lens=full_lens,
                max_len=MAXLEN_INCREASE_TO,
                target_count=TARGET_SAMPLE_COUNT,
                seed=RNG_SEED,
            )
            if len(selected) < TARGET_SAMPLE_COUNT:
                raise RuntimeError(
                    f"Still insufficient samples: {len(selected)} < {TARGET_SAMPLE_COUNT} "
                    f"even with max_len={MAXLEN_INCREASE_TO}"
                )
            else:
                max_len_used = MAXLEN_INCREASE_TO
        else:
            raise RuntimeError("Not enough samples under current MAX_LENGTH; increase MAX_LENGTH.")
    else:
        max_len_used = MAX_LENGTH

    # Save selection metadata for reproducibility
    meta = {
        "model_id": MODEL_ID,
        "dataset_id": DATASET_ID,
        "target_sample_count": TARGET_SAMPLE_COUNT,
        "max_len_used": max_len_used,
        "selection_strategy": "bucketed_by_token_length",
        "seed": RNG_SEED,
        "selected_indices_in_original_dataset": selected,
        "teacher_logits": {
            "enabled": SAVE_TEACHER_LOGITS,
            "topk": TEACHER_LOGITS_TOPK,
            "dtype": str(TEACHER_LOGITS_DTYPE),
            "scope": "y_tokens_only_final_layer_lm_head",
        },
    }
    with open(os.path.join(OUTPUT_DIR, "selection_meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"Saved selection meta to {os.path.join(OUTPUT_DIR, 'selection_meta.json')}")

    # Spawn GPU workers
    mp.set_start_method("spawn", force=True)
    print(f"Starting extraction on {NUM_GPUS} GPUs (max_len={max_len_used})...")

    procs = []
    for rank in range(NUM_GPUS):
        p = mp.Process(target=run_extraction, args=(rank, NUM_GPUS, selected, max_len_used))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("All extraction completed!")


if __name__ == "__main__":
    main()
