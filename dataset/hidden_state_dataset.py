import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from safetensors import safe_open
from torch.utils.data import Dataset


_HIDDEN_KEY_RE = re.compile(r"^sample_(\d+)_hidden$")
_QLEN_KEY_TMPL = "sample_{sid}_qlen"
_HIDDEN_KEY_TMPL = "sample_{sid}_hidden"
_INPUT_IDS_KEY_TMPL = "sample_{sid}_input_ids"
_TEACHER_LOGITS_KEY_TMPL = "sample_{sid}_teacher_logits"
_TEACHER_LOGITS_TOPK_VALS_KEY_TMPL = "sample_{sid}_teacher_logits_topk_vals"
_TEACHER_LOGITS_TOPK_IDX_KEY_TMPL = "sample_{sid}_teacher_logits_topk_idx"


def _extract_sample_ids(keys: Sequence[str]) -> List[int]:
    ids: List[int] = []
    for key in keys:
        m = _HIDDEN_KEY_RE.match(key)
        if m is not None:
            ids.append(int(m.group(1)))
    ids.sort()
    return ids


class FlowHiddenStateDataset(Dataset):
    """
    Loads teacher hidden-state chunks saved by dataset/collect_data.py.

    Stored tensors per sample:
      - sample_{id}_hidden: [L_total, seq_len, D]
      - sample_{id}_qlen:   [1] (x length)
      - sample_{id}_input_ids: [seq_len]
      - optional sample_{id}_teacher_logits: [y_len, vocab]
      - optional sample_{id}_teacher_logits_topk_vals / _idx: [y_len, topk]

    Fixed-length training convention:
      - Every sample is mapped to fixed_total_length.
      - Positions beyond real seq_len are treated as PAD.
      - Y-side hidden states are padded with hidden_pad_value up to target_y_len.
    """

    def __init__(
        self,
        data_root: str,
        pad_token_id: int = 0,
        ignore_index: int = -100,
        fixed_total_length: int = 1024,
        include_padded_y_in_loss: bool = False,
        hidden_pad_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.pad_token_id = int(pad_token_id)
        self.ignore_index = int(ignore_index)
        self.fixed_total_length = int(fixed_total_length)
        self.include_padded_y_in_loss = bool(include_padded_y_in_loss)
        self.hidden_pad_value = float(hidden_pad_value)

        if self.fixed_total_length <= 1:
            raise ValueError(f"fixed_total_length must be > 1, got {self.fixed_total_length}.")

        if not self.data_root.exists():
            raise FileNotFoundError(f"data_root not found: {self.data_root}")

        self.chunk_paths = sorted(self.data_root.glob("shard_*/*.safetensors"))
        if len(self.chunk_paths) == 0:
            raise FileNotFoundError(
                f"No safetensors chunks found under: {self.data_root}/shard_*/*.safetensors"
            )

        self.sample_index: List[Tuple[int, int]] = []
        self.hidden_size: int = -1
        self.total_layers: int = -1
        self.teacher_logits_dense_count: int = 0
        self.teacher_logits_topk_count: int = 0

        for chunk_idx, chunk_path in enumerate(self.chunk_paths):
            with safe_open(str(chunk_path), framework="pt") as f:
                keys = list(f.keys())
                key_set = set(keys)
                sample_ids = _extract_sample_ids(keys)

                for sid in sample_ids:
                    q_key = _QLEN_KEY_TMPL.format(sid=sid)
                    input_ids_key = _INPUT_IDS_KEY_TMPL.format(sid=sid)
                    if q_key not in key_set:
                        raise KeyError(f"Missing {q_key} in {chunk_path}")
                    if input_ids_key not in key_set:
                        raise KeyError(
                            f"Missing {input_ids_key} in {chunk_path}. "
                            "Regenerate data with dataset/collect_data.py that stores input_ids."
                        )

                    dense_key = _TEACHER_LOGITS_KEY_TMPL.format(sid=sid)
                    topk_vals_key = _TEACHER_LOGITS_TOPK_VALS_KEY_TMPL.format(sid=sid)
                    topk_idx_key = _TEACHER_LOGITS_TOPK_IDX_KEY_TMPL.format(sid=sid)
                    has_dense = dense_key in key_set
                    has_topk_vals = topk_vals_key in key_set
                    has_topk_idx = topk_idx_key in key_set

                    if has_topk_vals != has_topk_idx:
                        raise KeyError(
                            f"teacher_logits_topk keys are incomplete for sample_{sid} in {chunk_path}. "
                            f"Need both {topk_vals_key} and {topk_idx_key}."
                        )
                    if has_dense and has_topk_vals:
                        raise ValueError(
                            f"Both dense and topk teacher logits exist for sample_{sid} in {chunk_path}. "
                            "Store only one format per sample."
                        )
                    if has_dense:
                        self.teacher_logits_dense_count += 1
                    elif has_topk_vals:
                        self.teacher_logits_topk_count += 1

                if sample_ids:
                    h = f.get_tensor(_HIDDEN_KEY_TMPL.format(sid=sample_ids[0]))
                    if h.ndim != 3:
                        raise ValueError(
                            f"Expected hidden tensor rank 3 [L,S,D], got {tuple(h.shape)} in {chunk_path}"
                        )
                    layers, _, hidden_size = h.shape
                    if self.hidden_size < 0:
                        self.hidden_size = int(hidden_size)
                        self.total_layers = int(layers)
                    elif hidden_size != self.hidden_size or layers != self.total_layers:
                        raise ValueError(
                            "Inconsistent hidden schema across chunks: "
                            f"got layers={layers}, hidden={hidden_size} in {chunk_path}, "
                            f"expected layers={self.total_layers}, hidden={self.hidden_size}."
                        )

                self.sample_index.extend((chunk_idx, sid) for sid in sample_ids)

        if self.hidden_size < 0 or self.total_layers < 2:
            raise ValueError("Dataset appears empty or missing anchor layers.")

        self.num_anchor_layers = self.total_layers - 1
        self.num_samples = len(self.sample_index)
        if self.teacher_logits_dense_count == self.num_samples and self.teacher_logits_topk_count == 0:
            self.teacher_logits_mode = "dense"
        elif self.teacher_logits_topk_count == self.num_samples and self.teacher_logits_dense_count == 0:
            self.teacher_logits_mode = "topk"
        elif self.teacher_logits_dense_count == 0 and self.teacher_logits_topk_count == 0:
            self.teacher_logits_mode = "none"
        else:
            self.teacher_logits_mode = "mixed"

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk_idx, sid = self.sample_index[idx]
        chunk_path = self.chunk_paths[chunk_idx]

        hidden_key = _HIDDEN_KEY_TMPL.format(sid=sid)
        qlen_key = _QLEN_KEY_TMPL.format(sid=sid)
        input_ids_key = _INPUT_IDS_KEY_TMPL.format(sid=sid)
        dense_logits_key = _TEACHER_LOGITS_KEY_TMPL.format(sid=sid)
        topk_vals_key = _TEACHER_LOGITS_TOPK_VALS_KEY_TMPL.format(sid=sid)
        topk_idx_key = _TEACHER_LOGITS_TOPK_IDX_KEY_TMPL.format(sid=sid)

        with safe_open(str(chunk_path), framework="pt") as f:
            key_set = set(f.keys())
            hidden = f.get_tensor(hidden_key)  # [L_total, seq_len, D]
            qlen = int(f.get_tensor(qlen_key).item())
            input_ids = f.get_tensor(input_ids_key).to(torch.long)  # [seq_len]
            has_dense = dense_logits_key in key_set
            has_topk_vals = topk_vals_key in key_set
            has_topk_idx = topk_idx_key in key_set

            if has_topk_vals != has_topk_idx:
                raise KeyError(
                    f"teacher_logits_topk keys are incomplete for sample_{sid} in {chunk_path}. "
                    f"Need both {topk_vals_key} and {topk_idx_key}."
                )
            if has_dense and has_topk_vals:
                raise ValueError(
                    f"Both dense and topk teacher logits exist for sample_{sid} in {chunk_path}. "
                    "Store only one format per sample."
                )

            teacher_logits = f.get_tensor(dense_logits_key) if has_dense else None
            teacher_logits_topk_vals = f.get_tensor(topk_vals_key) if has_topk_vals else None
            teacher_logits_topk_idx = f.get_tensor(topk_idx_key) if has_topk_idx else None

        if input_ids.ndim != 1:
            raise ValueError(
                f"Expected input_ids rank 1 [seq_len], got {tuple(input_ids.shape)} "
                f"(sample_{sid} in {chunk_path})."
            )

        total_layers, seq_len, hidden_size = hidden.shape
        if input_ids.size(0) != seq_len:
            raise ValueError(
                f"Mismatched seq lengths for sample_{sid} in {chunk_path}: "
                f"hidden seq_len={seq_len}, input_ids len={input_ids.size(0)}."
            )
        if total_layers != self.total_layers or hidden_size != self.hidden_size:
            raise ValueError(
                f"Inconsistent sample tensor shape in {chunk_path}: got {tuple(hidden.shape)}, "
                f"expected [{self.total_layers}, seq_len, {self.hidden_size}]"
            )

        # Hard truncate to fixed_total_length if needed.
        if seq_len > self.fixed_total_length:
            hidden = hidden[:, : self.fixed_total_length, :]
            input_ids = input_ids[: self.fixed_total_length]
            seq_len = self.fixed_total_length

        if qlen <= 0 or qlen >= seq_len:
            raise ValueError(
                f"Invalid qlen={qlen} for seq_len={seq_len} (sample_{sid} in {chunk_path})."
            )

        # Full fixed-length token/mask tensors.
        input_ids_fixed = torch.full(
            (self.fixed_total_length,),
            self.pad_token_id,
            dtype=torch.long,
        )
        input_ids_fixed[:seq_len] = input_ids

        attention_mask = torch.zeros((self.fixed_total_length,), dtype=torch.long)
        attention_mask[:seq_len] = 1

        # Interval-wise X states (for x_t construction at sampled interval).
        x_layers = hidden[:, :qlen, :].contiguous()  # [L_total, x_len, D]

        # Y states padded to fixed target length for this sample.
        y_hidden_real = hidden[:, qlen:, :]  # [L_total, y_len_real, D]
        y_len_real = y_hidden_real.size(1)
        target_y_len = self.fixed_total_length - qlen
        pad_y = target_y_len - y_len_real
        if pad_y < 0:
            raise ValueError(
                f"Negative y padding for sample_{sid}: qlen={qlen}, seq_len={seq_len}, "
                f"fixed_total_length={self.fixed_total_length}."
            )
        if pad_y > 0:
            y_pad = torch.full(
                (total_layers, pad_y, hidden_size),
                self.hidden_pad_value,
                dtype=y_hidden_real.dtype,
            )
            y_hidden = torch.cat([y_hidden_real, y_pad], dim=1)
        else:
            y_hidden = y_hidden_real

        h0 = y_hidden[0].contiguous()  # [target_y_len, D]
        teacher_anchors = y_hidden[1:].contiguous()  # [M, target_y_len, D]

        x_ids = input_ids[:qlen].contiguous()
        x_mask = torch.ones((qlen,), dtype=torch.long)
        y_ids = torch.full((target_y_len,), self.ignore_index, dtype=torch.long)
        y_ids[:y_len_real] = input_ids[qlen:seq_len]
        y_mask = torch.zeros((target_y_len,), dtype=torch.long)
        if self.include_padded_y_in_loss:
            y_mask[:] = 1
        else:
            y_mask[:y_len_real] = 1

        out = {
            "x_ids": x_ids,
            "x_mask": x_mask,
            "y_ids": y_ids,
            "y_mask": y_mask,
            "teacher_anchors": teacher_anchors,
            "teacher_x_layers": x_layers,
            "h0": h0,
            "input_ids_fixed": input_ids_fixed,
            "attention_mask": attention_mask,
            "q_len": torch.tensor(qlen, dtype=torch.long),
            "seq_len_real": torch.tensor(seq_len, dtype=torch.long),
            "y_len_real": torch.tensor(y_len_real, dtype=torch.long),
            "sample_id": torch.tensor(sid, dtype=torch.long),
        }

        if teacher_logits is not None:
            if teacher_logits.ndim != 2:
                raise ValueError(
                    f"teacher_logits must have rank 2 [y_len, vocab], got {tuple(teacher_logits.shape)} "
                    f"(sample_{sid} in {chunk_path})."
                )
            if teacher_logits.size(0) < y_len_real:
                raise ValueError(
                    f"teacher_logits y_len {teacher_logits.size(0)} is smaller than y_len_real={y_len_real} "
                    f"(sample_{sid} in {chunk_path})."
                )
            vocab_size = teacher_logits.size(1)
            teacher_logits = teacher_logits[:y_len_real]
            teacher_logits_padded = torch.zeros((target_y_len, vocab_size), dtype=teacher_logits.dtype)
            teacher_logits_padded[:y_len_real] = teacher_logits
            out["teacher_logits"] = teacher_logits_padded

        if teacher_logits_topk_vals is not None and teacher_logits_topk_idx is not None:
            if teacher_logits_topk_vals.ndim != 2 or teacher_logits_topk_idx.ndim != 2:
                raise ValueError(
                    f"teacher_logits_topk tensors must be rank 2, got "
                    f"{tuple(teacher_logits_topk_vals.shape)} and {tuple(teacher_logits_topk_idx.shape)} "
                    f"(sample_{sid} in {chunk_path})."
                )
            if teacher_logits_topk_vals.shape != teacher_logits_topk_idx.shape:
                raise ValueError(
                    f"teacher_logits_topk shapes mismatch: "
                    f"{tuple(teacher_logits_topk_vals.shape)} vs {tuple(teacher_logits_topk_idx.shape)} "
                    f"(sample_{sid} in {chunk_path})."
                )
            if teacher_logits_topk_vals.size(0) < y_len_real:
                raise ValueError(
                    f"teacher_logits_topk y_len {teacher_logits_topk_vals.size(0)} is smaller than "
                    f"y_len_real={y_len_real} (sample_{sid} in {chunk_path})."
                )
            topk = teacher_logits_topk_vals.size(1)
            teacher_logits_topk_vals = teacher_logits_topk_vals[:y_len_real]
            teacher_logits_topk_idx = teacher_logits_topk_idx[:y_len_real]
            topk_vals_padded = torch.zeros((target_y_len, topk), dtype=teacher_logits_topk_vals.dtype)
            topk_idx_padded = torch.zeros((target_y_len, topk), dtype=teacher_logits_topk_idx.dtype)
            topk_vals_padded[:y_len_real] = teacher_logits_topk_vals
            topk_idx_padded[:y_len_real] = teacher_logits_topk_idx
            out["teacher_logits_topk_vals"] = topk_vals_padded
            out["teacher_logits_topk_idx"] = topk_idx_padded

        return out


class FlowBatchCollator:
    def __init__(self, pad_token_id: int = 0, ignore_index: int = -100):
        self.pad_token_id = int(pad_token_id)
        self.ignore_index = int(ignore_index)

    def __call__(self, batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if len(batch) == 0:
            raise ValueError("Empty batch.")

        batch_size = len(batch)
        max_x = max(item["x_ids"].size(0) for item in batch)
        max_y = max(item["y_ids"].size(0) for item in batch)
        fixed_len = batch[0]["input_ids_fixed"].size(0)

        m = batch[0]["teacher_anchors"].size(0)
        l_total = batch[0]["teacher_x_layers"].size(0)
        d = batch[0]["teacher_anchors"].size(2)
        anchor_dtype = batch[0]["teacher_anchors"].dtype
        x_layer_dtype = batch[0]["teacher_x_layers"].dtype
        h0_dtype = batch[0]["h0"].dtype

        x_ids = torch.full((batch_size, max_x), self.pad_token_id, dtype=torch.long)
        x_mask = torch.zeros((batch_size, max_x), dtype=torch.long)
        y_ids = torch.full((batch_size, max_y), self.ignore_index, dtype=torch.long)
        y_mask = torch.zeros((batch_size, max_y), dtype=torch.long)

        teacher_anchors = torch.zeros((batch_size, m, max_y, d), dtype=anchor_dtype)
        teacher_x_layers = torch.zeros((batch_size, l_total, max_x, d), dtype=x_layer_dtype)
        h0 = torch.zeros((batch_size, max_y, d), dtype=h0_dtype)

        input_ids_fixed = torch.full((batch_size, fixed_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, fixed_len), dtype=torch.long)
        q_lens = torch.zeros((batch_size,), dtype=torch.long)
        seq_len_real = torch.zeros((batch_size,), dtype=torch.long)
        y_len_real = torch.zeros((batch_size,), dtype=torch.long)
        sample_ids = torch.zeros((batch_size,), dtype=torch.long)

        dense_logits_flags = [("teacher_logits" in item) for item in batch]
        topk_logits_flags = [
            ("teacher_logits_topk_vals" in item and "teacher_logits_topk_idx" in item) for item in batch
        ]
        use_dense_logits = all(dense_logits_flags)
        use_topk_logits = (not use_dense_logits) and all(topk_logits_flags)

        teacher_logits = None
        teacher_logits_topk_vals = None
        teacher_logits_topk_idx = None
        if use_dense_logits:
            vocab_size = batch[0]["teacher_logits"].size(1)
            logits_dtype = batch[0]["teacher_logits"].dtype
            teacher_logits = torch.zeros((batch_size, max_y, vocab_size), dtype=logits_dtype)
        elif use_topk_logits:
            topk = batch[0]["teacher_logits_topk_vals"].size(1)
            topk_vals_dtype = batch[0]["teacher_logits_topk_vals"].dtype
            topk_idx_dtype = batch[0]["teacher_logits_topk_idx"].dtype
            teacher_logits_topk_vals = torch.zeros((batch_size, max_y, topk), dtype=topk_vals_dtype)
            teacher_logits_topk_idx = torch.zeros((batch_size, max_y, topk), dtype=topk_idx_dtype)

        for i, item in enumerate(batch):
            x_len = item["x_ids"].size(0)
            y_len = item["y_ids"].size(0)

            if item["teacher_anchors"].size(0) != m:
                raise ValueError("Inconsistent number of anchor layers in batch.")
            if item["teacher_x_layers"].size(0) != l_total:
                raise ValueError("Inconsistent number of total layers in batch.")
            if item["teacher_anchors"].size(2) != d or item["teacher_x_layers"].size(2) != d:
                raise ValueError("Inconsistent hidden size in batch.")
            if item["input_ids_fixed"].size(0) != fixed_len:
                raise ValueError("Inconsistent fixed sequence length in batch.")

            x_ids[i, :x_len] = item["x_ids"]
            x_mask[i, :x_len] = item["x_mask"]
            y_ids[i, :y_len] = item["y_ids"]
            y_mask[i, :y_len] = item["y_mask"]
            teacher_anchors[i, :, :y_len, :] = item["teacher_anchors"]
            teacher_x_layers[i, :, :x_len, :] = item["teacher_x_layers"]
            h0[i, :y_len, :] = item["h0"]

            input_ids_fixed[i] = item["input_ids_fixed"]
            attention_mask[i] = item["attention_mask"]
            q_lens[i] = item["q_len"]
            seq_len_real[i] = item["seq_len_real"]
            y_len_real[i] = item["y_len_real"]
            sample_ids[i] = item["sample_id"]

            if use_dense_logits:
                sample_logits = item["teacher_logits"]
                if sample_logits.ndim != 2:
                    raise ValueError(f"teacher_logits must be rank 2, got {tuple(sample_logits.shape)}.")
                if sample_logits.size(1) != teacher_logits.size(2):
                    raise ValueError("Inconsistent teacher_logits vocab size in batch.")
                if sample_logits.size(0) != y_len:
                    raise ValueError(
                        f"teacher_logits y_len mismatch: expected {y_len}, got {sample_logits.size(0)}."
                    )
                teacher_logits[i, :y_len, :] = sample_logits

            if use_topk_logits:
                sample_topk_vals = item["teacher_logits_topk_vals"]
                sample_topk_idx = item["teacher_logits_topk_idx"]
                if sample_topk_vals.ndim != 2 or sample_topk_idx.ndim != 2:
                    raise ValueError(
                        "teacher_logits_topk tensors must be rank 2, got "
                        f"{tuple(sample_topk_vals.shape)} and {tuple(sample_topk_idx.shape)}."
                    )
                if sample_topk_vals.shape != sample_topk_idx.shape:
                    raise ValueError("teacher_logits_topk vals/idx shape mismatch in batch.")
                if sample_topk_vals.size(1) != teacher_logits_topk_vals.size(2):
                    raise ValueError("Inconsistent teacher_logits_topk size in batch.")
                if sample_topk_vals.size(0) != y_len:
                    raise ValueError(
                        f"teacher_logits_topk y_len mismatch: expected {y_len}, got {sample_topk_vals.size(0)}."
                    )
                teacher_logits_topk_vals[i, :y_len, :] = sample_topk_vals
                teacher_logits_topk_idx[i, :y_len, :] = sample_topk_idx

        times = torch.linspace(0.0, 1.0, steps=m + 1, dtype=torch.float32)

        out = {
            "x_ids": x_ids,
            "x_mask": x_mask,
            "y_ids": y_ids,
            "y_mask": y_mask,
            "teacher_anchors": teacher_anchors,
            "teacher_x_layers": teacher_x_layers,
            "h0": h0,
            "input_ids_fixed": input_ids_fixed,
            "attention_mask": attention_mask,
            "q_lens": q_lens,
            "seq_len_real": seq_len_real,
            "y_len_real": y_len_real,
            "times": times,
            "sample_ids": sample_ids,
        }

        if use_dense_logits:
            out["teacher_logits"] = teacher_logits
        elif use_topk_logits:
            out["teacher_logits_topk_vals"] = teacher_logits_topk_vals
            out["teacher_logits_topk_idx"] = teacher_logits_topk_idx

        return out
