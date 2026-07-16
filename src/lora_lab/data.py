"""Data layer: raw samples -> chat-formatted, label-masked multimodal batches.

Contract (the only place that knows the data format):

    build_dataset(data_cfg, processor) -> (train_ds, eval_ds, collator)

Each raw sample is a dict ``{"image": PIL.Image, "target_json": str}``. All
heavy lifting (chat template application, tokenization, image processing, label
masking) happens inside ``MultimodalJSONCollator`` at batch time, so the code
path is identical for the tiny smoke model and Qwen3.5-9B.

Label masking rule: everything except the assistant's JSON answer is set to
``-100`` so the loss is computed ONLY on the tokens we want the model to learn
to generate. Image/prompt/thinking tokens never contribute to the loss.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image

from .config import DataConfig
from .registry import get_dataset_builder, register_dataset

IGNORE_INDEX = -100

# Keys produced by VL processors that must be concatenated (not padded) across a batch.
_VISION_KEYS = ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw")


# --------------------------------------------------------------------------- #
# Dataset builders (registry). Add a new domain by writing one function.
# --------------------------------------------------------------------------- #
def _downscale(image: Image.Image, max_pixels: int | None) -> Image.Image:
    if max_pixels is None:
        return image
    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = (max_pixels / (w * h)) ** 0.5
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))))


def _cord_ground_truth_to_target(gt_raw: str) -> dict[str, Any]:
    """Normalize CORD-v2 ``ground_truth`` into our simplified schema.

    CORD stores a JSON string like {"gt_parse": {"menu": [...], "sub_total": {...},
    "total": {...}}}. We flatten it to {menu, subtotal, tax, total}.
    """
    try:
        parsed = json.loads(gt_raw).get("gt_parse", {})
    except (json.JSONDecodeError, AttributeError):
        parsed = {}

    def _num(x: Any) -> Any:
        if isinstance(x, str):
            return x.replace(",", "").strip()
        return x

    def _as_dict(x: Any) -> dict:
        # CORD sometimes stores sub_total/total as a list of dicts instead of a
        # single dict; merge list entries into one dict, ignore other types.
        if isinstance(x, dict):
            return x
        if isinstance(x, list):
            merged: dict = {}
            for item in x:
                if isinstance(item, dict):
                    merged.update(item)
            return merged
        return {}

    menu_raw = parsed.get("menu", [])
    if isinstance(menu_raw, dict):
        menu_raw = [menu_raw]
    menu = []
    for item in menu_raw:
        if not isinstance(item, dict):
            continue
        menu.append(
            {
                "name": item.get("nm", ""),
                "count": _num(item.get("cnt", "")),
                "price": _num(item.get("price", "")),
            }
        )

    sub = _as_dict(parsed.get("sub_total", {}))
    tot = _as_dict(parsed.get("total", {}))
    return {
        "menu": menu,
        "subtotal": _num(sub.get("subtotal_price", "")),
        "tax": _num(sub.get("tax_price", "")),
        "total": _num(tot.get("total_price", "")),
    }


@register_dataset("cord")
def _build_cord(data_cfg: DataConfig):
    from datasets import load_dataset

    def _prep(split: str, limit: int | None):
        ds = load_dataset(data_cfg.dataset_id, split=split)
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))

        def _map(example):
            target = _cord_ground_truth_to_target(example["ground_truth"])
            return {"target_json": json.dumps(target, ensure_ascii=False, separators=(",", ":"))}

        ds = ds.map(_map, remove_columns=[c for c in ds.column_names if c != "image"])
        return ds

    train = _prep(data_cfg.train_split, data_cfg.max_train_samples)
    eval_ = _prep(data_cfg.eval_split, data_cfg.max_eval_samples)
    return train, eval_


@register_dataset("synthetic_invoices")
def _build_synthetic_invoices(data_cfg: DataConfig):
    """Locally generated receipts (see ``synthetic.py``) - no downloads needed.

    Train and eval use disjoint seed ranges so there is no leakage. Sizes come
    from ``max_train_samples`` / ``max_eval_samples`` (with small defaults).
    """
    from .synthetic import generate_invoice

    n_train = data_cfg.max_train_samples or 256
    n_eval = data_cfg.max_eval_samples or 64

    class _SyntheticDataset:
        """Sequence of lazily generated samples (datasets.Dataset not needed:
        the trainer/eval only require __len__ and __getitem__)."""

        def __init__(self, start: int, n: int):
            self.start = start
            self.n = n

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, idx: int) -> dict:
            if isinstance(idx, slice):
                return [self[i] for i in range(*idx.indices(self.n))]
            if not -self.n <= idx < self.n:
                raise IndexError(idx)
            return generate_invoice(self.start + (idx % self.n))

    return _SyntheticDataset(0, n_train), _SyntheticDataset(1_000_000, n_eval)


# --------------------------------------------------------------------------- #
# Collator: turns a list of {image, target_json} into a masked, batched tensor.
# --------------------------------------------------------------------------- #
@dataclass
class MultimodalJSONCollator:
    processor: Any
    instruction: str
    max_seq_len: int = 2048
    image_max_pixels: int | None = 1_048_576
    train: bool = True
    # The prompt (image placeholder + fixed instruction) tokenizes to the same
    # length for every image of the same post-downscale size, so its length is
    # cached per (w, h). This halves processor work: without the cache every
    # sample costs TWO full processor passes (full text + prompt) per epoch.
    cache_prompt_len: bool = True
    _prompt_len_cache: dict[tuple[int, int], int] = field(
        default_factory=dict, init=False, repr=False
    )

    def _messages(self, target_json: str | None) -> list[dict[str, Any]]:
        user = {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": self.instruction},
            ],
        }
        msgs = [user]
        if target_json is not None:
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": target_json}]})
        return msgs

    def _prompt_len(self, image: Image.Image) -> int:
        """Length of the (image-expanded) prompt in tokens.

        Only the *length* of the prompt encoding is ever needed (to place the
        label mask), and it depends solely on the post-downscale image size:
        the instruction text is fixed and the number of image tokens is a
        function of the image grid, not the pixel content. Cache it per size.
        """
        size_key = (int(image.size[0]), int(image.size[1]))
        if self.cache_prompt_len:
            cached = self._prompt_len_cache.get(size_key)
            if cached is not None:
                return cached

        # Prompt-only text (up to where the assistant should start generating),
        # encoded with the SAME image so the placeholder expands identically.
        prompt_text = self.processor.apply_chat_template(
            self._messages(None), tokenize=False, add_generation_prompt=True
        )
        prompt = self.processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
            max_length=self.max_seq_len,
            truncation=True,
        )
        prompt_len = int(prompt["input_ids"].shape[1])
        if self.cache_prompt_len:
            self._prompt_len_cache[size_key] = prompt_len
        return prompt_len

    def _encode_one(self, image: Image.Image, target_json: str):
        image = _downscale(image.convert("RGB"), self.image_max_pixels)

        # Full conversation text (prompt + assistant answer).
        full_text = self.processor.apply_chat_template(
            self._messages(target_json), tokenize=False, add_generation_prompt=False
        )
        full = self.processor(
            text=[full_text],
            images=[image],
            return_tensors="pt",
            max_length=self.max_seq_len,
            truncation=True,
        )

        input_ids = full["input_ids"][0]
        prompt_len = min(self._prompt_len(image), input_ids.shape[0])

        labels = input_ids.clone()
        labels[:prompt_len] = IGNORE_INDEX  # mask image + prompt tokens

        encoded = {k: v[0] for k, v in full.items() if k not in _VISION_KEYS}
        encoded["labels"] = labels
        vision = {k: full[k] for k in _VISION_KEYS if k in full}
        return encoded, vision

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.processor.tokenizer.eos_token_id

        encoded_list, vision_list = [], []
        for f in features:
            enc, vis = self._encode_one(f["image"], f["target_json"])
            if "attention_mask" not in enc:
                enc["attention_mask"] = torch.ones_like(enc["input_ids"])
            encoded_list.append(enc)
            vision_list.append(vis)

        max_len = max(e["input_ids"].shape[0] for e in encoded_list)

        # Pad value per key; anything else per-token (e.g. mm_token_type_ids,
        # token_type_ids) pads with 0. Modern VL processors return extra per-token
        # tensors the model requires (M-RoPE needs mm_token_type_ids), so we pad
        # EVERY sequence-aligned tensor rather than a hard-coded few.
        pad_values = {"input_ids": pad_id, "labels": IGNORE_INDEX, "attention_mask": 0}
        ref_len = encoded_list[0]["input_ids"].shape[0]
        seq_keys = [
            k
            for k, v in encoded_list[0].items()
            if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == ref_len
        ]

        out: dict[str, torch.Tensor] = {}
        for key in seq_keys:
            fill = pad_values.get(key, 0)
            rows = []
            for e in encoded_list:
                t = e[key]
                pad = max_len - t.shape[0]
                if pad > 0:
                    pad_shape = (pad,) + tuple(t.shape[1:])
                    t = torch.cat([t, torch.full(pad_shape, fill, dtype=t.dtype)], dim=0)
                rows.append(t)
            out[key] = torch.stack(rows)

        # Vision tensors are concatenated along dim 0 (patch/image axis), not padded.
        for key in _VISION_KEYS:
            tensors = [v[key] for v in vision_list if key in v]
            if tensors:
                out[key] = torch.cat(tensors, dim=0)
        return out


def build_dataset(data_cfg: DataConfig, processor: Any):
    """Return ``(train_ds, eval_ds, train_collator, eval_collator)``."""
    builder = get_dataset_builder(data_cfg.type)
    train_ds, eval_ds = builder(data_cfg)
    train_collator = MultimodalJSONCollator(
        processor=processor,
        instruction=data_cfg.instruction,
        max_seq_len=data_cfg.max_seq_len,
        image_max_pixels=data_cfg.image_max_pixels,
        train=True,
    )
    eval_collator = MultimodalJSONCollator(
        processor=processor,
        instruction=data_cfg.instruction,
        max_seq_len=data_cfg.max_seq_len,
        image_max_pixels=data_cfg.image_max_pixels,
        train=False,
    )
    return train_ds, eval_ds, train_collator, eval_collator
