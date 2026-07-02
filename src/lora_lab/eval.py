"""Objective evaluation for the image->JSON extractor.

Three metrics, all computed on the SAME held-out set for both the baseline
(no adapter) and the adapted model, so the effect of LoRA is a number:

  - json_valid_rate : fraction of outputs that parse as JSON
  - field_f1        : F1 over flattened (key -> value) pairs
  - exact_match     : fraction where the parsed JSON equals the target exactly

The generation prompt reuses the same chat template / instruction as training
but WITHOUT the assistant turn (``add_generation_prompt=True``).
"""

from __future__ import annotations

import json
import re
from typing import Any

import torch

from .config import DataConfig
from .data import _downscale

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict | None:
    """Best-effort: pull the first {...} block and parse it."""
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _flatten(obj: Any, prefix: str = "") -> set[tuple[str, str]]:
    """Flatten a nested dict/list into a set of (path, value) pairs for F1."""
    pairs: set[tuple[str, str]] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            pairs |= _flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        # Order-insensitive for list items (e.g. menu lines).
        for item in obj:
            pairs |= _flatten(item, f"{prefix}[]")
    else:
        pairs.add((prefix, str(obj).strip()))
    return pairs


def field_prf(pred: dict | None, gold: dict) -> tuple[float, float, float]:
    gold_pairs = _flatten(gold)
    if pred is None:
        return 0.0, 0.0, 0.0
    pred_pairs = _flatten(pred)
    if not pred_pairs and not gold_pairs:
        return 1.0, 1.0, 1.0
    tp = len(pred_pairs & gold_pairs)
    precision = tp / len(pred_pairs) if pred_pairs else 0.0
    recall = tp / len(gold_pairs) if gold_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


@torch.no_grad()
def _generate_json(model, processor, image, instruction: str, data_cfg: DataConfig) -> str:
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }
    ]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image = _downscale(image.convert("RGB"), data_cfg.image_max_pixels)
    inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=768, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1] :]
    return processor.tokenizer.decode(gen, skip_special_tokens=True)


def evaluate(model, processor, eval_ds, data_cfg: DataConfig, limit: int | None = None) -> dict:
    """Run generation over ``eval_ds`` and return aggregate metrics."""
    model.eval()
    n = len(eval_ds) if limit is None else min(limit, len(eval_ds))
    valid = 0
    exact = 0
    f1_sum = 0.0
    for i in range(n):
        row = eval_ds[i]
        gold = json.loads(row["target_json"])
        raw = _generate_json(model, processor, row["image"], data_cfg.instruction, data_cfg)
        pred = extract_json(raw)
        if pred is not None:
            valid += 1
            if pred == gold:
                exact += 1
        _, _, f1 = field_prf(pred, gold)
        f1_sum += f1

    return {
        "n": n,
        "json_valid_rate": round(valid / n, 4) if n else 0.0,
        "field_f1": round(f1_sum / n, 4) if n else 0.0,
        "exact_match": round(exact / n, 4) if n else 0.0,
    }
