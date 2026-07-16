"""Objective evaluation for the image->JSON extractor.

Metrics (all computed on the SAME held-out set for baseline and adapter, so the
effect of LoRA is a number):

  - json_valid_rate   : fraction of outputs that parse as JSON
  - schema_valid_rate : fraction that also conform to the JSON Schema
  - field_f1          : sample-averaged F1 over flattened (path -> value) pairs
  - exact_match       : fraction where parsed JSON equals the target exactly
  - per_field         : micro precision/recall/F1 per top-level schema key

Values are normalized before comparison (currency/commas stripped, numbers
canonicalized) so "1,200.00" and "1200" match. Generation is batched with left
padding. Optionally uses grammar-constrained decoding (see ``constrained``).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .config import DataConfig
from .data import _downscale
from .generate_utils import safe_generate
from .registry import register_metric
from .schema import load_schema, validate_json

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_CURRENCY_RE = re.compile(r"[^\d.\-]")


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


def normalize_value(value: Any) -> str:
    """Canonicalize a leaf value so equivalent numbers/strings compare equal.

    "1,200.00" -> "1200", "12.50" -> "12.5", "  Coke " -> "coke".
    """
    s = str(value).strip()
    if not s:
        return ""
    cleaned = _CURRENCY_RE.sub("", s)
    if cleaned not in ("", "-", ".", "-."):
        try:
            num = float(cleaned)
            # Canonical numeric form: drop trailing zeros / trailing dot.
            return f"{num:.4f}".rstrip("0").rstrip(".")
        except ValueError:
            pass
    return s.lower()


def _top_key(path: str) -> str:
    """Top-level schema key for a flattened path: 'menu[].name' -> 'menu'."""
    for sep in (".", "["):
        idx = path.find(sep)
        if idx != -1:
            path = path[:idx]
    return path


def _flatten(obj: Any, prefix: str = "") -> set[tuple[str, str]]:
    """Flatten nested dict/list into a set of (path, normalized_value) pairs."""
    pairs: set[tuple[str, str]] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            pairs |= _flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        # Order-insensitive for list items (e.g. menu lines).
        for item in obj:
            pairs |= _flatten(item, f"{prefix}[]")
    else:
        pairs.add((prefix, normalize_value(obj)))
    return pairs


@register_metric("field_prf")
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


def _accumulate_per_field(pred: dict | None, gold: dict, acc: dict[str, list[int]]) -> None:
    """Accumulate micro tp/fp/fn per top-level key into ``acc`` (in place)."""
    gold_pairs = _flatten(gold)
    pred_pairs = _flatten(pred) if pred else set()
    keys = {_top_key(p) for p, _ in gold_pairs} | {_top_key(p) for p, _ in pred_pairs}
    for key in keys:
        g = {pair for pair in gold_pairs if _top_key(pair[0]) == key}
        p = {pair for pair in pred_pairs if _top_key(pair[0]) == key}
        tp = len(g & p)
        acc[key][0] += tp
        acc[key][1] += len(p) - tp  # fp
        acc[key][2] += len(g) - tp  # fn


def _prf_from_counts(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


@torch.no_grad()
def _generate_batch(
    model,
    processor,
    images: list,
    instruction: str,
    data_cfg: DataConfig,
    max_new_tokens: int = 768,
    logits_processor=None,
) -> list[str]:
    """Batched, left-padded generation. Returns decoded completions."""
    tok = processor.tokenizer
    old_side = tok.padding_side
    tok.padding_side = "left"  # required for correct batched decoder generation
    try:
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}
        ]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts = [prompt] * len(images)
        imgs = [_downscale(im.convert("RGB"), data_cfg.image_max_pixels) for im in images]
        inputs = processor(text=prompts, images=imgs, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": False}
        if logits_processor is not None:
            gen_kwargs["logits_processor"] = logits_processor
        out = safe_generate(model, inputs, gen_kwargs)
        gen = out[:, inputs["input_ids"].shape[1] :]
        return tok.batch_decode(gen, skip_special_tokens=True)
    finally:
        tok.padding_side = old_side


def evaluate(
    model,
    processor,
    eval_ds,
    data_cfg: DataConfig,
    limit: int | None = None,
    batch_size: int = 4,
    predictions_path: str | Path | None = None,
    constrained: bool = False,
) -> dict:
    """Run batched generation over ``eval_ds`` and return aggregate metrics."""
    model.eval()
    schema = load_schema(data_cfg.json_schema)

    logits_processor = None
    if constrained:
        from .constrained import build_json_logits_processor

        logits_processor = build_json_logits_processor(model, processor, schema)

    n = len(eval_ds) if limit is None else min(limit, len(eval_ds))
    valid = 0
    schema_valid = 0
    exact = 0
    f1_sum = 0.0
    per_field: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

    fh = open(predictions_path, "w", encoding="utf-8") if predictions_path else None
    try:
        for start in range(0, n, batch_size):
            rows = [eval_ds[i] for i in range(start, min(start + batch_size, n))]
            golds = [json.loads(r["target_json"]) for r in rows]
            raws = _generate_batch(
                model,
                processor,
                [r["image"] for r in rows],
                data_cfg.instruction,
                data_cfg,
                logits_processor=logits_processor,
            )
            for offset, (raw, gold) in enumerate(zip(raws, golds, strict=True)):
                pred = extract_json(raw)
                is_valid = pred is not None
                is_schema_valid = is_valid and validate_json(pred, schema)
                if is_valid:
                    valid += 1
                    if pred == gold:
                        exact += 1
                if is_schema_valid:
                    schema_valid += 1
                _, _, f1 = field_prf(pred, gold)
                f1_sum += f1
                _accumulate_per_field(pred, gold, per_field)
                if fh:
                    fh.write(
                        json.dumps(
                            {
                                "index": start + offset,
                                "gold": gold,
                                "pred": pred,
                                "raw": raw,
                                "json_valid": is_valid,
                                "schema_valid": is_schema_valid,
                                "field_f1": round(f1, 4),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    finally:
        if fh:
            fh.close()

    return {
        "n": n,
        "json_valid_rate": round(valid / n, 4) if n else 0.0,
        "schema_valid_rate": round(schema_valid / n, 4) if n else 0.0,
        "field_f1": round(f1_sum / n, 4) if n else 0.0,
        "exact_match": round(exact / n, 4) if n else 0.0,
        "per_field": {k: _prf_from_counts(*v) for k, v in sorted(per_field.items())},
    }
