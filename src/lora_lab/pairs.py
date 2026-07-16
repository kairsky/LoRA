"""Preference pairs (chosen vs rejected) for DPO.

Two sources:

  1. ``build_pairs_from_dataset`` - offline, no GPU: chosen = the gold JSON,
     rejected = a *plausibly wrong* perturbation of it (corrupted digits,
     dropped/renamed key, prose wrapper, truncation). These mimic the typical
     failure modes of an SFT extractor.
  2. ``build_pairs_from_predictions`` - from an eval ``predictions_*.jsonl``:
     rejected = what the model actually said when it was wrong.

A pair references its image by ``(split, index)`` instead of embedding pixels,
so the jsonl stays tiny and the DPO stage re-reads images from the dataset.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .config import DataConfig
from .registry import get_dataset_builder

_PROSE_PREFIXES = (
    "Here is the extracted JSON:\n",
    "Sure! The receipt contains the following data: ",
    "```json\n",
)
_KEY_RENAMES = {
    "total": "sum",
    "subtotal": "sub_total",
    "tax": "vat",
    "menu": "items",
    "items": "menu",
    "vendor": "store",
    "date": "day",
}


def _corrupt_digits(text: str, rng: random.Random) -> str:
    digit_positions = [i for i, ch in enumerate(text) if ch.isdigit()]
    if not digit_positions:
        return text
    out = list(text)
    for pos in rng.sample(digit_positions, k=min(2, len(digit_positions))):
        old = out[pos]
        choices = [d for d in "0123456789" if d != old]
        out[pos] = rng.choice(choices)
    return "".join(out)


def perturb_target(target_json: str, rng: random.Random) -> str:
    """Return a wrong-but-plausible completion, guaranteed != the input."""
    target = json.loads(target_json)
    modes = ["digits", "drop_key", "rename_key", "prose", "truncate"]
    rng.shuffle(modes)
    for mode in modes:
        if mode == "digits":
            candidate = _corrupt_digits(target_json, rng)
        elif mode == "drop_key" and len(target) > 1:
            broken = dict(target)
            broken.pop(rng.choice(sorted(broken.keys())))
            candidate = json.dumps(broken, ensure_ascii=False, separators=(",", ":"))
        elif mode == "rename_key":
            keys = [k for k in target if k in _KEY_RENAMES]
            if not keys:
                continue
            broken = {(_KEY_RENAMES.get(k) if k == rng.choice(keys) else k): v
                      for k, v in target.items()}
            candidate = json.dumps(broken, ensure_ascii=False, separators=(",", ":"))
        elif mode == "prose":
            candidate = rng.choice(_PROSE_PREFIXES) + target_json
        elif mode == "truncate":
            candidate = target_json[: max(2, int(len(target_json) * 0.7))]
        else:
            continue
        if candidate != target_json:
            return candidate
    # Unreachable for any non-trivial JSON, but keep a hard guarantee.
    return target_json[:-1]


def build_pairs_from_dataset(
    data_cfg: DataConfig,
    split: str = "train",
    n: int | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """chosen = gold target, rejected = perturbed gold. Uses the TRAIN split by
    default so eval images never leak into preference training."""
    train_ds, eval_ds = get_dataset_builder(data_cfg.type)(data_cfg)
    ds = train_ds if split == "train" else eval_ds
    rng = random.Random(seed)
    n = len(ds) if n is None else min(n, len(ds))
    pairs = []
    for idx in range(n):
        gold = ds[idx]["target_json"]
        pairs.append(
            {
                "split": split,
                "index": idx,
                "chosen": gold,
                "rejected": perturb_target(gold, rng),
            }
        )
    return pairs


def build_pairs_from_predictions(predictions_path: str | Path, split: str) -> list[dict[str, Any]]:
    """Pairs from an eval predictions jsonl: keep only samples the model got
    wrong (invalid JSON or field_f1 < 1) - those are informative preferences."""
    pairs = []
    with open(predictions_path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            wrong = (not row.get("json_valid", False)) or row.get("field_f1", 0.0) < 1.0
            if not wrong:
                continue
            pairs.append(
                {
                    "split": split,
                    "index": row["index"],
                    "chosen": json.dumps(
                        row["gold"], ensure_ascii=False, separators=(",", ":")
                    ),
                    "rejected": row["raw"],
                }
            )
    return pairs


def save_pairs(pairs: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
    return path


def load_pairs(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
