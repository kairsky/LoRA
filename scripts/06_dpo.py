"""Preference-tune (DPO) on top of an SFT adapter.

Examples:
    # Full pipeline on the 9B target (after SFT):
    python scripts/05_make_pairs.py --data configs/data/cord.yaml --n 256 --out outputs/pairs/cord.jsonl
    python scripts/06_dpo.py `
        --model configs/model/qwen3_5_9b_mm_qlora.yaml `
        --data  configs/data/cord.yaml `
        --pairs outputs/pairs/cord.jsonl `
        --from-adapter outputs/<sft_run_id>/adapter

    # Smoke test: tiny model + synthetic invoices, fully offline
    python scripts/06_dpo.py --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lora_lab.data  # noqa: E402, F401  (registers dataset builders)
from lora_lab.config import DataConfig, _load_yaml, load_run_config  # noqa: E402
from lora_lab.dpo import run_dpo  # noqa: E402
from lora_lab.pairs import build_pairs_from_dataset, save_pairs  # noqa: E402


def _parse_overrides(items: list[str]) -> dict:
    out: dict = {}
    for item in items:
        key, _, value = item.partition("=")
        v: object = value
        for cast in (int, float):
            try:
                v = cast(value)
                break
            except ValueError:
                continue
        if value.lower() in ("true", "false"):
            v = value.lower() == "true"
        out[key] = v
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="DPO stage for the JSON extractor.")
    p.add_argument("--model", default="configs/model/qwen3_5_9b_mm_qlora.yaml")
    p.add_argument("--data", default="configs/data/cord.yaml")
    p.add_argument("--train", default="configs/train/dpo.yaml")
    p.add_argument("--pairs", default=None, help="Pairs .jsonl from 05_make_pairs.py.")
    p.add_argument(
        "--from-adapter",
        default=None,
        metavar="ADAPTER_DIR",
        help="SFT adapter to start from (recommended). Omit to train from scratch.",
    )
    p.add_argument("--smoke", action="store_true", help="Tiny model + synthetic pairs.")
    p.add_argument("--set", nargs="*", default=[], metavar="key=value")
    args = p.parse_args()

    overrides = _parse_overrides(args.set)
    model_cfg, data_cfg_path = args.model, args.data
    if args.smoke:
        model_cfg = "configs/model/tiny_mm_smoke.yaml"
        data_cfg_path = "configs/data/synthetic_invoices.yaml"
        overrides.setdefault("train.max_steps", 10)
        overrides.setdefault("train.logging_steps", 1)
        overrides.setdefault("train.save_steps", 10)
        overrides.setdefault("data.max_train_samples", 16)
        overrides.setdefault("data.max_eval_samples", 4)

    cfg = load_run_config(model_cfg, data_cfg_path, args.train, overrides=overrides)

    pairs_path = args.pairs
    if pairs_path is None:
        if not args.smoke:
            p.error("--pairs is required (or use --smoke)")
        data_cfg = DataConfig(**{**_load_yaml(data_cfg_path), "max_train_samples": 16})
        pairs = build_pairs_from_dataset(data_cfg, split="train", n=16, seed=0)
        pairs_path = save_pairs(pairs, cfg.run_dir / "pairs.jsonl")
        print(f"[dpo] smoke pairs -> {pairs_path}")

    run_dpo(cfg, pairs_path, sft_adapter=args.from_adapter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
