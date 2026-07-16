"""Train a QLoRA adapter.

Examples:
    # Full run on the 9B target:
    python scripts/02_train.py \
        --model configs/model/qwen3_5_9b_mm_qlora.yaml \
        --data  configs/data/cord.yaml \
        --train configs/train/sft_qlora.yaml

    # Fast pipeline smoke test on a tiny model (few steps, few samples):
    python scripts/02_train.py --smoke

    # Resume an interrupted run from its last checkpoint:
    python scripts/02_train.py --resume outputs/<run_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src/`` importable when running the script directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lora_lab.data  # noqa: E402, F401  (import registers the "cord" dataset builder)
from lora_lab.config import load_run_config, load_snapshot  # noqa: E402
from lora_lab.train import run  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a LoRA/QLoRA adapter.")
    p.add_argument("--model", default="configs/model/qwen3_5_9b_mm_qlora.yaml")
    p.add_argument("--data", default="configs/data/cord.yaml")
    p.add_argument("--train", default="configs/train/sft_qlora.yaml")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny model + few steps/samples to validate the pipeline fast.",
    )
    p.add_argument(
        "--resume",
        default=None,
        metavar="RUN_DIR",
        help="Resume from the last checkpoint of an existing run directory "
        "(uses its config.snapshot.yaml; --model/--data/--train are ignored).",
    )
    p.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="key=value",
        help="Dotted config overrides, e.g. --set model.lora.r=32 train.max_steps=50",
    )
    return p.parse_args()


def _parse_overrides(items: list[str]) -> dict:
    out: dict = {}
    for item in items:
        key, _, value = item.partition("=")
        # Best-effort type coercion.
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
    args = parse_args()
    overrides = _parse_overrides(args.set)

    if args.resume:
        cfg = load_snapshot(args.resume, overrides=overrides)
        run(cfg, resume=True)
        return 0

    model_cfg = args.model
    if args.smoke:
        model_cfg = "configs/model/tiny_mm_smoke.yaml"
        overrides.setdefault("train.max_steps", 10)
        overrides.setdefault("train.eval_steps", 5)
        overrides.setdefault("train.eval_strategy", "steps")
        overrides.setdefault("train.save_steps", 10)
        overrides.setdefault("train.logging_steps", 1)
        overrides.setdefault("train.gen_eval_steps", 5)
        overrides.setdefault("train.gen_eval_samples", 4)
        overrides.setdefault("data.max_train_samples", 16)
        overrides.setdefault("data.max_eval_samples", 8)

    cfg = load_run_config(model_cfg, args.data, args.train, overrides=overrides)
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
