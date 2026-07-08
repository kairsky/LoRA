"""Evaluate baseline vs LoRA-adapted model on the same held-out set.

Examples:
    # Compare base model against a trained adapter:
    python scripts/03_evaluate.py \
        --model configs/model/qwen3_5_9b_mm_qlora.yaml \
        --data  configs/data/cord.yaml \
        --adapter outputs/<run_id>/adapter \
        --limit 100

    # Baseline only (no adapter):
    python scripts/03_evaluate.py --adapter none
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lora_lab.data  # noqa: E402, F401  (registers "cord")
from lora_lab.config import load_run_config  # noqa: E402
from lora_lab.eval import evaluate  # noqa: E402
from lora_lab.model import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate baseline vs adapter.")
    p.add_argument("--model", default="configs/model/qwen3_5_9b_mm_qlora.yaml")
    p.add_argument("--data", default="configs/data/cord.yaml")
    p.add_argument("--train", default="configs/train/sft_qlora.yaml")
    p.add_argument(
        "--adapter",
        default="none",
        help="Path to a trained adapter dir, or 'none' for baseline only.",
    )
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--out",
        default=None,
        help="Directory to write predictions.jsonl (baseline/adapter). Default: none.",
    )
    p.add_argument(
        "--constrained",
        action="store_true",
        help="Use grammar-constrained JSON decoding (requires .[constrained] extras).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_run_config(args.model, args.data, args.train)
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Build the (quantized) base model wrapped for PEFT; this is our baseline too.
    model, processor, _targets = build_model(cfg.model, verbose=False)
    _, eval_ds, _tc, _ec = _load_eval(cfg, processor)

    print(f"\n== Baseline (no adapter) on {min(args.limit, len(eval_ds))} samples ==")
    base_metrics = evaluate(
        model,
        processor,
        eval_ds,
        cfg.data,
        limit=args.limit,
        batch_size=args.batch_size,
        predictions_path=(out_dir / "predictions_baseline.jsonl") if out_dir else None,
        constrained=args.constrained,
    )
    print(json.dumps(base_metrics, indent=2, ensure_ascii=False))

    results = {"baseline": base_metrics}

    if args.adapter and args.adapter.lower() != "none":
        model.load_adapter(args.adapter, adapter_name="trained")
        model.set_adapter("trained")
        print(f"\n== Adapter ({args.adapter}) ==")
        adapter_metrics = evaluate(
            model,
            processor,
            eval_ds,
            cfg.data,
            limit=args.limit,
            batch_size=args.batch_size,
            predictions_path=(out_dir / "predictions_adapter.jsonl") if out_dir else None,
            constrained=args.constrained,
        )
        print(json.dumps(adapter_metrics, indent=2, ensure_ascii=False))
        results["adapter"] = adapter_metrics

        print("\n== Delta (adapter - baseline) ==")
        for k in ("json_valid_rate", "schema_valid_rate", "field_f1", "exact_match"):
            print(f"  {k}: {adapter_metrics[k] - base_metrics[k]:+.4f}")

    if out_dir:
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        print(f"\n[eval] wrote metrics + predictions -> {out_dir}")

    return 0


def _load_eval(cfg, processor):
    from lora_lab.data import build_dataset

    return build_dataset(cfg.data, processor)


if __name__ == "__main__":
    raise SystemExit(main())
