"""Hyper-parameter sweep runner.

Trains + evaluates every combination from a sweep YAML and writes a comparison
table. Two axes compose:

  grid:     cartesian product of numeric axes (r, alpha, lr, ...)
  variants: mutually exclusive method toggles (LoRA vs DoRA vs rsLoRA vs
            PiSSA ...) - each entry is a dict of dotted overrides + a "label"

Examples:
    python scripts/04_sweep.py --sweep configs/sweep/lora_rank.yaml
    python scripts/04_sweep.py --sweep configs/sweep/methods.yaml
    python scripts/04_sweep.py --sweep configs/sweep/quant.yaml
    python scripts/04_sweep.py --sweep configs/sweep/target_modules.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lora_lab.data  # noqa: E402, F401  (registers "cord")
from lora_lab.config import _load_yaml, load_run_config  # noqa: E402
from lora_lab.data import build_dataset  # noqa: E402
from lora_lab.eval import evaluate  # noqa: E402
from lora_lab.sweep import expand_runs  # noqa: E402
from lora_lab.train import run  # noqa: E402

METRIC_COLUMNS = ("field_f1", "exact_match", "json_valid_rate", "schema_valid_rate")


def main() -> int:
    import torch

    p = argparse.ArgumentParser(description="Run a hyper-parameter sweep.")
    p.add_argument("--sweep", required=True, help="Path to a sweep YAML config.")
    args = p.parse_args()

    spec = _load_yaml(args.sweep)
    name = spec.get("name", "sweep")
    fixed = spec.get("fixed", {})
    grid = spec.get("grid", {})
    variants = spec.get("variants", [])
    eval_cfg = spec.get("eval", {})
    runs = expand_runs(grid, variants)

    out_dir = ROOT / "outputs" / "sweeps" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] {name}: {len(runs)} runs -> {out_dir}")

    rows: list[dict] = []
    grid_keys = list(grid.keys())
    for idx, (label, combo) in enumerate(runs, 1):
        overrides = {**fixed, **combo}
        cfg = load_run_config(spec["model"], spec["data"], spec["train"], overrides=overrides)
        tag = f"{label}: {combo}" if label else f"{combo}"
        print(f"\n[sweep] run {idx}/{len(runs)}: {tag}")

        adapter_dir, model, processor = run(cfg, return_model=True)
        _, eval_ds, _tc, _ec = build_dataset(cfg.data, processor)
        metrics = evaluate(
            model,
            processor,
            eval_ds,
            cfg.data,
            limit=eval_cfg.get("limit", 32),
            batch_size=eval_cfg.get("batch_size", 4),
        )

        row: dict = {}
        if variants:
            row["variant"] = label
        row.update({k.split(".")[-1]: v for k, v in combo.items() if k in grid_keys})
        row.update({m: metrics[m] for m in METRIC_COLUMNS})
        row["run_id"] = cfg.run_id
        rows.append(row)
        print(f"[sweep] -> {json.dumps({k: row[k] for k in ('field_f1', 'exact_match')})}")

        # Free VRAM between runs (each combo builds its own PEFT model).
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Write results table.
    fieldnames = (["variant"] if variants else []) + [k.split(".")[-1] for k in grid_keys]
    fieldnames += list(METRIC_COLUMNS) + ["run_id"]
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[sweep] done. Results -> {csv_path}\n")
    # Pretty console table sorted by field_f1.
    axis_cols = fieldnames[: len(fieldnames) - len(METRIC_COLUMNS) - 1]
    for row in sorted(rows, key=lambda r: r["field_f1"], reverse=True):
        keys = " ".join(f"{c}={row[c]}" for c in axis_cols)
        print(f"  {keys:44s} f1={row['field_f1']:.4f} em={row['exact_match']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
