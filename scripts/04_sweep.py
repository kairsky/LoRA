"""Hyper-parameter sweep runner.

Trains + evaluates the cartesian product of a grid of config overrides and
writes a comparison table. Great for building intuition about how LoRA `r`,
`lora_alpha`, learning rate, DoRA/rsLoRA, etc. affect the metrics.

Example:
    python scripts/04_sweep.py --sweep configs/sweep/lora_rank.yaml
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lora_lab.data  # noqa: E402, F401  (registers "cord")
from lora_lab.config import _load_yaml, load_run_config  # noqa: E402
from lora_lab.data import build_dataset  # noqa: E402
from lora_lab.eval import evaluate  # noqa: E402
from lora_lab.train import run  # noqa: E402


def _expand_grid(grid: dict[str, list]) -> list[dict]:
    """Cartesian product of {key: [values]} -> list of {key: value} dicts."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    combos = itertools.product(*[grid[k] for k in keys])
    return [dict(zip(keys, values)) for values in combos]


def main() -> int:
    import torch

    p = argparse.ArgumentParser(description="Run a hyper-parameter sweep.")
    p.add_argument("--sweep", required=True, help="Path to a sweep YAML config.")
    args = p.parse_args()

    spec = _load_yaml(args.sweep)
    name = spec.get("name", "sweep")
    fixed = spec.get("fixed", {})
    grid = spec.get("grid", {})
    eval_cfg = spec.get("eval", {})
    combos = _expand_grid(grid)

    out_dir = ROOT / "outputs" / "sweeps" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] {name}: {len(combos)} runs -> {out_dir}")

    rows: list[dict] = []
    grid_keys = list(grid.keys())
    for idx, combo in enumerate(combos, 1):
        overrides = {**fixed, **combo}
        cfg = load_run_config(spec["model"], spec["data"], spec["train"], overrides=overrides)
        print(f"\n[sweep] run {idx}/{len(combos)}: {combo}")

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

        row = {k.split(".")[-1]: v for k, v in combo.items()}
        row.update(
            {
                "run_id": cfg.run_id,
                "json_valid_rate": metrics["json_valid_rate"],
                "schema_valid_rate": metrics["schema_valid_rate"],
                "field_f1": metrics["field_f1"],
                "exact_match": metrics["exact_match"],
            }
        )
        rows.append(row)
        print(f"[sweep] -> {json.dumps({k: row[k] for k in ('field_f1', 'exact_match')})}")

        # Free VRAM between runs (each combo builds its own PEFT model).
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Write results table.
    fieldnames = [k.split(".")[-1] for k in grid_keys] + [
        "field_f1",
        "exact_match",
        "json_valid_rate",
        "schema_valid_rate",
        "run_id",
    ]
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[sweep] done. Results -> {csv_path}\n")
    # Pretty console table sorted by field_f1.
    for row in sorted(rows, key=lambda r: r["field_f1"], reverse=True):
        keys = " ".join(f"{k.split('.')[-1]}={combo_v}" for k, combo_v in zip(grid_keys, [row[k.split('.')[-1]] for k in grid_keys]))
        print(f"  {keys:40s} f1={row['field_f1']:.4f} em={row['exact_match']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
