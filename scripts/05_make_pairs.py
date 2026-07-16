"""Build preference pairs (chosen vs rejected) for the DPO stage.

Examples:
    # Offline (no GPU): rejected = perturbed gold JSON, train split
    python scripts/05_make_pairs.py --data configs/data/cord.yaml --n 128 `
        --out outputs/pairs/cord_train.jsonl

    # From real model mistakes (run scripts/03_evaluate.py --out ... first).
    # NOTE: predictions come from the eval split - fine for experiments, but
    # remember those images then participate in training.
    python scripts/05_make_pairs.py --from-predictions outputs/eval/predictions_baseline.jsonl `
        --split validation --out outputs/pairs/cord_hard.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lora_lab.data  # noqa: E402, F401  (registers dataset builders)
from lora_lab.config import DataConfig, _load_yaml  # noqa: E402
from lora_lab.pairs import (  # noqa: E402
    build_pairs_from_dataset,
    build_pairs_from_predictions,
    save_pairs,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Build DPO preference pairs.")
    p.add_argument("--data", default="configs/data/cord.yaml")
    p.add_argument("--split", default="train", choices=["train", "validation"])
    p.add_argument("--n", type=int, default=128, help="Max pairs (perturb mode).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--from-predictions",
        default=None,
        metavar="JSONL",
        help="Use real model mistakes from an eval predictions file instead of perturbations.",
    )
    p.add_argument("--out", required=True, help="Output pairs .jsonl path.")
    args = p.parse_args()

    if args.from_predictions:
        pairs = build_pairs_from_predictions(args.from_predictions, split=args.split)
    else:
        data_cfg = DataConfig(**_load_yaml(args.data))
        pairs = build_pairs_from_dataset(data_cfg, split=args.split, n=args.n, seed=args.seed)

    path = save_pairs(pairs, args.out)
    print(f"[pairs] wrote {len(pairs)} pairs -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
