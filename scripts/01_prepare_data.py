"""Inspect / pre-download the dataset and preview the image->JSON samples.

This does NOT need a GPU. It verifies the dataset loads, normalizes a few
samples into our target schema, and prints them so you can sanity-check the
labels before spending GPU time.

Example:
    python scripts/01_prepare_data.py --data configs/data/cord.yaml --n 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lora_lab.data  # noqa: E402, F401  (registers "cord")
from lora_lab.config import DataConfig, _load_yaml  # noqa: E402
from lora_lab.registry import get_dataset_builder  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="configs/data/cord.yaml")
    p.add_argument("--n", type=int, default=3, help="How many samples to preview")
    args = p.parse_args()

    data_cfg = DataConfig(**_load_yaml(args.data))
    # Preview only: cap samples so it downloads/normalizes quickly.
    data_cfg.max_train_samples = args.n
    data_cfg.max_eval_samples = args.n

    builder = get_dataset_builder(data_cfg.type)
    train_ds, eval_ds = builder(data_cfg)

    print(f"[data] type={data_cfg.type} dataset_id={data_cfg.dataset_id}")
    print(f"[data] train={len(train_ds)} eval={len(eval_ds)} columns={train_ds.column_names}\n")

    for i in range(min(args.n, len(train_ds))):
        row = train_ds[i]
        img = row["image"]
        print(f"--- sample {i} --- image size={getattr(img, 'size', '?')}")
        target = json.loads(row["target_json"])
        print(json.dumps(target, indent=2, ensure_ascii=False))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
