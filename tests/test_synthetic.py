"""Synthetic invoices: valid targets, schema conformance, pipeline contract."""

import json
from pathlib import Path

from fakes import FakeProcessor

from lora_lab.config import DataConfig
from lora_lab.data import IGNORE_INDEX, build_dataset
from lora_lab.schema import load_schema, validate_json
from lora_lab.synthetic import generate_invoice

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = load_schema(str(ROOT / "configs/data/synthetic_invoices_schema.json"))


def test_generate_is_deterministic():
    a, b = generate_invoice(7), generate_invoice(7)
    assert a["target_json"] == b["target_json"]
    assert generate_invoice(8)["target_json"] != a["target_json"]


def test_target_parses_and_matches_schema():
    for seed in range(5):
        sample = generate_invoice(seed)
        target = json.loads(sample["target_json"])
        assert validate_json(target, SCHEMA)
        # Money must be consistent: subtotal + tax == total (2-decimal money).
        assert abs(
            float(target["subtotal"]) + float(target["tax"]) - float(target["total"])
        ) < 0.01


def test_image_is_renderable():
    sample = generate_invoice(0)
    img = sample["image"]
    assert img.size[0] > 100 and img.size[1] > 100
    assert img.mode == "RGB"


def _data_cfg(**kwargs) -> DataConfig:
    return DataConfig(
        type="synthetic_invoices",
        max_train_samples=kwargs.get("train", 8),
        max_eval_samples=kwargs.get("eval", 4),
    )


def test_registry_builds_disjoint_splits():
    train_ds, eval_ds, _tc, _ec = build_dataset(_data_cfg(), FakeProcessor())
    assert len(train_ds) == 8 and len(eval_ds) == 4
    train_targets = {train_ds[i]["target_json"] for i in range(len(train_ds))}
    eval_targets = {eval_ds[i]["target_json"] for i in range(len(eval_ds))}
    assert not (train_targets & eval_targets)  # no leakage


def test_collator_masks_synthetic_batch():
    train_ds, _eval_ds, collator, _ec = build_dataset(_data_cfg(), FakeProcessor())
    # PIL images work with the real collator + fake processor path.
    batch = collator([train_ds[0], train_ds[1]])
    assert batch["labels"].shape == batch["input_ids"].shape
    # Prompt/image region must be masked, answer region must not be empty.
    assert (batch["labels"][0][:4] == IGNORE_INDEX).all()
    assert (batch["labels"] != IGNORE_INDEX).any()
