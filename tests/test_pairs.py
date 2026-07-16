"""Preference-pair building: perturbations, dataset mode, predictions mode."""

import json
import random

from lora_lab.config import DataConfig
from lora_lab.pairs import (
    build_pairs_from_dataset,
    build_pairs_from_predictions,
    load_pairs,
    perturb_target,
    save_pairs,
)

GOLD = '{"vendor":"CAFE","date":"2026-01-02","total":"12.50"}'


def test_perturb_always_differs():
    rng = random.Random(0)
    for _ in range(50):
        assert perturb_target(GOLD, rng) != GOLD


def test_perturb_produces_known_failure_modes():
    rng = random.Random(1)
    outputs = {perturb_target(GOLD, rng) for _ in range(200)}
    parsed, broken = 0, 0
    for out in outputs:
        try:
            json.loads(out)
            parsed += 1
        except json.JSONDecodeError:
            broken += 1
    # Both "valid JSON with wrong content" and "not JSON at all" must occur.
    assert parsed > 0 and broken > 0


def test_pairs_from_synthetic_dataset(tmp_path):
    import lora_lab.data  # noqa: F401  (registers synthetic_invoices)

    cfg = DataConfig(type="synthetic_invoices", max_train_samples=6, max_eval_samples=2)
    pairs = build_pairs_from_dataset(cfg, split="train", n=4, seed=0)
    assert len(pairs) == 4
    for pair in pairs:
        assert pair["split"] == "train"
        assert pair["chosen"] != pair["rejected"]
        json.loads(pair["chosen"])  # gold side is always valid JSON

    path = save_pairs(pairs, tmp_path / "pairs.jsonl")
    assert load_pairs(path) == pairs


def test_pairs_from_predictions_keep_only_mistakes(tmp_path):
    rows = [
        {"index": 0, "gold": {"a": "1"}, "raw": '{"a":"1"}', "json_valid": True, "field_f1": 1.0},
        {"index": 1, "gold": {"a": "2"}, "raw": '{"a":"9"}', "json_valid": True, "field_f1": 0.0},
        {"index": 2, "gold": {"a": "3"}, "raw": "not json", "json_valid": False, "field_f1": 0.0},
    ]
    path = tmp_path / "predictions.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    pairs = build_pairs_from_predictions(path, split="validation")
    assert [p["index"] for p in pairs] == [1, 2]  # the perfect sample is dropped
    assert pairs[0]["chosen"] == '{"a":"2"}'
    assert pairs[0]["rejected"] == '{"a":"9"}'
    assert all(p["split"] == "validation" for p in pairs)
