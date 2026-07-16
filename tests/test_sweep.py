"""Sweep expansion: grid product, variants, and their combination."""

from lora_lab.sweep import expand_grid, expand_runs


def test_grid_product():
    combos = expand_grid({"a": [1, 2], "b": [10, 20]})
    assert len(combos) == 4
    assert {"a": 1, "b": 10} in combos
    assert {"a": 2, "b": 20} in combos


def test_empty_grid_is_single_run():
    assert expand_grid(None) == [{}]
    assert expand_runs(None, None) == [("", {})]


def test_variants_without_grid():
    runs = expand_runs(
        None,
        [
            {"label": "lora"},
            {"label": "dora", "model.lora.use_dora": True},
        ],
    )
    assert runs == [
        ("lora", {}),
        ("dora", {"model.lora.use_dora": True}),
    ]


def test_variants_cross_grid():
    runs = expand_runs(
        {"model.lora.r": [8, 16]},
        [{"label": "lora"}, {"label": "dora", "model.lora.use_dora": True}],
    )
    assert len(runs) == 4
    assert ("dora", {"model.lora.r": 16, "model.lora.use_dora": True}) in runs


def test_variant_label_defaults_to_overrides():
    runs = expand_runs(None, [{"x": 1}, {}])
    assert runs[0][0] == "x=1"
    assert runs[1][0] == "base"
