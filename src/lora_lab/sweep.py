"""Sweep expansion: cartesian ``grid`` x named ``variants`` -> run overrides.

``grid`` is for numeric axes (r, alpha, lr, ...) where the cartesian product is
what you want. ``variants`` is for mutually exclusive method toggles (LoRA vs
DoRA vs rsLoRA vs PiSSA ...) where a product would produce nonsense like
DoRA+PiSSA at once; each variant is a dict of dotted overrides plus an optional
``label`` used in the results table.
"""

from __future__ import annotations

import itertools
from typing import Any

LABEL_KEY = "label"


def expand_grid(grid: dict[str, list] | None) -> list[dict[str, Any]]:
    """Cartesian product of ``{key: [values]}`` -> list of ``{key: value}``."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    combos = itertools.product(*[grid[k] for k in keys])
    return [dict(zip(keys, values, strict=True)) for values in combos]


def expand_runs(
    grid: dict[str, list] | None,
    variants: list[dict[str, Any]] | None,
) -> list[tuple[str, dict[str, Any]]]:
    """Return ``(label, overrides)`` per run: every variant x every grid combo."""
    combos = expand_grid(grid)
    if not variants:
        return [("", dict(combo)) for combo in combos]

    runs: list[tuple[str, dict[str, Any]]] = []
    for variant in variants:
        v = dict(variant)
        label = str(v.pop(LABEL_KEY, "")) or (
            "+".join(f"{k}={val}" for k, val in v.items()) or "base"
        )
        for combo in combos:
            runs.append((label, {**combo, **v}))
    return runs
