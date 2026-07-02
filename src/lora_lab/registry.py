"""A tiny name -> builder registry.

Adding a new dataset (or metric) becomes: write a builder function and decorate
it with ``@register_dataset("my_task")``. The training/eval orchestrators only
ever look things up by the string in the config, so the core never changes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

_DATASETS: dict[str, Callable[..., Any]] = {}
_METRICS: dict[str, Callable[..., Any]] = {}


def register_dataset(name: str) -> Callable[[T], T]:
    def deco(fn: T) -> T:
        if name in _DATASETS:
            raise ValueError(f"dataset '{name}' already registered")
        _DATASETS[name] = fn  # type: ignore[assignment]
        return fn

    return deco


def get_dataset_builder(name: str) -> Callable[..., Any]:
    if name not in _DATASETS:
        raise KeyError(
            f"Unknown dataset '{name}'. Registered: {sorted(_DATASETS)}. "
            "Did you import the module that defines it?"
        )
    return _DATASETS[name]


def register_metric(name: str) -> Callable[[T], T]:
    def deco(fn: T) -> T:
        _METRICS[name] = fn  # type: ignore[assignment]
        return fn

    return deco


def get_metric(name: str) -> Callable[..., Any]:
    return _METRICS[name]


def list_datasets() -> list[str]:
    return sorted(_DATASETS)
