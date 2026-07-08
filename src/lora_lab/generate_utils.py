"""Robust ``model.generate`` wrapper for multimodal models.

Some transformers VL models are self-contradictory about ``mm_token_type_ids``
(returned by the processor for M-RoPE): ``generate``'s ``_validate_model_kwargs``
rejects it as "not used by the model", yet the model's ``forward`` *requires* it
when image grids are present (e.g. Qwen2-VL in current transformers). Qwen3.5, by
contrast, accepts it directly.

``safe_generate`` first tries a normal call. If generate rejects a processor kwarg
as unused, it retries once with kwarg validation disabled, so the processor
outputs flow through to ``forward`` (which is what actually needs them).
"""

from __future__ import annotations

from typing import Any


def safe_generate(model, inputs: dict[str, Any], gen_kwargs: dict[str, Any]):
    try:
        return model.generate(**inputs, **gen_kwargs)
    except ValueError as exc:
        if "not used by the model" not in str(exc):
            raise

    # The transformers model that actually runs generate (unwrap PEFT).
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    original = base._validate_model_kwargs
    base._validate_model_kwargs = lambda *a, **k: None  # bypass the buggy check
    try:
        return model.generate(**inputs, **gen_kwargs)
    finally:
        base._validate_model_kwargs = original
