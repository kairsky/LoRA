"""Optional grammar-constrained JSON decoding via ``outlines``.

If ``outlines`` is installed and compatible with the loaded model, this returns
a ``LogitsProcessorList`` that forces ``model.generate`` to emit only strings
that conform to the given JSON Schema - so ``json_valid_rate`` becomes 100% by
construction. If anything is unavailable (common on a bleeding-edge custom VLM
under Windows), it returns ``None`` and the caller falls back to free decoding.
"""

from __future__ import annotations

import json
from typing import Any


def build_json_logits_processor(model, processor, schema: dict[str, Any] | None):
    """Return a logits-processor list for schema-constrained JSON, or None."""
    if schema is None:
        return None
    try:
        from outlines.processors import JSONLogitsProcessor
        from transformers import LogitsProcessorList

        try:
            # Newer outlines expects a tokenizer wrapper.
            from outlines.models.transformers import TransformerTokenizer

            tokenizer = TransformerTokenizer(processor.tokenizer)
        except Exception:
            tokenizer = processor.tokenizer

        schema_str = json.dumps(schema)
        proc = JSONLogitsProcessor(schema_str, tokenizer)
        return LogitsProcessorList([proc])
    except Exception as exc:  # noqa: BLE001
        print(
            f"[constrained] outlines constrained decoding unavailable ({exc}); "
            "falling back to free decoding. Install extras: pip install -e .[constrained]"
        )
        return None
