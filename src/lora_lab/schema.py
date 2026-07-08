"""JSON Schema as a first-class contract for the structured extractor.

The schema is used in three places so it is not just decoration:
  1. Prompt   - a compact description is appended to the instruction.
  2. Eval     - outputs are validated against the schema (``schema_valid_rate``).
  3. Decoding - optionally drives grammar-constrained generation (see ``decode``).

``jsonschema`` is a hard dependency; ``outlines`` (for constrained decoding) is
optional and imported lazily elsewhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_schema(spec: dict[str, Any] | str | None) -> dict[str, Any] | None:
    """Resolve a schema from an inline dict or a path to a .json file."""
    if spec is None:
        return None
    if isinstance(spec, dict):
        return spec
    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(f"JSON schema file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def validate_json(obj: Any, schema: dict[str, Any] | None) -> bool:
    """Return True if ``obj`` conforms to ``schema`` (or if no schema given)."""
    if schema is None:
        return obj is not None
    try:
        import jsonschema

        jsonschema.validate(instance=obj, schema=schema)
        return True
    except Exception:
        return False


def schema_to_prompt(schema: dict[str, Any] | None) -> str:
    """A compact, human-readable rendering of the schema for the prompt."""
    if schema is None:
        return ""
    props = schema.get("properties", {})
    if not props:
        return ""
    lines = ["The JSON must match this schema (keys and types):"]
    for key, spec in props.items():
        typ = spec.get("type", "any")
        desc = spec.get("description", "")
        suffix = f" - {desc}" if desc else ""
        lines.append(f"  - {key} ({typ}){suffix}")
    return "\n".join(lines)


def schema_top_level_keys(schema: dict[str, Any] | None) -> list[str]:
    if schema is None:
        return []
    return list(schema.get("properties", {}).keys())
