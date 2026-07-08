from pathlib import Path

from lora_lab.schema import (
    load_schema,
    schema_to_prompt,
    schema_top_level_keys,
    validate_json,
)

ROOT = Path(__file__).resolve().parents[1]
CORD_SCHEMA = ROOT / "configs" / "data" / "cord_schema.json"


def test_load_schema_from_path():
    schema = load_schema(str(CORD_SCHEMA))
    assert schema["type"] == "object"
    assert set(schema_top_level_keys(schema)) == {"menu", "subtotal", "tax", "total"}


def test_load_schema_inline_and_none():
    inline = {"type": "object", "properties": {"a": {"type": "string"}}}
    assert load_schema(inline) is inline
    assert load_schema(None) is None


def test_validate_valid_object():
    schema = load_schema(str(CORD_SCHEMA))
    obj = {
        "menu": [{"name": "Coke", "count": "1", "price": "3"}],
        "subtotal": "3",
        "tax": "0",
        "total": "3",
    }
    assert validate_json(obj, schema) is True


def test_validate_invalid_object():
    schema = load_schema(str(CORD_SCHEMA))
    # Missing required keys / wrong types.
    assert validate_json({"menu": "not-a-list"}, schema) is False
    assert validate_json({"total": 3}, schema) is False  # number, not string; missing keys


def test_validate_no_schema_falls_back_to_not_none():
    assert validate_json({"anything": 1}, None) is True
    assert validate_json(None, None) is False


def test_schema_to_prompt_mentions_keys():
    schema = load_schema(str(CORD_SCHEMA))
    text = schema_to_prompt(schema)
    for key in ("menu", "subtotal", "tax", "total"):
        assert key in text
