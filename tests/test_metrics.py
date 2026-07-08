from lora_lab.eval import extract_json, field_prf, normalize_value


def test_normalize_numbers():
    assert normalize_value("1,200.00") == "1200"
    assert normalize_value("12.50") == "12.5"
    assert normalize_value("$3.00") == "3"
    assert normalize_value("  Coke ") == "coke"


def test_extract_json_valid():
    assert extract_json('here is {"a": 1} done') == {"a": 1}


def test_extract_json_invalid():
    assert extract_json("no json here") is None
    assert extract_json("{broken json") is None


def test_field_prf_perfect_match():
    gold = {"total": "12.00", "menu": [{"name": "Coke", "price": "3.0"}]}
    pred = {"total": "12", "menu": [{"name": "coke", "price": "3"}]}
    p, r, f1 = field_prf(pred, gold)
    # Values are normalized, so these should match exactly.
    assert f1 == 1.0
    assert p == 1.0 and r == 1.0


def test_field_prf_partial():
    gold = {"a": "1", "b": "2"}
    pred = {"a": "1", "b": "99"}
    _, _, f1 = field_prf(pred, gold)
    assert 0.0 < f1 < 1.0


def test_field_prf_none_pred():
    assert field_prf(None, {"a": "1"}) == (0.0, 0.0, 0.0)
