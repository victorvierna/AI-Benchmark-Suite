from benchkit.evaluators import evaluate_all, evaluate_exact_fields, evaluate_json_schema
from benchkit.types import FailureReason


def test_json_schema_ok():
    schema = {
        "type": "object",
        "properties": {"label": {"type": "string"}},
        "required": ["label"],
        "additionalProperties": False,
    }
    res = evaluate_json_schema('{"label":"ok"}', schema, strict=True)
    assert res.passed


def test_json_schema_extra_text_fails():
    schema = {
        "type": "object",
        "properties": {"label": {"type": "string"}},
        "required": ["label"],
        "additionalProperties": False,
    }
    res = evaluate_json_schema('hello {"label":"ok"}', schema, strict=True)
    assert not res.passed


def test_exact_fields_mismatch():
    parsed = {"label": "no"}
    res = evaluate_exact_fields(parsed, [("$.label", "ok")])
    assert not res.passed
    assert res.failure_reason == FailureReason.MISMATCH_FIELD


def test_composite():
    schema = {
        "type": "object",
        "properties": {"label": {"type": "string"}},
        "required": ["label"],
        "additionalProperties": False,
    }
    res1 = evaluate_json_schema('{"label":"ok"}', schema, strict=True)
    res2 = evaluate_exact_fields(res1.parsed, [("$.label", "ok")])
    res = evaluate_all([res1, res2])
    assert res.passed
