from __future__ import annotations

import json
from typing import Any, Dict

from jsonschema import validate, ValidationError as SchemaValidationError

from ..types import EvalResult, FailureReason


def _extract_json(text: str) -> str:
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last < first:
        return text
    return text[first : last + 1]


def evaluate_json_schema(
    text: str,
    schema: Dict[str, Any],
    strict: bool = True,
    allow_extraction: bool = False,
) -> EvalResult:
    raw = text.strip()
    if not raw:
        return EvalResult(False, FailureReason.EMPTY_RESPONSE, {"error": "empty"})

    target = raw
    if not strict and allow_extraction:
        target = _extract_json(raw)

    try:
        parsed = json.loads(target)
    except json.JSONDecodeError as e:
        return EvalResult(False, FailureReason.PARSE_ERROR, {"error": str(e)})

    try:
        validate(instance=parsed, schema=schema)
    except SchemaValidationError as e:
        return EvalResult(False, FailureReason.SCHEMA_INVALID, {"error": str(e)})

    # strict mode: ensure no extra text outside JSON
    if strict:
        # If target is not the full raw string, there was extra content
        if target != raw:
            return EvalResult(False, FailureReason.SCHEMA_INVALID, {"error": "extra_text"})

    return EvalResult(True, FailureReason.NONE, {"schema_ok": True}, parsed=parsed)
