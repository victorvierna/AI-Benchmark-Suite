from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..types import EvalResult, FailureReason


def _get_path(obj: Any, path: str) -> Any:
    if not path.startswith("$."):
        return None
    cur = obj
    parts = path[2:].split(".") if path else []
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur


def evaluate_exact_fields(parsed: Dict[str, Any], expected_map: List[Tuple[str, Any]]) -> EvalResult:
    details: Dict[str, Any] = {}
    for path, expected in expected_map:
        actual = _get_path(parsed, path)
        details[path] = {"actual": actual, "expected": expected}
        if actual != expected:
            return EvalResult(False, FailureReason.MISMATCH_FIELD, details)
    return EvalResult(True, FailureReason.NONE, details, parsed=parsed)
