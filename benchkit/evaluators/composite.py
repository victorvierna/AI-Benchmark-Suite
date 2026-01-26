from __future__ import annotations

from typing import Dict, List

from ..types import EvalResult, FailureReason


def evaluate_all(results: List[EvalResult]) -> EvalResult:
    merged: Dict[str, any] = {}
    parsed = None
    for res in results:
        merged.update(res.details or {})
        if res.parsed is not None:
            parsed = res.parsed
        if not res.passed:
            return EvalResult(False, res.failure_reason, merged, parsed=parsed)
    return EvalResult(True, FailureReason.NONE, merged, parsed=parsed)
