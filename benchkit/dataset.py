from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional


class DatasetError(ValueError):
    pass


def load_cases(path: str) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    seen_ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise DatasetError(f"Invalid JSON on line {idx}: {e}") from e
            if not isinstance(obj, dict):
                raise DatasetError(f"Case on line {idx} must be an object")
            case_id = obj.get("id")
            if not isinstance(case_id, str) or not case_id.strip():
                raise DatasetError(f"Case on line {idx} missing valid 'id'")
            if case_id in seen_ids:
                raise DatasetError(f"Duplicate case id '{case_id}' on line {idx}")
            seen_ids.add(case_id)
            inp = obj.get("input")
            exp = obj.get("expected")
            if inp is None or not isinstance(inp, dict):
                raise DatasetError(f"Case '{case_id}' has invalid 'input' (must be object)")
            if exp is not None and not isinstance(exp, dict):
                raise DatasetError(f"Case '{case_id}' has invalid 'expected' (must be object)")
            cases.append(obj)
    return cases


def filter_cases(cases: List[Dict[str, Any]], filters: Optional[List[str]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    out = cases
    if filters:
        for f in filters:
            f = f.strip()
            if not f:
                continue
            if f.startswith("tag:"):
                tag = f.split(":", 1)[1]
                out = [c for c in out if tag in (c.get("tags") or [])]
            elif f.startswith("id:"):
                prefix = f.split(":", 1)[1]
                out = [c for c in out if str(c.get("id", "")).startswith(prefix)]
            else:
                raise DatasetError(f"Unknown filter '{f}'. Use tag: or id: prefix.")
    if limit is not None:
        out = out[: max(0, limit)]
    return out
