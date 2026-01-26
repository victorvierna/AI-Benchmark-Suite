from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from .report import load_json, load_jsonl


def _model_key(m: Dict[str, Any]) -> str:
    name = m.get("name")
    provider = m.get("provider")
    label = m.get("label")
    return f"{provider}:{name}:{label or ''}"


def diff_runs(run_a: str, run_b: str) -> Dict[str, Any]:
    summary_a = load_json(os.path.join(run_a, "summary.json"))
    summary_b = load_json(os.path.join(run_b, "summary.json"))

    if summary_a.get("suite_id") != summary_b.get("suite_id"):
        raise ValueError("Suite IDs do not match")
    if summary_a.get("suite_version") != summary_b.get("suite_version"):
        raise ValueError("Suite versions do not match")

    models_a = { _model_key(m): m for m in summary_a.get("models", []) }
    models_b = { _model_key(m): m for m in summary_b.get("models", []) }

    model_diffs = []
    for key, m_a in models_a.items():
        m_b = models_b.get(key)
        if not m_b:
            continue
        model_diffs.append({
            "model": key,
            "pass_rate_delta": (m_b.get("pass_rate") or 0) - (m_a.get("pass_rate") or 0),
            "p95_delta": (m_b.get("latency_ms_p95") or 0) - (m_a.get("latency_ms_p95") or 0),
            "cost_delta": (m_b.get("cost_usd_total") or 0) - (m_a.get("cost_usd_total") or 0),
        })

    # case-level diffs (pass->fail and fail->pass)
    attempts_a = load_jsonl(os.path.join(run_a, "attempts.jsonl"))
    attempts_b = load_jsonl(os.path.join(run_b, "attempts.jsonl"))

    def idx(attempts: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
        out = {}
        for a in attempts:
            key = (
                a.get("model", {}).get("provider"),
                a.get("model", {}).get("name"),
                a.get("case_id"),
            )
            out[key] = a
        return out

    idx_a = idx(attempts_a)
    idx_b = idx(attempts_b)

    case_changes = []
    for key, a in idx_a.items():
        b = idx_b.get(key)
        if not b:
            continue
        pass_a = bool(a.get("eval", {}).get("pass"))
        pass_b = bool(b.get("eval", {}).get("pass"))
        if pass_a != pass_b:
            case_changes.append({
                "model": f"{key[0]}:{key[1]}",
                "case_id": key[2],
                "from": pass_a,
                "to": pass_b,
            })

    return {
        "suite_id": summary_a.get("suite_id"),
        "suite_version": summary_a.get("suite_version"),
        "model_diffs": model_diffs,
        "case_changes": case_changes,
    }


def print_diff(diff: Dict[str, Any]) -> None:
    print("\n=== Diff ===")
    print(f"Suite: {diff.get('suite_id')} v{diff.get('suite_version')}")
    for md in diff.get("model_diffs", []):
        print(f"- {md['model']}: pass_rate_delta={md['pass_rate_delta']:.3f} p95_delta={md['p95_delta']} cost_delta={md['cost_delta']}")
    if diff.get("case_changes"):
        print("\nCase changes:")
        for c in diff["case_changes"]:
            print(f"- {c['model']} {c['case_id']}: {c['from']} -> {c['to']}")
