from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from jinja2 import Environment, FileSystemLoader

from .redact import redact_text


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    return rows


def render_html_report(results_dir: str, redact_patterns: Optional[List[str]] = None, replacement: str = "***") -> str:
    summary_path = os.path.join(results_dir, "summary.json")
    attempts_path = os.path.join(results_dir, "attempts.jsonl")
    summary = load_json(summary_path)
    attempts = load_jsonl(attempts_path)

    # redact if needed
    if redact_patterns:
        for row in attempts:
            resp = row.get("response") or {}
            if isinstance(resp, dict):
                text = resp.get("text")
                if isinstance(text, str):
                    resp["text"] = redact_text(text, redact_patterns, replacement)
            req = row.get("request") or {}
            if isinstance(req, dict):
                payload = req.get("payload")
                if isinstance(payload, dict):
                    # best-effort redact in prompt text
                    if isinstance(payload.get("input"), list):
                        for item in payload["input"]:
                            if isinstance(item, dict) and isinstance(item.get("content"), str):
                                item["content"] = redact_text(item["content"], redact_patterns, replacement)

    env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "templates")))
    tmpl = env.get_template("report.html.j2")
    html = tmpl.render(summary=summary, attempts=attempts)

    out_path = os.path.join(results_dir, "report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


def print_console_summary(summary: Dict[str, Any]) -> None:
    print("\n=== Summary ===")
    print(f"Suite: {summary.get('suite_id')} v{summary.get('suite_version')}")
    print(f"Started: {summary.get('started_at')}  Finished: {summary.get('finished_at')}")
    if summary.get("stopped_reason"):
        print(f"Stopped: {summary['stopped_reason']}")
    for m in summary.get("models", []):
        label = m.get("label") or m.get("name")
        print(f"- {label}: pass_rate={m.get('pass_rate'):.3f} total={m.get('passed')}/{m.get('total')} p50={m.get('latency_ms_p50')}ms p95={m.get('latency_ms_p95')}ms cost={m.get('cost_usd_total')}")
