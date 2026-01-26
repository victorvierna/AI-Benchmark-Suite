from __future__ import annotations

from typing import Any, Dict

from jinja2 import Environment, StrictUndefined


_env = Environment(undefined=StrictUndefined)


def render_template(template_str: str, context: Dict[str, Any]) -> str:
    tmpl = _env.from_string(template_str)
    return tmpl.render(**context)


def render_with_case(template_str: str, case: Dict[str, Any]) -> str:
    context = {
        "input": case.get("input", {}),
        "expected": case.get("expected", {}),
        "meta": case.get("meta", {}),
    }
    return render_template(template_str, context)
