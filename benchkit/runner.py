from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .config import ModelsConfig, SuiteConfig
from .dataset import filter_cases, load_cases
from .evaluators import evaluate_all, evaluate_exact_fields, evaluate_json_schema
from .pricing import compute_cost_usd
from .providers.openai_responses import OpenAIResponsesProvider
from .template import render_with_case
from .types import AttemptRecord, EvalResult, FailureReason, LLMRequest, ModelSummary, Summary, Usage
from .redact import redact_text


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _git_info() -> Dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = None
    try:
        dirty = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(dirty)
    except Exception:
        dirty = None
    return {"commit": commit, "dirty": dirty}


def _percentile(values: List[int], p: float) -> Optional[int]:
    if not values:
        return None
    values_sorted = sorted(values)
    k = int(round((len(values_sorted) - 1) * p))
    return values_sorted[k]


def _build_request(suite: SuiteConfig, model: Dict[str, Any], case: Dict[str, Any]) -> LLMRequest:
    user_text = render_with_case(suite.request.user_template, case)
    messages = []
    if suite.request.system:
        messages.append({"role": "system", "content": suite.request.system})
    messages.append({"role": "user", "content": user_text})

    payload: Dict[str, Any] = {
        "model": model["name"],
        "input": messages,
    }

    if suite.request.response_format:
        payload["text"] = {"format": suite.request.response_format.model_dump(exclude_none=True)}

    # merge params: suite first, then model overrides
    payload.update(suite.request.params)
    payload.update(model.get("params") or {})

    return LLMRequest(provider=suite.provider, payload=payload)


def _case_expected(case: Dict[str, Any], expected_from: str) -> Any:
    if expected_from.startswith("expected."):
        key = expected_from.split(".", 1)[1]
        return (case.get("expected") or {}).get(key)
    if expected_from.startswith("input."):
        key = expected_from.split(".", 1)[1]
        return (case.get("input") or {}).get(key)
    return None


def _eval_case(suite: SuiteConfig, case: Dict[str, Any], response_text: str) -> EvalResult:
    evaluator = suite.evaluation.evaluator
    eval_dict = evaluator.model_dump()

    def eval_one(cfg: Dict[str, Any], parsed: Optional[Dict[str, Any]]) -> EvalResult:
        etype = cfg.get("type")
        if etype == "json_schema":
            schema = cfg.get("schema")
            if not schema and suite.request.response_format:
                schema = suite.request.response_format.schema
            if not schema:
                return EvalResult(False, FailureReason.SCHEMA_INVALID, {"error": "missing schema"})
            strict = cfg.get("strict", True)
            allow_extraction = cfg.get("allow_extraction", False)
            return evaluate_json_schema(response_text, schema, strict=strict, allow_extraction=allow_extraction)
        if etype == "exact_fields":
            if parsed is None:
                return EvalResult(False, FailureReason.PARSE_ERROR, {"error": "missing parsed json"})
            fields = cfg.get("fields") or []
            expected_map: List[Tuple[str, Any]] = []
            for f in fields:
                path = f.get("path")
                expected_from = f.get("expected_from")
                if not path or not expected_from:
                    continue
                expected_val = _case_expected(case, expected_from)
                expected_map.append((path, expected_val))
            return evaluate_exact_fields(parsed, expected_map)
        return EvalResult(False, FailureReason.UNKNOWN_ERROR, {"error": f"Unknown evaluator: {etype}"})

    if eval_dict.get("type") == "composite":
        results: List[EvalResult] = []
        parsed: Optional[Dict[str, Any]] = None
        for cfg in eval_dict.get("all") or eval_dict.get("items") or eval_dict.get("evaluators") or []:
            cfg_dict = cfg if isinstance(cfg, dict) else getattr(cfg, "model_dump", lambda: cfg)()
            res = eval_one(cfg_dict, parsed)
            if res.parsed is not None:
                parsed = res.parsed
            results.append(res)
            if not res.passed:
                break
        return evaluate_all(results)

    # single evaluator
    return eval_one(eval_dict, None)


def run_suite(
    suite: SuiteConfig,
    suite_path: str,
    models: ModelsConfig,
    pricing: Optional[Any],
    provider_override: Optional[Any] = None,
    runs: int = 1,
    limit: Optional[int] = None,
    filters: Optional[List[str]] = None,
    warmup: int = 0,
    max_cost_usd: Optional[float] = None,
    max_requests: Optional[int] = None,
    max_time_s: Optional[int] = None,
    out_dir: Optional[str] = None,
    redact: bool = False,
) -> str:
    dataset_path = os.path.join(os.path.dirname(suite_path), suite.dataset.path)
    cases = load_cases(dataset_path)
    cases = filter_cases(cases, filters, limit)
    if not cases:
        raise ValueError("No cases to run")

    bench_root = os.path.abspath(os.path.join(os.path.dirname(suite_path), "..", ".."))
    results_root = out_dir or os.path.join(bench_root, "results", suite.id, _now_iso().replace(":", "-"))
    os.makedirs(results_root, exist_ok=True)
    attempts_path = os.path.join(results_root, "attempts.jsonl")

    # provider selection
    if provider_override is not None:
        provider = provider_override
    else:
        if suite.provider == "openai":
            provider = OpenAIResponsesProvider()
        else:
            raise ValueError(f"Unsupported provider: {suite.provider}")

    started_at = _now_iso()
    attempts: List[AttemptRecord] = []
    stopped_reason: Optional[str] = None
    total_requests = 0
    total_cost = 0.0
    start_time = time.time()
    if max_cost_usd is not None and pricing is None:
        print("[WARN] max_cost_usd set but pricing config missing; cost guard may be inaccurate.")

    def should_stop() -> bool:
        nonlocal stopped_reason
        if max_requests is not None and total_requests >= max_requests:
            stopped_reason = "max_requests"
            return True
        if max_time_s is not None and (time.time() - start_time) >= max_time_s:
            stopped_reason = "max_time"
            return True
        if max_cost_usd is not None and total_cost >= max_cost_usd:
            stopped_reason = "max_cost_usd"
            return True
        return False

    output_cfg = suite.output or {}
    save_requests = output_cfg.get("save_requests", True)
    save_raw_responses = output_cfg.get("save_raw_responses", True)
    redact_enabled = redact or bool(suite.redact and suite.redact.enabled)

    with open(attempts_path, "w", encoding="utf-8") as attempts_file:
        for run_idx in range(1, runs + 1):
            for model in models.models:
                # warmup
                for w in range(warmup):
                    if should_stop():
                        break
                    case = cases[w % len(cases)]
                    attempt = _run_attempt(
                        suite,
                        model,
                        case,
                        provider,
                        pricing,
                        run_idx,
                        is_warmup=True,
                        redact=redact_enabled,
                        save_requests=save_requests,
                        save_raw_responses=save_raw_responses,
                    )
                    attempts_file.write(json.dumps(attempt.to_dict()) + "\n")
                    total_requests += 1
                    if attempt.cost_usd:
                        total_cost += attempt.cost_usd
                if should_stop():
                    break

                for case in cases:
                    if should_stop():
                        break
                    attempt = _run_attempt(
                        suite,
                        model,
                        case,
                        provider,
                        pricing,
                        run_idx,
                        is_warmup=False,
                        redact=redact_enabled,
                        save_requests=save_requests,
                        save_raw_responses=save_raw_responses,
                    )
                    attempts_file.write(json.dumps(attempt.to_dict()) + "\n")
                    total_requests += 1
                    if attempt.cost_usd:
                        total_cost += attempt.cost_usd
                if should_stop():
                    break
            if should_stop():
                break

    # compute summary from attempts file
    attempts = _load_attempts(attempts_path)
    summary = _summarize(
        suite,
        models,
        attempts,
        started_at,
        stopped_reason,
        config_overrides={
            "runs": runs,
            "limit": limit,
            "filters": filters,
            "warmup": warmup,
            "max_cost_usd": max_cost_usd,
            "max_requests": max_requests,
            "max_time_s": max_time_s,
        },
    )
    summary_path = os.path.join(results_root, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2)

    return results_root


# helpers

def _load_attempts(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _summarize(
    suite: SuiteConfig,
    models: ModelsConfig,
    attempts: List[Dict[str, Any]],
    started_at: str,
    stopped_reason: Optional[str],
    config_overrides: Optional[Dict[str, Any]] = None,
) -> Summary:
    finished_at = _now_iso()
    summaries: List[ModelSummary] = []
    for model in models.models:
        label = model.label or model.name
        model_attempts = [
            a
            for a in attempts
            if a.get("model", {}).get("name") == model.name
            and a.get("model", {}).get("provider") == model.provider
            and not a.get("is_warmup")
        ]
        total = len(model_attempts)
        passed = sum(1 for a in model_attempts if a.get("eval", {}).get("pass"))
        pass_rate = (passed / total) if total else 0.0
        latencies = [a.get("timing", {}).get("latency_ms") for a in model_attempts if a.get("timing", {}).get("latency_ms") is not None]
        latency_p50 = _percentile(latencies, 0.50) if latencies else None
        latency_p95 = _percentile(latencies, 0.95) if latencies else None
        latency_avg = int(sum(latencies) / len(latencies)) if latencies else None
        costs = [a.get("cost_usd") for a in model_attempts if a.get("cost_usd") is not None]
        cost_total = float(sum(costs)) if costs else None
        cost_avg = float(sum(costs) / len(costs)) if costs else None
        tokens_total = 0
        for a in model_attempts:
            usage = a.get("usage", {})
            if usage and usage.get("total_tokens"):
                tokens_total += int(usage.get("total_tokens"))
        summaries.append(ModelSummary(
            provider=model.provider,
            name=model.name,
            label=label,
            pass_rate=pass_rate,
            passed=passed,
            total=total,
            latency_ms_p50=latency_p50,
            latency_ms_p95=latency_p95,
            latency_ms_avg=latency_avg,
            cost_usd_total=cost_total,
            cost_usd_avg=cost_avg,
            tokens_total=tokens_total or None,
        ))

    return Summary(
        suite_id=suite.id,
        suite_version=suite.version,
        started_at=started_at,
        finished_at=finished_at,
        git=_git_info(),
        config=config_overrides or {},
        models=summaries,
        stopped_reason=stopped_reason,
    )


def _run_attempt(
    suite: SuiteConfig,
    model: Any,
    case: Dict[str, Any],
    provider: Any,
    pricing: Optional[Any],
    run_index: int,
    is_warmup: bool,
    redact: bool,
    save_requests: bool,
    save_raw_responses: bool,
) -> AttemptRecord:
    request = _build_request(suite, model.model_dump(), case)
    start = time.time()
    response = provider.run(request)
    latency_ms = int((time.time() - start) * 1000)

    eval_result: EvalResult
    if response.error:
        eval_result = EvalResult(False, response.error.error_type, {"error": response.error.message})
    elif response.refusal:
        eval_result = EvalResult(False, FailureReason.REFUSAL, {"refusal": True})
    else:
        eval_result = _eval_case(suite, case, response.text)

    usage = response.usage or Usage()

    cost = None
    if pricing is not None:
        cost = compute_cost_usd(pricing, model.provider, model.name, usage, model.pricing_key)

    # redaction
    if redact and suite.redact and suite.redact.patterns:
        response_text = redact_text(response.text, suite.redact.patterns, suite.redact.replacement)
    else:
        response_text = response.text

    request_payload = None
    if save_requests:
        request_payload = request.payload
        if redact and suite.redact and suite.redact.patterns:
            request_payload = _redact_request_payload(
                request_payload,
                suite.redact.patterns,
                suite.redact.replacement,
            )

    response_raw = response.raw if save_raw_responses else None

    attempt = AttemptRecord(
        suite_id=suite.id,
        suite_version=suite.version,
        case_id=case.get("id"),
        model={
            "provider": model.provider,
            "name": model.name,
            "label": model.label,
        },
        run_index=run_index,
        timing={"latency_ms": latency_ms},
        usage={
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
        },
        cost_usd=cost,
        eval={
            "pass": eval_result.passed,
            "failure_reason": eval_result.failure_reason.value,
            "details": eval_result.details,
        },
        request={"payload": request_payload} if save_requests else None,
        response={"text": response_text, "raw": response_raw, "refusal": response.refusal},
        is_warmup=is_warmup,
    )
    return attempt


def _redact_request_payload(payload: Dict[str, Any], patterns: List[str], replacement: str) -> Dict[str, Any]:
    try:
        redacted = json.loads(json.dumps(payload))
    except Exception:
        return payload
    items = redacted.get("input")
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                item["content"] = redact_text(item["content"], patterns, replacement)
    return redacted


 
