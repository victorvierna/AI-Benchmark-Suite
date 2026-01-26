from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class FailureReason(str, Enum):
    NONE = "none"
    PARSE_ERROR = "parse_error"
    SCHEMA_INVALID = "schema_invalid"
    MISMATCH_FIELD = "mismatch_field"
    EMPTY_RESPONSE = "empty_response"
    REFUSAL = "refusal"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    BAD_REQUEST = "bad_request"
    SERVER_ERROR = "server_error"
    NETWORK_ERROR = "network_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class Usage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass
class ErrorInfo:
    error_type: FailureReason
    message: str
    status_code: Optional[int] = None


@dataclass
class LLMRequest:
    provider: str
    payload: Dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    usage: Usage
    raw: Dict[str, Any]
    refusal: Optional[str] = None
    error: Optional[ErrorInfo] = None


@dataclass
class EvalResult:
    passed: bool
    failure_reason: FailureReason
    details: Dict[str, Any]
    parsed: Optional[Dict[str, Any]] = None


@dataclass
class AttemptRecord:
    suite_id: str
    suite_version: int
    case_id: str
    model: Dict[str, Any]
    run_index: int
    timing: Dict[str, Any]
    usage: Dict[str, Any]
    cost_usd: Optional[float]
    eval: Dict[str, Any]
    request: Optional[Dict[str, Any]]
    response: Optional[Dict[str, Any]]
    is_warmup: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelSummary:
    provider: str
    name: str
    label: Optional[str]
    pass_rate: float
    passed: int
    total: int
    latency_ms_p50: Optional[int]
    latency_ms_p95: Optional[int]
    latency_ms_avg: Optional[int]
    cost_usd_total: Optional[float]
    cost_usd_avg: Optional[float]
    tokens_total: Optional[int]


@dataclass
class Summary:
    suite_id: str
    suite_version: int
    started_at: str
    finished_at: str
    git: Dict[str, Any]
    config: Dict[str, Any]
    models: List[ModelSummary]
    stopped_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["models"] = [asdict(m) for m in self.models]
        return data
