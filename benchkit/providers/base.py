from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..types import LLMRequest, LLMResponse


class ProviderClient(ABC):
    @abstractmethod
    def run(self, request: LLMRequest, timeout_s: int = 60) -> LLMResponse:
        raise NotImplementedError


def map_status_to_failure(status_code: int) -> str:
    if status_code == 429:
        return "rate_limited"
    if status_code == 400:
        return "bad_request"
    if status_code in (408, 504):
        return "timeout"
    if 500 <= status_code <= 599:
        return "server_error"
    return "unknown_error"
