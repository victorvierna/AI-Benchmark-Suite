from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests

from .base import ProviderClient, map_status_to_failure
from .openai_responses import _extract_output_text, _extract_refusal, _extract_usage
from ..env import load_env, get_lmstudio_api_key
from ..types import ErrorInfo, FailureReason, LLMRequest, LLMResponse, Usage


class LMStudioResponsesProvider(ProviderClient):
    def __init__(self) -> None:
        load_env()
        self.base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")
        self.api_key = get_lmstudio_api_key()

    def run(self, request: LLMRequest, timeout_s: int = 60) -> LLMResponse:
        url = f"{self.base_url}/responses"
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = request.payload

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
        except requests.Timeout as e:
            return LLMResponse(
                text="",
                usage=Usage(),
                raw={},
                error=ErrorInfo(FailureReason.TIMEOUT, f"Timeout: {e}"),
            )
        except requests.RequestException as e:
            return LLMResponse(
                text="",
                usage=Usage(),
                raw={},
                error=ErrorInfo(FailureReason.NETWORK_ERROR, f"Network error: {e}"),
            )

        status = resp.status_code
        text = resp.text
        if status < 200 or status >= 300:
            failure = map_status_to_failure(status)
            return LLMResponse(
                text="",
                usage=Usage(),
                raw={"status": status, "body": text},
                error=ErrorInfo(FailureReason(failure), f"HTTP {status}", status_code=status),
            )

        try:
            payload = resp.json()
        except ValueError:
            return LLMResponse(
                text="",
                usage=Usage(),
                raw={"status": status, "body": text},
                error=ErrorInfo(FailureReason.PARSE_ERROR, "Invalid JSON in response"),
            )

        output_text = _extract_output_text(payload)
        refusal = _extract_refusal(payload)
        usage = _extract_usage(payload)

        return LLMResponse(
            text=output_text,
            usage=usage,
            raw=payload,
            refusal=refusal,
            error=None,
        )
