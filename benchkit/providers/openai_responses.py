from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import requests

from .base import ProviderClient, map_status_to_failure
from ..env import load_env, get_openai_api_key
from ..types import ErrorInfo, FailureReason, LLMRequest, LLMResponse, Usage


def _extract_output_text(payload: Dict[str, Any]) -> str:
    chunks = []
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
    text = "".join(chunks).strip()
    if text:
        return text
    fallback = payload.get("output_text")
    if isinstance(fallback, str):
        return fallback.strip()
    return ""


def _extract_refusal(payload: Dict[str, Any]) -> Optional[str]:
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "refusal" and isinstance(part.get("refusal"), str):
                    return part["refusal"]
    return None


def _extract_usage(payload: Dict[str, Any]) -> Usage:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return Usage()
    return Usage(
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        total_tokens=usage.get("total_tokens"),
    )


class OpenAIResponsesProvider(ProviderClient):
    def __init__(self) -> None:
        load_env()
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.api_key = get_openai_api_key()

    def run(self, request: LLMRequest, timeout_s: int = 60) -> LLMResponse:
        if not self.api_key:
            return LLMResponse(
                text="",
                usage=Usage(),
                raw={},
                error=ErrorInfo(FailureReason.BAD_REQUEST, "Missing OPENAI_API_KEY"),
            )

        url = f"{self.base_url}/responses"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
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
