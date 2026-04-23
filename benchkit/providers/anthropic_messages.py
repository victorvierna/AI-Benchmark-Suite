from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

from .base import ProviderClient, map_status_to_failure
from ..env import get_anthropic_api_key, load_env
from ..types import ErrorInfo, FailureReason, LLMRequest, LLMResponse, Usage


def _split_messages(payload: Dict[str, Any]) -> tuple[Optional[str], List[Dict[str, Any]]]:
    system_parts: List[str] = []
    messages: List[Dict[str, Any]] = []
    for item in payload.get("input") or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if not isinstance(content, str):
            continue
        if role == "system":
            system_parts.append(content)
            continue
        if role not in {"user", "assistant"}:
            role = "user"
        messages.append({"role": role, "content": content})
    system_text = "\n\n".join(system_parts).strip() or None
    return system_text, messages


def _extract_schema(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    text_cfg = payload.get("text")
    if not isinstance(text_cfg, dict):
        return None
    fmt = text_cfg.get("format")
    if not isinstance(fmt, dict):
        return None
    schema = fmt.get("schema")
    return schema if isinstance(schema, dict) else None


def _extract_output_text(payload: Dict[str, Any]) -> str:
    chunks: List[str] = []
    content = payload.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                chunks.append(part["text"])
    return "".join(chunks).strip()


def _extract_usage(payload: Dict[str, Any]) -> Usage:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return Usage()
    input_tokens = 0
    saw_input = False
    for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            input_tokens += value
            saw_input = True
    output_tokens = usage.get("output_tokens")
    total_tokens = None
    if saw_input or isinstance(output_tokens, int):
        total_tokens = (input_tokens if saw_input else 0) + (output_tokens if isinstance(output_tokens, int) else 0)
    return Usage(
        input_tokens=input_tokens if saw_input else None,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


class AnthropicMessagesProvider(ProviderClient):
    def __init__(self) -> None:
        load_env()
        self.base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1").rstrip("/")
        self.api_key = get_anthropic_api_key()
        self.version = os.getenv("ANTHROPIC_VERSION", "2023-06-01")

    def run(self, request: LLMRequest, timeout_s: int = 60) -> LLMResponse:
        if not self.api_key:
            return LLMResponse(
                text="",
                usage=Usage(),
                raw={},
                error=ErrorInfo(FailureReason.BAD_REQUEST, "Missing ANTHROPIC_API_KEY"),
            )

        model = request.payload.get("model")
        if not isinstance(model, str) or not model:
            return LLMResponse(
                text="",
                usage=Usage(),
                raw={},
                error=ErrorInfo(FailureReason.BAD_REQUEST, "Missing model"),
            )

        system_text, messages = _split_messages(request.payload)
        max_tokens = request.payload.get("max_output_tokens", request.payload.get("max_tokens", 1024))
        body: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system_text:
            body["system"] = system_text
        if "temperature" in request.payload:
            body["temperature"] = request.payload["temperature"]
        if "top_p" in request.payload:
            body["top_p"] = request.payload["top_p"]
        if "top_k" in request.payload:
            body["top_k"] = request.payload["top_k"]
        if "stop_sequences" in request.payload:
            body["stop_sequences"] = request.payload["stop_sequences"]

        schema = _extract_schema(request.payload)
        if schema:
            body["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": schema,
                }
            }

        url = f"{self.base_url}/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.version,
        }

        try:
            resp = requests.post(url, json=body, headers=headers, timeout=timeout_s)
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

        refusal = "refusal" if payload.get("stop_reason") == "refusal" else None
        return LLMResponse(
            text=_extract_output_text(payload),
            usage=_extract_usage(payload),
            raw=payload,
            refusal=refusal,
            error=None,
        )
