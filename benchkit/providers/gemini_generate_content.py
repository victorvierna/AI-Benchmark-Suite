from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

from .base import ProviderClient, map_status_to_failure
from ..env import get_gemini_api_key, load_env
from ..types import ErrorInfo, FailureReason, LLMRequest, LLMResponse, Usage


_GENERATION_CONFIG_KEYS = {
    "candidateCount",
    "maxOutputTokens",
    "temperature",
    "topP",
    "topK",
    "stopSequences",
    "responseMimeType",
    "responseJsonSchema",
    "thinkingConfig",
}


def _split_messages(payload: Dict[str, Any]) -> tuple[Optional[str], List[Dict[str, Any]]]:
    system_parts: List[str] = []
    contents: List[Dict[str, Any]] = []
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
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": content}]})
    system_text = "\n\n".join(system_parts).strip() or None
    return system_text, contents


def _extract_schema(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    text_cfg = payload.get("text")
    if not isinstance(text_cfg, dict):
        return None
    fmt = text_cfg.get("format")
    if not isinstance(fmt, dict):
        return None
    schema = fmt.get("schema")
    return schema if isinstance(schema, dict) else None


def _build_generation_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}

    max_tokens = payload.get("max_output_tokens", payload.get("max_tokens"))
    if max_tokens is not None:
        cfg["maxOutputTokens"] = max_tokens

    key_map = {
        "temperature": "temperature",
        "top_p": "topP",
        "topP": "topP",
        "top_k": "topK",
        "topK": "topK",
        "candidate_count": "candidateCount",
        "candidateCount": "candidateCount",
        "stop_sequences": "stopSequences",
        "stopSequences": "stopSequences",
    }
    for source, target in key_map.items():
        if source in payload:
            cfg[target] = payload[source]

    extra_cfg = payload.get("generationConfig") or payload.get("generation_config")
    if isinstance(extra_cfg, dict):
        for key, value in extra_cfg.items():
            cfg[key] = value

    schema = _extract_schema(payload)
    if schema:
        cfg.setdefault("responseMimeType", "application/json")
        cfg.setdefault("responseJsonSchema", schema)

    return {key: value for key, value in cfg.items() if key in _GENERATION_CONFIG_KEYS}


def _extract_output_text(payload: Dict[str, Any]) -> str:
    chunks: List[str] = []
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
    return "".join(chunks).strip()


def _extract_refusal(payload: Dict[str, Any]) -> Optional[str]:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        finish_reason = candidate.get("finishReason")
        if finish_reason in {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}:
            return str(finish_reason)
    prompt_feedback = payload.get("promptFeedback")
    if isinstance(prompt_feedback, dict) and prompt_feedback.get("blockReason"):
        return str(prompt_feedback["blockReason"])
    return None


def _extract_usage(payload: Dict[str, Any]) -> Usage:
    usage = payload.get("usageMetadata")
    if not isinstance(usage, dict):
        return Usage()
    return Usage(
        input_tokens=usage.get("promptTokenCount"),
        output_tokens=usage.get("candidatesTokenCount"),
        total_tokens=usage.get("totalTokenCount"),
    )


class GeminiGenerateContentProvider(ProviderClient):
    def __init__(self) -> None:
        load_env()
        self.base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        self.api_key = get_gemini_api_key()

    def run(self, request: LLMRequest, timeout_s: int = 60) -> LLMResponse:
        if not self.api_key:
            return LLMResponse(
                text="",
                usage=Usage(),
                raw={},
                error=ErrorInfo(FailureReason.BAD_REQUEST, "Missing GEMINI_API_KEY"),
            )

        model = request.payload.get("model")
        if not isinstance(model, str) or not model:
            return LLMResponse(
                text="",
                usage=Usage(),
                raw={},
                error=ErrorInfo(FailureReason.BAD_REQUEST, "Missing model"),
            )

        system_text, contents = _split_messages(request.payload)
        body: Dict[str, Any] = {"contents": contents}
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}

        generation_config = _build_generation_config(request.payload)
        if generation_config:
            body["generationConfig"] = generation_config

        safety_settings = request.payload.get("safetySettings") or request.payload.get("safety_settings")
        if isinstance(safety_settings, list):
            body["safetySettings"] = safety_settings

        url = f"{self.base_url}/models/{model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
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

        return LLMResponse(
            text=_extract_output_text(payload),
            usage=_extract_usage(payload),
            raw=payload,
            refusal=_extract_refusal(payload),
            error=None,
        )
