from __future__ import annotations

import importlib
import os
from typing import List, Optional

import requests

from .config import load_models, load_pricing, load_suite, ValidationError
from .dataset import load_cases, DatasetError
from .env import (
    get_anthropic_api_key,
    get_gemini_api_key,
    get_lmstudio_api_key,
    get_openai_api_key,
    load_env,
)


def _check_import(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def run_doctor(suite_path: str, models_path: Optional[str], pricing_path: Optional[str], ping: bool = False) -> int:
    ok = True
    models = None

    print("\n=== BenchKit Doctor ===")
    load_env()

    # deps
    for mod in ["yaml", "jinja2", "jsonschema", "requests", "pydantic"]:
        if _check_import(mod):
            print(f"[OK] import {mod}")
        else:
            print(f"[FAIL] import {mod} (missing dependency)")
            ok = False

    # load configs
    try:
        suite = load_suite(suite_path)
        print("[OK] suite config")
    except (ValidationError, Exception) as e:
        print(f"[FAIL] suite config: {e}")
        ok = False
        suite = None

    if models_path:
        try:
            models = load_models(models_path)
            print("[OK] models config")
        except (ValidationError, Exception) as e:
            print(f"[FAIL] models config: {e}")
            ok = False

    if pricing_path:
        try:
            load_pricing(pricing_path)
            print("[OK] pricing config")
        except (ValidationError, Exception) as e:
            print(f"[FAIL] pricing config: {e}")
            ok = False

    if suite:
        try:
            dataset_path = os.path.join(os.path.dirname(suite_path), suite.dataset.path)
            load_cases(dataset_path)
            print("[OK] dataset")
        except DatasetError as e:
            print(f"[FAIL] dataset: {e}")
            ok = False

        providers = {suite.provider}
        if models:
            providers = {m.provider for m in models.models}
        ok = _check_provider_env(providers) and ok

    if ping:
        providers = {suite.provider} if suite else set()
        if models:
            providers = {m.provider for m in models.models}
        for provider in sorted(providers):
            if not _ping_provider(provider):
                ok = False

    return 0 if ok else 1


def _check_provider_env(providers: set[str]) -> bool:
    ok = True
    if "openai" in providers:
        if get_openai_api_key():
            print("[OK] OPENAI_API_KEY set")
        else:
            print("[FAIL] OPENAI_API_KEY missing")
            ok = False
    if "gemini" in providers:
        if get_gemini_api_key():
            print("[OK] GEMINI_API_KEY set")
        else:
            print("[FAIL] GEMINI_API_KEY missing")
            ok = False
    if "anthropic" in providers:
        if get_anthropic_api_key():
            print("[OK] ANTHROPIC_API_KEY set")
        else:
            print("[FAIL] ANTHROPIC_API_KEY missing")
            ok = False
    if "lmstudio" in providers:
        base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        print(f"[OK] LMSTUDIO_BASE_URL={base_url}")
    supported = {"openai", "gemini", "anthropic", "lmstudio"}
    for provider in sorted(providers - supported):
        print(f"[FAIL] unsupported provider {provider}")
        ok = False
    return ok


def _ping_provider(provider: str) -> bool:
    if provider == "openai":
        if _ping_openai():
            print("[OK] provider ping (openai)")
            return True
        print("[FAIL] provider ping (openai)")
        return False
    if provider == "lmstudio":
        if _ping_lmstudio():
            print("[OK] provider ping (lmstudio)")
            return True
        print("[FAIL] provider ping (lmstudio)")
        return False
    if provider == "gemini":
        if _ping_gemini():
            print("[OK] provider ping (gemini)")
            return True
        print("[FAIL] provider ping (gemini)")
        return False
    if provider == "anthropic":
        if _ping_anthropic():
            print("[OK] provider ping (anthropic)")
            return True
        print("[FAIL] provider ping (anthropic)")
        return False
    print(f"[WARN] ping skipped (unsupported provider: {provider})")
    return False


def _ping_openai(timeout_s: int = 10) -> bool:
    api_key = get_openai_api_key()
    if not api_key:
        return False
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = f"{base_url}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout_s)
    except requests.RequestException:
        return False
    return 200 <= resp.status_code < 300


def _ping_lmstudio(timeout_s: int = 5) -> bool:
    base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")
    url = f"{base_url}/models"
    headers = {"Content-Type": "application/json"}
    api_key = get_lmstudio_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = requests.get(url, headers=headers, timeout=timeout_s)
    except requests.RequestException:
        return False
    return 200 <= resp.status_code < 300


def _ping_gemini(timeout_s: int = 10) -> bool:
    api_key = get_gemini_api_key()
    if not api_key:
        return False
    base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
    url = f"{base_url}/models"
    headers = {"x-goog-api-key": api_key}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout_s)
    except requests.RequestException:
        return False
    return 200 <= resp.status_code < 300


def _ping_anthropic(timeout_s: int = 10) -> bool:
    api_key = get_anthropic_api_key()
    if not api_key:
        return False
    base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1").rstrip("/")
    version = os.getenv("ANTHROPIC_VERSION", "2023-06-01")
    url = f"{base_url}/models"
    headers = {"x-api-key": api_key, "anthropic-version": version}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout_s)
    except requests.RequestException:
        return False
    return 200 <= resp.status_code < 300
