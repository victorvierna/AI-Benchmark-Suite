from __future__ import annotations

import importlib
import os
from typing import List, Optional

import requests

from .config import load_models, load_pricing, load_suite, ValidationError
from .dataset import load_cases, DatasetError


def _check_import(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def run_doctor(suite_path: str, models_path: Optional[str], pricing_path: Optional[str], ping: bool = False) -> int:
    ok = True

    print("\n=== BenchKit Doctor ===")

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
            load_models(models_path)
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

        if suite.provider == "openai":
            if os.getenv("OPENAI_API_KEY"):
                print("[OK] OPENAI_API_KEY set")
            else:
                print("[FAIL] OPENAI_API_KEY missing")
                ok = False

    if ping:
        if suite and suite.provider == "openai":
            if _ping_openai():
                print("[OK] provider ping (openai)")
            else:
                print("[FAIL] provider ping (openai)")
                ok = False
        else:
            print("[WARN] ping skipped (unsupported provider)")

    return 0 if ok else 1


def _ping_openai(timeout_s: int = 10) -> bool:
    api_key = os.getenv("OPENAI_API_KEY", "")
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
