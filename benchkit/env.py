from __future__ import annotations

from typing import Optional
import os

_loaded = False


def load_env(dotenv_path: Optional[str] = None) -> None:
    """Load environment variables from a .env file once."""
    global _loaded
    if _loaded:
        return
    path = dotenv_path or ".env"
    try:
        from dotenv import load_dotenv
    except Exception:
        _manual_load_env(path)
        _loaded = True
        return
    load_dotenv(dotenv_path=path)
    _loaded = True


def _manual_load_env(path: str) -> None:
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw or raw.startswith("#") or "=" not in raw:
                    continue
                key, val = raw.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if not key:
                    continue
                if key in os.environ:
                    continue
                os.environ[key] = val
    except Exception:
        # Ignore .env parsing errors to avoid blocking execution.
        return


def get_openai_api_key() -> str:
    return (os.getenv("OPENAI_API_KEY") or "").strip()


def get_gemini_api_key() -> str:
    return (os.getenv("GEMINI_API_KEY") or "").strip()


def get_anthropic_api_key() -> str:
    return (os.getenv("ANTHROPIC_API_KEY") or "").strip()


def get_lmstudio_api_key() -> str:
    return (os.getenv("LMSTUDIO_API_KEY") or "").strip()
