from __future__ import annotations

import re
from typing import Iterable, Optional


def redact_text(text: str, patterns: Optional[Iterable[str]], replacement: str = "***") -> str:
    if not patterns:
        return text
    out = text
    for p in patterns:
        try:
            out = re.sub(p, replacement, out)
        except re.error:
            # ignore invalid regex to avoid crashing
            continue
    return out
