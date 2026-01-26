from __future__ import annotations

from typing import Optional

from .config import PricingConfig
from .types import Usage


def compute_cost_usd(
    pricing: PricingConfig,
    provider: str,
    model_name: str,
    usage: Usage,
    pricing_key: Optional[str] = None,
) -> Optional[float]:
    if usage is None:
        return None
    if usage.input_tokens is None and usage.output_tokens is None:
        return None
    provider_cfg = pricing.providers.get(provider)
    if not isinstance(provider_cfg, dict):
        return None
    key = pricing_key or model_name
    model_cfg = provider_cfg.get(key)
    if not isinstance(model_cfg, dict):
        return None
    in_price = model_cfg.get("input_per_1m_tokens")
    out_price = model_cfg.get("output_per_1m_tokens")
    if in_price is None and out_price is None:
        return None
    total = 0.0
    if in_price is not None and usage.input_tokens is not None:
        total += (usage.input_tokens / 1_000_000.0) * float(in_price)
    if out_price is not None and usage.output_tokens is not None:
        total += (usage.output_tokens / 1_000_000.0) * float(out_price)
    return total
