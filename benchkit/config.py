from __future__ import annotations

from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError


class ResponseFormatConfig(BaseModel):
    type: str
    name: Optional[str] = None
    strict: Optional[bool] = None
    schema: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


class RequestConfig(BaseModel):
    type: str = "openai_responses"
    system: Optional[str] = None
    user_template: str
    response_format: Optional[ResponseFormatConfig] = None
    params: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class DatasetConfig(BaseModel):
    path: str

    class Config:
        extra = "allow"


class EvaluatorConfig(BaseModel):
    type: str

    class Config:
        extra = "allow"


class EvaluationConfig(BaseModel):
    mode: str = "binary"
    evaluator: EvaluatorConfig

    class Config:
        extra = "allow"


class RedactConfig(BaseModel):
    enabled: bool = False
    patterns: List[str] = Field(default_factory=list)
    replacement: str = "***"

    class Config:
        extra = "allow"


class SuiteConfig(BaseModel):
    schema_version: int = 1
    id: str
    version: int = 1
    description: Optional[str] = None
    provider: str = "openai"
    request: RequestConfig
    dataset: DatasetConfig
    evaluation: EvaluationConfig
    output: Dict[str, Any] = Field(default_factory=dict)
    redact: Optional[RedactConfig] = None

    class Config:
        extra = "allow"


class ModelSpec(BaseModel):
    provider: str
    name: str
    label: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    pricing_key: Optional[str] = None

    class Config:
        extra = "allow"


class ModelsConfig(BaseModel):
    schema_version: int = 1
    models: List[ModelSpec]

    class Config:
        extra = "allow"


class PricingConfig(BaseModel):
    schema_version: int = 1
    version: Optional[str] = None
    currency: str = "USD"
    providers: Dict[str, Any]

    class Config:
        extra = "allow"


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping")
    return data


def load_suite(path: str) -> SuiteConfig:
    data = load_yaml(path)
    return SuiteConfig(**data)


def load_models(path: str) -> ModelsConfig:
    data = load_yaml(path)
    return ModelsConfig(**data)


def load_pricing(path: str) -> PricingConfig:
    data = load_yaml(path)
    return PricingConfig(**data)


__all__ = [
    "SuiteConfig",
    "ModelsConfig",
    "ModelSpec",
    "PricingConfig",
    "load_suite",
    "load_models",
    "load_pricing",
    "ValidationError",
]
