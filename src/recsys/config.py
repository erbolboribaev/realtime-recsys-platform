"""Runtime settings, read from environment variables.

A plain frozen dataclass instead of a settings framework: no extra dependency, and
trivial to construct in tests. Each field maps to the upper-cased environment variable
of the same name (``kafka_bootstrap`` -> ``KAFKA_BOOTSTRAP``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path


def read_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """``KEY=value`` lines from a ``.env`` file; blank values and comments are skipped.

    Blank values are ignored on purpose so the template (``LLM_API_KEY=``) never overrides a
    default with an empty string. Returns ``{}`` when the file does not exist.
    """
    file = Path(path)
    if not file.is_file():
        return {}
    values: dict[str, str] = {}
    for line in file.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("\"'")
        if key.strip() and value:
            values[key.strip().removeprefix("export ").strip()] = value
    return values


@dataclass(frozen=True)
class Settings:
    kafka_bootstrap: str = "localhost:19092"
    topic_user_events: str = "user-events"
    topic_impressions: str = "impressions"
    topic_explain_requests: str = "explain-requests"

    s3_endpoint: str = "http://localhost:9000"
    s3_bucket: str = "recsys-lake"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"

    redis_url: str = "redis://localhost:6379/0"
    feature_ttl_s: int = 86400

    models_dir: str = "models"
    web_dir: str = "web"  # the storefront files the API serves at /
    ab_treatment_share: float = 0.5
    admin_token: str = ""  # when set, /admin/stats needs the X-Admin-Token header

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "items"
    mlflow_tracking_uri: str = "./mlruns"

    llm_provider: str = "groq"  # groq | openai-compatible (any server with /chat/completions)
    llm_api_key: str = ""  # falls back to GROQ_API_KEY for the groq provider
    llm_model: str = ""  # empty = the provider's default model
    llm_base_url: str = ""  # empty = the provider's default endpoint
    llm_price_input_per_m: float = 0.0  # USD per 1M tokens; overrides the built-in price
    llm_price_output_per_m: float = 0.0
    llm_reasoning_effort: str = ""  # low | medium | high; empty = "low" for gpt-oss models
    llm_timeout_s: float = 15.0
    llm_daily_budget_usd: float = 1.0
    llm_cache_ttl_s: int = 86400
    groq_api_key: str = ""

    sink_batch_size: int = 500
    sink_flush_interval_s: float = 30.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        if env is None:  # real environment variables win over the .env file
            env = {**read_dotenv(), **os.environ}
        defaults = cls()
        values: dict[str, object] = {}
        for field in fields(cls):
            raw = env.get(field.name.upper())
            if raw is not None:
                values[field.name] = type(getattr(defaults, field.name))(raw)
        return cls(**values)
