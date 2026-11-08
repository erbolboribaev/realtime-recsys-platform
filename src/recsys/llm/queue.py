"""The explanation request queue (Kafka topic) and where finished explanations are kept.

Flow: the API publishes an ``ExplainRequest`` after a recommendation; a worker turns it
into an explanation and stores it in Redis under the request id; the browser reads it over
SSE. The API never waits for the model, so recommendation latency is unaffected.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from recsys.llm.explain import Explanation


class ExplainRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(min_length=1, max_length=64)
    user_id: str = Field(min_length=1)
    items: list[str] = Field(min_length=1)

    def kafka_key(self) -> bytes:
        return self.user_id.encode()

    def kafka_value(self) -> bytes:
        return self.model_dump_json().encode()


class ExplainPublisher(Protocol):
    def publish(self, request: ExplainRequest) -> None: ...


class NullPublisher:
    """Used when Kafka is unavailable: recommendations work, explanations just never arrive."""

    def publish(self, request: ExplainRequest) -> None:
        return None


class KafkaExplainPublisher:
    def __init__(self, producer, topic: str) -> None:
        self._producer = producer
        self._topic = topic

    def publish(self, request: ExplainRequest) -> None:
        self._producer.send(self._topic, key=request.kafka_key(), value=request.kafka_value())


def result_key(request_id: str) -> str:
    return f"explain:{request_id}"


def store_result(redis, request_id: str, explanation: Explanation, ttl_s: int = 3600) -> bool:
    """Save an explanation. Returns False when one already exists (a redelivered request)."""
    payload = asdict(explanation)
    payload.pop("raw_text", None)  # model output we rejected is for evals, not for users
    return bool(redis.set(result_key(request_id), json.dumps(payload), nx=True, ex=ttl_s))


def load_result(redis, request_id: str) -> dict | None:
    raw = redis.get(result_key(request_id))
    if raw is None:
        return None
    return json.loads(raw.decode() if isinstance(raw, bytes) else raw)
