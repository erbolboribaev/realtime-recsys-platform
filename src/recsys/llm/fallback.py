"""A template explanation for when the worker has not answered in time.

At request time the API remembers what was recommended (``store_request``). If the SSE wait
runs out, ``template_for_request`` rebuilds a plain explanation from that memory and the user's
recent history, so the shopper still gets text even when the worker or the model is down.
"""

from __future__ import annotations

from pydantic import ValidationError

from recsys.llm.catalog import SimulatedCatalog
from recsys.llm.explain import template_explanation
from recsys.llm.queue import ExplainRequest
from recsys.serving.features import read_user_state


def request_key(request_id: str) -> str:
    return f"explain:req:{request_id}"


def store_request(redis, request: ExplainRequest, ttl_s: int = 300) -> None:
    redis.set(request_key(request.request_id), request.model_dump_json(), ex=ttl_s)


def template_for_request(redis, catalog: SimulatedCatalog, request_id: str) -> dict | None:
    raw = redis.get(request_key(request_id))
    if raw is None:
        return None
    try:
        request = ExplainRequest.model_validate_json(raw)
    except ValidationError:
        return None
    _, recent_ids = read_user_state(redis, request.user_id)
    history = [i for i in map(catalog.describe, recent_ids) if i]
    recommended = [i for i in map(catalog.describe, request.items) if i]
    return {
        "text": template_explanation(history, recommended[:3]),
        "source": "template",
        "reason": "the explanation was not ready in time",
        "cost_micro_usd": 0,
        "cached": False,
    }
