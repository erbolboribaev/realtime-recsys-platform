"""HTTP API.

Run locally:  ``uvicorn --factory recsys.serving.app:create_app_from_env --port 8000``
(needs ``make up`` and ``python -m recsys.ml.publish --index`` first).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi import Path as PathParam
from fastapi.responses import StreamingResponse

from recsys.config import Settings
from recsys.llm.fallback import store_request, template_for_request
from recsys.llm.queue import ExplainPublisher, ExplainRequest, load_result
from recsys.llm.sse import sse_events
from recsys.serving.service import DependencyError, RecommendationService

log = logging.getLogger("recsys.api")


@dataclass
class ExplainSupport:
    """What the API needs for explanations: a queue to publish to and Redis to read from."""

    publisher: ExplainPublisher
    redis: object
    timeout_s: float = 15.0
    catalog: object | None = None  # enables the template fallback when the worker is silent


def create_app(
    service: RecommendationService, explain_support: ExplainSupport | None = None
) -> FastAPI:
    app = FastAPI(title="Recommendation API")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "users": len(service.bundle.user_ids),
            "items": len(service.bundle.item_ids),
            "dropped_impressions": service.dropped_impressions,
        }

    @app.get("/recommend")
    def recommend(
        user_id: str = Query(min_length=1),
        k: int = Query(10, ge=1, le=50),
        want_explanation: bool = Query(False, alias="explain"),
    ) -> dict:
        try:
            result = service.recommend(user_id, k)
        except DependencyError as error:
            raise HTTPException(
                status_code=503, detail=f"dependency unavailable: {error}"
            ) from error
        body = {
            "request_id": result.request_id,
            "user_id": result.user_id,
            "variant": result.variant,
            "source": result.source,
            "model_version": result.model_version,
            "candidates": result.candidates,
            "latency_ms": round(result.latency_ms, 2),
            "timings_ms": {name: round(v, 2) for name, v in result.timings_ms.items()},
            "items": result.items,
        }
        if want_explanation and explain_support is not None and result.items:
            try:  # an explanation problem must never break the recommendation itself
                queued = ExplainRequest(
                    request_id=result.request_id, user_id=result.user_id, items=result.items
                )
                store_request(explain_support.redis, queued)
                explain_support.publisher.publish(queued)
                body["explain_url"] = f"/explain/{result.request_id}"
            except Exception:
                log.warning("could not queue an explanation", exc_info=True)
        return body

    @app.get("/explain/{request_id}")
    def explain(
        request_id: str = PathParam(max_length=64, pattern=r"^[A-Za-z0-9-]+$"),
    ) -> StreamingResponse:
        if explain_support is None:
            raise HTTPException(status_code=404, detail="explanations are not enabled")
        catalog = explain_support.catalog
        events = sse_events(
            lambda: load_result(explain_support.redis, request_id),
            explain_support.timeout_s,
            fallback=(
                (lambda: template_for_request(explain_support.redis, catalog, request_id))
                if catalog is not None
                else None
            ),
        )
        return StreamingResponse(
            events,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def build_service(settings: Settings) -> RecommendationService:
    import redis as redis_lib
    from qdrant_client import QdrantClient

    from recsys.ml.index import QdrantItemIndex
    from recsys.ml.rankers import OnnxRanker
    from recsys.serving.bundle import load_bundle
    from recsys.serving.impressions import KafkaImpressionSink, NullSink

    models = Path(settings.models_dir)
    bundle = load_bundle(models)
    index = QdrantItemIndex(
        QdrantClient(url=settings.qdrant_url, check_compatibility=False), settings.qdrant_collection
    )
    client = redis_lib.Redis.from_url(settings.redis_url)

    try:
        from kafka import KafkaProducer

        producer = KafkaProducer(bootstrap_servers=settings.kafka_bootstrap, linger_ms=20)
        sink = KafkaImpressionSink(producer, settings.topic_impressions)
    except Exception:
        log.warning("Kafka unavailable: impressions will not be recorded", exc_info=True)
        sink = NullSink()

    from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
    from redis.exceptions import RedisError

    return RecommendationService(
        bundle,
        index,
        client,
        OnnxRanker(models / "ranker.onnx"),
        sink=sink,
        treatment_share=settings.ab_treatment_share,
        dependency_errors=(RedisError, ResponseHandlingException, UnexpectedResponse),
    )


def build_explain_support(settings: Settings) -> ExplainSupport:
    import redis as redis_lib

    from recsys.llm.queue import KafkaExplainPublisher, NullPublisher

    try:
        from kafka import KafkaProducer

        publisher = KafkaExplainPublisher(
            KafkaProducer(bootstrap_servers=settings.kafka_bootstrap, linger_ms=20),
            settings.topic_explain_requests,
        )
    except Exception:
        log.warning("Kafka unavailable: explanations will not be produced", exc_info=True)
        publisher = NullPublisher()
    from recsys.events.simulator import EventSimulator, SimulatorConfig
    from recsys.llm.catalog import SimulatedCatalog

    return ExplainSupport(
        publisher,
        redis_lib.Redis.from_url(settings.redis_url),
        catalog=SimulatedCatalog(EventSimulator(SimulatorConfig()).catalog),
    )


def create_app_from_env() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = Settings.from_env()
    return create_app(build_service(settings), build_explain_support(settings))
