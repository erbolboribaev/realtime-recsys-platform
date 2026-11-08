"""Explanation worker: Kafka ``explain-requests`` -> LLM -> Redis.

Offsets are committed after a polled batch is handled (at-least-once). A redelivered request
finds its result already stored and is skipped, so the model is not called twice. One worker
handles roughly one request per model round trip (about a second); run one per topic
partition to scale.

Run: ``python -m recsys.llm.worker``   (Redis and Kafka up, LLM key set)
"""

from __future__ import annotations

import logging
import signal
import time
from collections.abc import Callable

from pydantic import ValidationError

from recsys.config import Settings
from recsys.llm.catalog import SimulatedCatalog
from recsys.llm.explain import Explainer
from recsys.llm.queue import ExplainRequest, load_result, store_result
from recsys.monitoring import beat, count
from recsys.serving.features import read_user_state

log = logging.getLogger("recsys.llm.worker")


class ExplainWorker:
    def __init__(
        self, redis, catalog: SimulatedCatalog, explainer: Explainer, ttl_s: int = 3600
    ) -> None:
        self._redis = redis
        self._catalog = catalog
        self._explainer = explainer
        self._ttl = ttl_s

    def handle(self, request: ExplainRequest) -> str:
        """Returns ``"done"`` or ``"duplicate"``."""
        if load_result(self._redis, request.request_id) is not None:
            return "duplicate"
        _, recent_ids = read_user_state(self._redis, request.user_id)
        history = [i for i in map(self._catalog.describe, recent_ids) if i]
        recommended = [i for i in map(self._catalog.describe, request.items) if i]
        explanation = self._explainer.explain(history, recommended)
        stored = store_result(self._redis, request.request_id, explanation, self._ttl)
        if stored:
            count(self._redis, "explain", explanation.source)
            if explanation.cached:
                count(self._redis, "explain", "cached")
        return "done" if stored else "duplicate"

    def heartbeat(self) -> None:
        beat(self._redis, "explain-worker")


def run(
    consumer,
    worker: ExplainWorker,
    should_stop: Callable[[], bool],
    poll_timeout_ms: int = 1000,
) -> dict[str, int]:
    stats = {"done": 0, "duplicate": 0, "invalid": 0}
    while not should_stop():
        worker.heartbeat()
        batches = consumer.poll(timeout_ms=poll_timeout_ms)
        if not batches:
            continue
        for messages in batches.values():
            for message in messages:
                try:
                    request = ExplainRequest.model_validate_json(message.value)
                except ValidationError:
                    stats["invalid"] += 1
                    log.warning("skipping invalid message at offset %s", message.offset)
                    continue
                stats[worker.handle(request)] += 1
        consumer.commit()  # only after every request in the batch has a stored result
        log.info("batch done: %s", stats)
    return stats


def run_forever(
    make_consumer: Callable[[], object],
    worker: ExplainWorker,
    should_stop: Callable[[], bool],
    sleep: Callable[[float], None] = time.sleep,
    retry_delay_s: float = 5.0,
) -> dict[str, int]:
    """Run until asked to stop, opening a fresh consumer whenever the old one breaks.

    kafka-python can die inside its socket handling (``Invalid file descriptor``) after a
    network blip or a sleeping laptop. Nothing was committed for the batch in flight, so
    reconnecting simply redelivers it, and ``handle`` skips requests that already have a result.
    """
    totals = {"done": 0, "duplicate": 0, "invalid": 0, "restarts": 0}
    while not should_stop():
        consumer = make_consumer()
        try:
            stats = run(consumer, worker, should_stop)
            for key, value in stats.items():
                totals[key] += value
        except Exception:
            totals["restarts"] += 1
            log.exception("consumer failed, reconnecting in %.0fs", retry_delay_s)
            sleep(retry_delay_s)
        finally:
            try:
                consumer.close()
            except Exception:
                log.debug("consumer.close() failed", exc_info=True)
    return totals


def main() -> None:
    import redis as redis_lib
    from kafka import KafkaConsumer

    from recsys.events.simulator import EventSimulator, SimulatorConfig
    from recsys.llm.gateway import build_gateway

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = Settings.from_env()
    store = redis_lib.Redis.from_url(settings.redis_url)
    worker = ExplainWorker(
        store,
        SimulatedCatalog(EventSimulator(SimulatorConfig()).catalog),
        Explainer(build_gateway(settings, store)),
    )

    def make_consumer():
        return KafkaConsumer(
            settings.topic_explain_requests,
            bootstrap_servers=settings.kafka_bootstrap,
            group_id="explain-worker",
            enable_auto_commit=False,
            auto_offset_reset="latest",  # an explanation for an old page view is of no use
            max_poll_records=10,
        )

    stop = {"requested": False}

    def request_stop(signum, _frame) -> None:
        log.info("signal %s received, stopping after the current batch", signum)
        stop["requested"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    log.info("consuming %s", settings.topic_explain_requests)
    stats = run_forever(make_consumer, worker, lambda: stop["requested"])
    log.info("stopped: %s", stats)


if __name__ == "__main__":
    main()
