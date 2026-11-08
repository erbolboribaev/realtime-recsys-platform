"""Kafka -> Redis feature job.

Reads ``user-events`` and keeps the online features current. Offsets are committed only
after a polled batch has been applied, and duplicates are ignored by ``apply_event``.

Run: ``python -m recsys.features.stream_job``
"""

from __future__ import annotations

import logging
import signal
from collections.abc import Callable

from pydantic import ValidationError

from recsys.config import Settings
from recsys.events.schema import UserEvent
from recsys.features.online import apply_event

log = logging.getLogger("recsys.features")


def run(
    consumer,
    redis,
    should_stop: Callable[[], bool],
    ttl_s: int = 86400,
    poll_timeout_ms: int = 1000,
) -> dict[str, int]:
    """Main loop. ``consumer`` only needs ``poll(timeout_ms)`` and ``commit()``."""
    stats = {"applied": 0, "duplicates": 0, "invalid": 0}
    while not should_stop():
        batches = consumer.poll(timeout_ms=poll_timeout_ms)
        if not batches:
            continue
        for messages in batches.values():
            for message in messages:
                try:
                    event = UserEvent.model_validate_json(message.value)
                except ValidationError:
                    stats["invalid"] += 1
                    log.warning("skipping invalid message at offset %s", message.offset)
                    continue
                if apply_event(redis, event.model_dump(), ttl_s):
                    stats["applied"] += 1
                else:
                    stats["duplicates"] += 1
        consumer.commit()  # only after the whole batch is in Redis
        log.info("batch done: %s", stats)
    return stats


def main() -> None:
    import redis as redis_lib
    from kafka import KafkaConsumer

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = Settings.from_env()

    client = redis_lib.Redis.from_url(settings.redis_url)
    consumer = KafkaConsumer(
        settings.topic_user_events,
        bootstrap_servers=settings.kafka_bootstrap,
        group_id="feature-job",
        enable_auto_commit=False,
        auto_offset_reset="earliest",
    )

    stop = {"requested": False}

    def request_stop(signum, _frame) -> None:
        log.info("signal %s received, stopping after the current batch", signum)
        stop["requested"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    log.info("consuming %s -> %s", settings.topic_user_events, settings.redis_url)
    try:
        stats = run(consumer, client, lambda: stop["requested"], settings.feature_ttl_s)
    finally:
        consumer.close()
    log.info("stopped: %s", stats)


if __name__ == "__main__":
    main()
