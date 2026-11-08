"""Kafka -> data lake sink.

Reads ``user-events``, validates each message against the event contract, buffers, and
writes date-partitioned Parquet to S3-compatible storage. Offsets are committed only after
a successful write, so a crash replays at most one batch (at-least-once delivery).

Run: ``python -m recsys.sink.consumer``
"""

from __future__ import annotations

import logging
import signal
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import ValidationError

from recsys.config import Settings
from recsys.events.schema import UserEvent
from recsys.sink.batching import EventBuffer

if TYPE_CHECKING:
    from recsys.sink.parquet_writer import ParquetWriter

log = logging.getLogger("recsys.sink")


def run(
    consumer,
    writer: ParquetWriter,
    buffer: EventBuffer,
    should_stop: Callable[[], bool],
    poll_timeout_ms: int = 1000,
) -> dict[str, int]:
    """Main loop. ``consumer`` only needs ``poll(timeout_ms)`` and ``commit()``."""
    stats = {"written": 0, "invalid": 0, "files": 0}

    def flush() -> None:
        if not len(buffer):
            return
        partitions = buffer.drain()
        paths = writer.write(partitions)
        consumer.commit()  # only after the files are durably written
        stats["written"] += sum(len(records) for records in partitions.values())
        stats["files"] += len(paths)
        log.info("flushed %s files, total events written=%s", len(paths), stats["written"])

    while not should_stop():
        batches = consumer.poll(timeout_ms=poll_timeout_ms)
        for messages in batches.values():
            for message in messages:
                try:
                    event = UserEvent.model_validate_json(message.value)
                except ValidationError as error:
                    # Phase 1 only counts and logs bad messages; a dead-letter topic
                    # is planned for a later phase.
                    stats["invalid"] += 1
                    problems = sorted({".".join(map(str, e["loc"])) for e in error.errors()})
                    log.warning(
                        "skipping invalid message at offset %s, bad fields: %s",
                        message.offset,
                        ", ".join(problems) or "<payload>",
                    )
                    continue
                buffer.add(event.model_dump())
        if buffer.should_flush():
            flush()

    flush()  # drain what is left on shutdown
    return stats


def ensure_bucket(
    fs,
    bucket: str,
    attempts: int = 30,
    delay_s: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Create the lake bucket if it is missing.

    Retries while the object store is still starting, so the sink does not depend on a
    separate init container or a storage-specific command-line client.
    """
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            if not fs.exists(bucket):
                fs.mkdir(bucket)
                log.info("created bucket %s", bucket)
            return
        except Exception as error:  # connection errors differ between storage backends
            last_error = error
            log.info("object store not ready (attempt %s/%s): %s", attempt, attempts, error)
            sleep(delay_s)
    raise RuntimeError(f"object store not reachable after {attempts} attempts") from last_error


def main() -> None:
    from kafka import KafkaConsumer

    from recsys.sink.parquet_writer import ParquetWriter
    from recsys.sink.storage import lake_root, s3_filesystem

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = Settings.from_env()

    fs = s3_filesystem(settings)
    ensure_bucket(fs, settings.s3_bucket)
    writer = ParquetWriter(fs, lake_root(settings))
    buffer = EventBuffer(settings.sink_batch_size, settings.sink_flush_interval_s)

    consumer = KafkaConsumer(
        settings.topic_user_events,
        bootstrap_servers=settings.kafka_bootstrap,
        group_id="lake-sink",
        enable_auto_commit=False,
        auto_offset_reset="earliest",
    )

    stop = {"requested": False}

    def request_stop(signum, _frame) -> None:
        log.info("signal %s received, finishing current batch", signum)
        stop["requested"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    log.info("consuming %s -> s3://%s", settings.topic_user_events, lake_root(settings))
    try:
        stats = run(consumer, writer, buffer, lambda: stop["requested"])
    finally:
        consumer.close()
    log.info("stopped: %s", stats)


if __name__ == "__main__":
    main()
