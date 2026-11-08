"""Publish simulated user events to Kafka.

Two modes:

* ``realtime``: events are paced at ``--rate`` per second and stamped with the current
  time, like live traffic.
* ``backfill``: events are sent as fast as possible with their simulated timestamps,
  to build a historical dataset quickly.

Run: ``python -m recsys.events.producer --mode realtime --rate 20``
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from itertools import islice

from recsys.config import Settings
from recsys.events.schema import UserEvent
from recsys.events.simulator import EventSimulator, SimulatorConfig

log = logging.getLogger("recsys.producer")

SendFn = Callable[[bytes, bytes], None]


def emit_events(
    events: Iterable[UserEvent],
    send: SendFn,
    *,
    rate: float | None = None,
    restamp_now: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Send events through ``send(key, value)``; returns how many were sent.

    ``rate`` (events/second) paces the loop; ``None`` means no pacing. Pacing is computed
    against a monotonic schedule rather than a fixed sleep per event, so the rate does not
    drift when sending itself takes time.
    """
    interval = 1.0 / rate if rate else 0.0
    next_at = clock()
    sent = 0

    for event in events:
        if restamp_now:
            event = event.model_copy(update={"timestamp": datetime.now(UTC)})
        send(event.kafka_key(), event.kafka_value())
        sent += 1

        if interval:
            next_at += interval
            delay = next_at - clock()
            if delay > 0:
                sleep(delay)
    return sent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish simulated user events to Kafka")
    parser.add_argument("--mode", choices=["realtime", "backfill"], default="realtime")
    parser.add_argument("--rate", type=float, default=20.0, help="events/s (realtime mode)")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--users", type=int, default=1000)
    parser.add_argument("--items", type=int, default=2000)
    return parser


def main(argv: list[str] | None = None) -> None:
    from kafka import KafkaProducer  # imported lazily so the module is importable without Kafka

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()

    simulator = EventSimulator(
        SimulatorConfig(seed=args.seed, n_users=args.users, n_items=args.items)
    )
    events = simulator.stream()
    if args.max_events is not None:
        events = islice(events, args.max_events)

    producer = KafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap,
        acks="all",
        retries=5,
        linger_ms=20,
    )

    def send(key: bytes, value: bytes) -> None:
        producer.send(settings.topic_user_events, key=key, value=value)

    realtime = args.mode == "realtime"
    log.info("publishing to %s (%s mode)", settings.topic_user_events, args.mode)
    try:
        sent = emit_events(
            events,
            send,
            rate=args.rate if realtime else None,
            restamp_now=realtime,
        )
    except KeyboardInterrupt:
        sent = -1
    finally:
        producer.flush()
        producer.close()
    log.info("done, sent=%s", sent if sent >= 0 else "interrupted")


if __name__ == "__main__":
    main()
