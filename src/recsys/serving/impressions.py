"""Where recommendation responses are recorded (the ``impressions`` Kafka topic)."""

from __future__ import annotations

from typing import Protocol

from recsys.events.schema import Impression


class ImpressionSink(Protocol):
    def publish(self, impression: Impression) -> None: ...


class NullSink:
    """Used when no Kafka is available; recommendations still work."""

    def publish(self, impression: Impression) -> None:
        return None


class KafkaImpressionSink:
    """Fire-and-forget publisher. ``producer`` is a ``kafka.KafkaProducer``."""

    def __init__(self, producer, topic: str) -> None:
        self._producer = producer
        self._topic = topic

    def publish(self, impression: Impression) -> None:
        self._producer.send(self._topic, key=impression.kafka_key(), value=impression.kafka_value())
