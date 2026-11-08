"""What the storefront page needs from the backend: product facts and user actions.

The page shows product names and prices (``describe_items``) and reports what the shopper
does (``build_event`` + a publisher). Actions go to the same ``user-events`` topic as the
simulated traffic, so the feature job picks them up and the next recommendation reflects them.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Protocol

from recsys.events.schema import EventType, UserEvent

MAX_ITEMS_PER_REQUEST = 50


def describe_items(catalog, item_ids: list[str]) -> list[dict]:
    """Facts for up to 50 ids, in the order asked. Unknown ids are skipped."""
    found = []
    for item_id in item_ids[:MAX_ITEMS_PER_REQUEST]:
        info = catalog.describe(item_id)
        if info is not None:
            found.append(
                {
                    "item_id": info.item_id,
                    "name": info.name,
                    "category": info.category,
                    "price": info.price,
                }
            )
    return found


def build_event(
    user_id: str, item_id: str, event_type: EventType, request_id: str | None = None
) -> UserEvent:
    """A shopper action as a regular event. One browser tab is one session per shopper."""
    return UserEvent(
        event_id=str(uuid.uuid4()),
        user_id=user_id,
        session_id=f"web-{user_id}",
        item_id=item_id,
        event_type=event_type,
        timestamp=datetime.now(UTC),
        request_id=request_id,
    )


class EventPublisher(Protocol):
    def publish(self, event: UserEvent) -> None: ...


class NullEventPublisher:
    """Used when Kafka is unavailable: the page works, actions just do not reach the pipeline."""

    def publish(self, event: UserEvent) -> None:
        return None


class KafkaEventPublisher:
    def __init__(self, producer, topic: str) -> None:
        self._producer = producer
        self._topic = topic

    def publish(self, event: UserEvent) -> None:
        self._producer.send(self._topic, key=event.kafka_key(), value=event.kafka_value())
