"""Event contract shared by the simulator, the Kafka topics and the data lake."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

EventType = Literal["view", "add_to_cart", "purchase"]


class UserEvent(BaseModel):
    """One user action on one item. Published to the ``user-events`` topic.

    ``request_id`` links an event to the recommendation request that produced the
    impression (``None`` for organic traffic); it is what lets A/B analysis attribute
    clicks and purchases to a model version later.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    user_id: str
    session_id: str
    item_id: str
    event_type: EventType
    timestamp: datetime
    request_id: str | None = None

    @field_validator("timestamp")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(UTC)

    def kafka_key(self) -> bytes:
        """Partition by user so one user's events stay ordered."""
        return self.user_id.encode()

    def kafka_value(self) -> bytes:
        return self.model_dump_json().encode()


class Impression(BaseModel):
    """One recommendation response shown to a user. Published to the ``impressions`` topic.

    ``request_id`` is the same id later attached to the user's clicks and purchases, which
    is what lets an A/B analysis attribute outcomes to a variant.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    user_id: str
    variant: str
    source: str
    model_version: str
    items: list[str]
    timestamp: datetime

    @field_validator("timestamp")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(UTC)

    def kafka_key(self) -> bytes:
        return self.user_id.encode()

    def kafka_value(self) -> bytes:
        return self.model_dump_json().encode()
