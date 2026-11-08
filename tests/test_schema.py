import json
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from recsys.events.schema import UserEvent


def make_event(**overrides) -> UserEvent:
    fields = {
        "event_id": "e1",
        "user_id": "user_00001",
        "session_id": "s1",
        "item_id": "item_00042",
        "event_type": "view",
        "timestamp": datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
    }
    fields.update(overrides)
    return UserEvent(**fields)


def test_valid_event_round_trips_through_json():
    event = make_event(request_id="req-1")
    restored = UserEvent.model_validate_json(event.kafka_value())
    assert restored == event


def test_naive_timestamp_is_rejected():
    with pytest.raises(ValidationError):
        make_event(timestamp=datetime(2026, 1, 1, 12, 0))


def test_timestamp_is_normalised_to_utc():
    plus_five = timezone(timedelta(hours=5))
    event = make_event(timestamp=datetime(2026, 1, 1, 17, 0, tzinfo=plus_five))
    assert event.timestamp == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    assert event.timestamp.utcoffset() == timedelta(0)


def test_unknown_event_type_is_rejected():
    with pytest.raises(ValidationError):
        make_event(event_type="refund")


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        UserEvent.model_validate({**json.loads(make_event().kafka_value()), "extra": 1})


def test_kafka_key_is_the_user_id():
    assert make_event(user_id="user_00007").kafka_key() == b"user_00007"
