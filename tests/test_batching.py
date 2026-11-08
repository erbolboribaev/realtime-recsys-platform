from datetime import UTC, datetime, timedelta, timezone

import pytest

from recsys.sink.batching import EventBuffer, partition_date


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def record(day: int, hour: int = 12) -> dict:
    return {"event_id": f"{day}-{hour}", "timestamp": datetime(2026, 1, day, hour, tzinfo=UTC)}


def test_partition_date_uses_utc_not_local_time():
    late_in_tashkent = datetime(2026, 1, 2, 2, 0, tzinfo=timezone(timedelta(hours=5)))
    assert partition_date(late_in_tashkent) == "2026-01-01"


def test_empty_buffer_never_asks_for_a_flush():
    clock = FakeClock()
    buffer = EventBuffer(batch_size=3, flush_interval_s=5, clock=clock)
    clock.now = 1000
    assert not buffer.should_flush()


def test_flush_when_batch_size_is_reached():
    buffer = EventBuffer(batch_size=3, flush_interval_s=60, clock=FakeClock())
    buffer.add(record(1))
    buffer.add(record(1))
    assert not buffer.should_flush()
    buffer.add(record(1))
    assert buffer.should_flush()


def test_flush_when_oldest_record_has_waited_long_enough():
    clock = FakeClock()
    buffer = EventBuffer(batch_size=100, flush_interval_s=5, clock=clock)
    buffer.add(record(1))
    clock.now = 4.9
    assert not buffer.should_flush()
    clock.now = 5.0
    assert buffer.should_flush()


def test_drain_groups_by_date_and_resets():
    clock = FakeClock()
    buffer = EventBuffer(batch_size=10, flush_interval_s=5, clock=clock)
    for day in (1, 1, 2, 3, 2):
        buffer.add(record(day))

    partitions = buffer.drain()

    assert {date: len(rows) for date, rows in partitions.items()} == {
        "2026-01-01": 2,
        "2026-01-02": 2,
        "2026-01-03": 1,
    }
    assert len(buffer) == 0
    clock.now = 1000
    assert not buffer.should_flush()


def test_wait_timer_restarts_after_a_drain():
    clock = FakeClock()
    buffer = EventBuffer(batch_size=100, flush_interval_s=5, clock=clock)
    buffer.add(record(1))
    clock.now = 10
    buffer.drain()
    buffer.add(record(1))  # first record of a new batch arrives at t=10
    clock.now = 12
    assert not buffer.should_flush()
    clock.now = 15
    assert buffer.should_flush()


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError):
        EventBuffer(batch_size=0, flush_interval_s=5)
