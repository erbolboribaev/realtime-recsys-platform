from datetime import UTC, datetime

from recsys.events.producer import emit_events
from recsys.events.schema import UserEvent
from recsys.events.simulator import EventSimulator, SimulatorConfig


def some_events(n: int) -> list[UserEvent]:
    return EventSimulator(SimulatorConfig(n_users=50, n_items=100, n_categories=5)).take(n)


def test_every_event_is_sent_with_user_key_and_json_value():
    events = some_events(30)
    sent: list[tuple[bytes, bytes]] = []

    count = emit_events(events, lambda key, value: sent.append((key, value)))

    assert count == 30
    assert [key for key, _ in sent] == [event.kafka_key() for event in events]
    restored = [UserEvent.model_validate_json(value) for _, value in sent]
    assert restored == events


def test_backfill_keeps_simulated_timestamps():
    events = some_events(10)
    sent: list[bytes] = []
    emit_events(events, lambda _key, value: sent.append(value), restamp_now=False)
    restored = [UserEvent.model_validate_json(value) for value in sent]
    assert [e.timestamp for e in restored] == [e.timestamp for e in events]


def test_realtime_mode_restamps_events_with_the_current_time():
    events = some_events(5)
    sent: list[bytes] = []
    before = datetime.now(UTC)
    emit_events(events, lambda _key, value: sent.append(value), restamp_now=True)
    after = datetime.now(UTC)
    for value in sent:
        assert before <= UserEvent.model_validate_json(value).timestamp <= after


def test_pacing_sleeps_to_hold_the_requested_rate():
    clock = {"now": 0.0}
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["now"] += seconds

    emit_events(
        some_events(10),
        lambda _k, _v: None,
        rate=10.0,
        sleep=fake_sleep,
        clock=lambda: clock["now"],
    )

    # 10 events at 10/s take one simulated second in total.
    assert abs(sum(sleeps) - 1.0) < 1e-9


def test_pacing_does_not_drift_when_sending_is_slow():
    clock = {"now": 0.0}
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["now"] += seconds

    def slow_send(_k: bytes, _v: bytes) -> None:
        clock["now"] += 0.04  # sending itself eats 40 ms of the 100 ms budget

    emit_events(
        some_events(10),
        slow_send,
        rate=10.0,
        sleep=fake_sleep,
        clock=lambda: clock["now"],
    )

    assert abs(clock["now"] - 1.0) < 1e-9  # still one second overall
    assert all(abs(s - 0.06) < 1e-9 for s in sleeps)
