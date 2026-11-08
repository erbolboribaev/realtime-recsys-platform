from dataclasses import dataclass

from recsys.events.simulator import EventSimulator, SimulatorConfig
from recsys.sink.batching import EventBuffer
from recsys.sink.consumer import run


@dataclass
class FakeMessage:
    value: bytes
    offset: int


class FakeConsumer:
    """Serves pre-built poll results, then asks the loop to stop."""

    def __init__(self, polls: list[list[FakeMessage]]) -> None:
        self._polls = list(polls)
        self.commits = 0
        self.stopped = False

    def poll(self, timeout_ms: int):
        if not self._polls:
            self.stopped = True
            return {}
        return {"tp": self._polls.pop(0)}

    def commit(self) -> None:
        self.commits += 1


class FakeWriter:
    def __init__(self) -> None:
        self.written: list[dict[str, list[dict]]] = []

    def write(self, partitions):
        self.written.append(partitions)
        return [f"file-{len(self.written)}-{date}" for date in partitions]


def messages(n: int, start_offset: int = 0) -> list[FakeMessage]:
    events = EventSimulator(SimulatorConfig(n_users=30, n_items=60, n_categories=4)).take(n)
    return [FakeMessage(e.kafka_value(), start_offset + i) for i, e in enumerate(events)]


def test_events_are_written_and_offsets_committed_after_the_write():
    consumer = FakeConsumer([messages(5), messages(5, 5)])
    writer = FakeWriter()
    buffer = EventBuffer(batch_size=5, flush_interval_s=3600)

    stats = run(consumer, writer, buffer, lambda: consumer.stopped)

    assert stats["written"] == 10
    assert stats["invalid"] == 0
    assert consumer.commits == len(writer.written) == 2


def test_invalid_messages_are_skipped_and_counted():
    polls = [[FakeMessage(b"not json", 0), FakeMessage(b'{"user_id": 1}', 1), *messages(3, 2)]]
    consumer = FakeConsumer(polls)
    writer = FakeWriter()
    buffer = EventBuffer(batch_size=100, flush_interval_s=3600)

    stats = run(consumer, writer, buffer, lambda: consumer.stopped)

    assert stats["invalid"] == 2
    assert stats["written"] == 3


def test_remaining_events_are_flushed_on_shutdown():
    consumer = FakeConsumer([messages(4)])
    writer = FakeWriter()
    buffer = EventBuffer(batch_size=1000, flush_interval_s=3600)

    stats = run(consumer, writer, buffer, lambda: consumer.stopped)

    assert stats["written"] == 4  # batch size never reached, final flush wrote them
    assert consumer.commits == 1


def test_nothing_is_committed_when_there_is_nothing_to_write():
    consumer = FakeConsumer([])
    writer = FakeWriter()
    buffer = EventBuffer(batch_size=10, flush_interval_s=3600)

    run(consumer, writer, buffer, lambda: consumer.stopped)

    assert consumer.commits == 0
    assert writer.written == []
