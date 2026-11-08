import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
fsspec = pytest.importorskip("fsspec")

from recsys.events.simulator import EventSimulator, SimulatorConfig  # noqa: E402
from recsys.sink.batching import EventBuffer  # noqa: E402
from recsys.sink.parquet_writer import EVENT_SCHEMA, ParquetWriter  # noqa: E402


def test_partitions_are_written_and_read_back_unchanged(tmp_path):
    events = EventSimulator(SimulatorConfig(n_users=30, n_items=60, n_categories=4)).take(300)
    buffer = EventBuffer(batch_size=1000, flush_interval_s=3600)
    for event in events:
        buffer.add(event.model_dump())

    writer = ParquetWriter(fsspec.filesystem("file"), str(tmp_path))
    paths = writer.write(buffer.drain())

    assert paths, "expected at least one file"
    assert all("/dt=2026-01-01/" in path for path in paths)  # simulator starts 2026-01-01

    table = pa.concat_tables([pq.read_table(path) for path in paths])
    assert table.schema.equals(EVENT_SCHEMA)
    assert table.num_rows == 300
    assert sorted(table["event_id"].to_pylist()) == sorted(e.event_id for e in events)


def test_replaying_a_batch_adds_a_file_instead_of_overwriting(tmp_path):
    events = EventSimulator(SimulatorConfig(n_users=30, n_items=60, n_categories=4)).take(20)
    writer = ParquetWriter(fsspec.filesystem("file"), str(tmp_path))

    def batch():
        buffer = EventBuffer(batch_size=100, flush_interval_s=3600)
        for event in events:
            buffer.add(event.model_dump())
        return buffer.drain()

    first = writer.write(batch())
    second = writer.write(batch())

    assert set(first).isdisjoint(second)
