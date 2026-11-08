import pytest

pytest.importorskip("pyarrow")
fsspec = pytest.importorskip("fsspec")

from collections import Counter  # noqa: E402

from recsys.events.simulator import EventSimulator, SimulatorConfig  # noqa: E402
from recsys.sink.batching import EventBuffer  # noqa: E402
from recsys.sink.lake_report import inspect_lake  # noqa: E402
from recsys.sink.parquet_writer import ParquetWriter  # noqa: E402


def test_report_matches_what_was_written(tmp_path):
    events = EventSimulator(SimulatorConfig(n_users=30, n_items=60, n_categories=4)).take(400)
    fs = fsspec.filesystem("file")
    writer = ParquetWriter(fs, str(tmp_path))

    for start in range(0, len(events), 100):  # four separate files
        buffer = EventBuffer(batch_size=1000, flush_interval_s=3600)
        for event in events[start : start + 100]:
            buffer.add(event.model_dump())
        writer.write(buffer.drain())

    summary = inspect_lake(fs, str(tmp_path))

    assert summary["files"] == 4
    assert summary["rows"] == 400
    assert sum(summary["partitions"].values()) == 400
    assert summary["event_types"] == dict(Counter(e.event_type for e in events))


def test_report_on_an_empty_location(tmp_path):
    summary = inspect_lake(fsspec.filesystem("file"), str(tmp_path / "nothing-here"))
    assert summary == {"files": 0, "rows": 0, "partitions": {}, "event_types": {}}
