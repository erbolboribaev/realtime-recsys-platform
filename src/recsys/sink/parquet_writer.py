"""Write partitioned Parquet files to any fsspec filesystem (local disk, MinIO, S3)."""

from __future__ import annotations

import uuid

import pyarrow as pa
import pyarrow.parquet as pq

from recsys.sink.batching import Record

EVENT_SCHEMA = pa.schema(
    [
        ("event_id", pa.string()),
        ("user_id", pa.string()),
        ("session_id", pa.string()),
        ("item_id", pa.string()),
        ("event_type", pa.string()),
        ("timestamp", pa.timestamp("us", tz="UTC")),
        ("request_id", pa.string()),
    ]
)


class ParquetWriter:
    """Writes each date partition as ``<root>/dt=YYYY-MM-DD/part-<uuid>.parquet``.

    File names are unique, so a replayed batch after a crash adds a second file instead of
    overwriting; readers deduplicate on ``event_id`` (the sink gives at-least-once delivery).
    """

    def __init__(self, fs, root: str) -> None:
        self._fs = fs
        self._root = root.rstrip("/")

    def write(self, partitions: dict[str, list[Record]]) -> list[str]:
        paths: list[str] = []
        for date, records in sorted(partitions.items()):
            table = pa.Table.from_pylist(records, schema=EVENT_SCHEMA)
            directory = f"{self._root}/dt={date}"
            path = f"{directory}/part-{uuid.uuid4().hex}.parquet"
            self._fs.makedirs(directory, exist_ok=True)
            with self._fs.open(path, "wb") as handle:
                pq.write_table(table, handle, compression="zstd")
            paths.append(path)
        return paths
