"""Show what the lake sink has written: files, rows per date partition, event mix.

Run: ``python -m recsys.sink.lake_report``
"""

from __future__ import annotations

from collections import Counter

import pyarrow.parquet as pq

from recsys.config import Settings
from recsys.sink.storage import lake_root, s3_filesystem


def inspect_lake(fs, root: str) -> dict:
    """Summarise the Parquet files under ``root`` on any fsspec filesystem."""
    files = sorted(path for path in fs.find(root.rstrip("/")) if path.endswith(".parquet"))
    event_types: Counter[str] = Counter()
    partitions: Counter[str] = Counter()
    rows = 0

    for path in files:
        with fs.open(path, "rb") as handle:
            table = pq.read_table(handle, columns=["event_type"])
        rows += table.num_rows
        event_types.update(table["event_type"].to_pylist())
        partitions[path.split("/dt=")[1].split("/")[0]] += table.num_rows

    return {
        "files": len(files),
        "rows": rows,
        "partitions": dict(sorted(partitions.items())),
        "event_types": dict(event_types),
    }


def main() -> None:
    settings = Settings.from_env()
    root = lake_root(settings)
    summary = inspect_lake(s3_filesystem(settings), root)

    print(f"lake: s3://{root}")
    if not summary["files"]:
        print("no Parquet files yet (is the pipeline running? the sink flushes every ~10 s)")
        return
    print(f"files: {summary['files']}, rows: {summary['rows']}")
    for date, count in summary["partitions"].items():
        print(f"  dt={date}: {count} rows")
    print(f"event types: {summary['event_types']}")


if __name__ == "__main__":
    main()
