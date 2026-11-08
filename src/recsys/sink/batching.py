"""Buffering and date partitioning for the data-lake sink (no I/O, fully testable)."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

Record = dict[str, Any]


def partition_date(timestamp: datetime) -> str:
    """UTC calendar date used as the ``dt=`` partition of the lake."""
    return timestamp.astimezone(UTC).strftime("%Y-%m-%d")


class EventBuffer:
    """Collects records and tells the caller when it is time to write a file.

    A flush is due when the buffer holds ``batch_size`` records, or when the oldest
    buffered record has waited ``flush_interval_s`` (so a quiet topic still gets written).
    """

    def __init__(
        self,
        batch_size: int,
        flush_interval_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._batch_size = batch_size
        self._flush_interval_s = flush_interval_s
        self._clock = clock
        self._records: list[Record] = []
        self._first_added_at: float | None = None

    def __len__(self) -> int:
        return len(self._records)

    def add(self, record: Record) -> None:
        if not self._records:
            self._first_added_at = self._clock()
        self._records.append(record)

    def should_flush(self) -> bool:
        if not self._records:
            return False
        if len(self._records) >= self._batch_size:
            return True
        assert self._first_added_at is not None
        return self._clock() - self._first_added_at >= self._flush_interval_s

    def drain(self) -> dict[str, list[Record]]:
        """Return buffered records grouped by date partition and reset the buffer."""
        partitions: dict[str, list[Record]] = defaultdict(list)
        for record in self._records:
            partitions[partition_date(record["timestamp"])].append(record)
        self._records = []
        self._first_added_at = None
        return dict(partitions)
