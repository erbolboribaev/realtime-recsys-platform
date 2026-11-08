"""Small operational signals for the admin dashboard.

Background jobs report to Redis (a heartbeat that expires, plus a few counters). The API keeps
a sliding window of its own recent requests in memory. Nothing here may break the work it
observes, so every Redis call is best effort.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from threading import Lock

log = logging.getLogger("recsys.monitoring")

HEARTBEAT_TTL_S = 30


def heartbeat_key(name: str) -> str:
    return f"heartbeat:{name}"


def beat(redis, name: str, ttl_s: int = HEARTBEAT_TTL_S, clock: Callable[[], float] = time.time):
    """Say "I am alive". The key expires, so a dead job turns red on its own."""
    try:
        redis.set(heartbeat_key(name), f"{clock():.3f}", ex=ttl_s)
    except Exception:
        log.debug("heartbeat for %s not written", name, exc_info=True)


def count(redis, group: str, field: str, amount: int = 1) -> None:
    if amount <= 0:
        return
    try:
        redis.hincrby(f"stats:{group}", field, amount)
    except Exception:
        log.debug("counter %s/%s not written", group, field, exc_info=True)


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def heartbeat_age_s(redis, name: str, clock: Callable[[], float] = time.time) -> float | None:
    """Seconds since the last heartbeat, or None when there is none (never started or expired)."""
    raw = redis.get(heartbeat_key(name))
    if raw is None:
        return None
    return max(0.0, clock() - float(_text(raw)))


def read_counts(redis, group: str) -> dict[str, int]:
    raw = redis.hgetall(f"stats:{group}") or {}
    return {_text(k): int(_text(v)) for k, v in raw.items()}


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile; ``q`` in [0, 100]."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * q // 100))  # ceil without importing math
    return ordered[int(rank) - 1]


class RequestWindow:
    """The API's recent traffic: lists served and shopper actions, per experiment variant.

    It lives in one process. With several API replicas each would report only its own share,
    which is fine for this demo and noted in the README.
    """

    def __init__(
        self,
        window_s: float = 300.0,
        max_items: int = 5000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._window_s = window_s
        self._clock = clock
        self._lists: deque[tuple[float, float, str]] = deque(maxlen=max_items)
        self._actions: deque[tuple[float, str]] = deque(maxlen=max_items)
        self._variant_of: OrderedDict[str, str] = OrderedDict()
        self._max_ids = max_items
        self._lock = Lock()

    def record_list(self, request_id: str, variant: str, latency_ms: float) -> None:
        with self._lock:
            self._lists.append((self._clock(), latency_ms, variant))
            self._variant_of[request_id] = variant
            while len(self._variant_of) > self._max_ids:
                self._variant_of.popitem(last=False)

    def record_action(self, request_id: str | None) -> None:
        """Counts an action for the variant that produced the list it came from."""
        if not request_id:
            return
        with self._lock:
            variant = self._variant_of.get(request_id)
            if variant is not None:
                self._actions.append((self._clock(), variant))

    def snapshot(self) -> dict:
        now = self._clock()
        cutoff = now - self._window_s
        with self._lock:
            lists = [x for x in self._lists if x[0] >= cutoff]
            actions = [x for x in self._actions if x[0] >= cutoff]
        latencies = [x[1] for x in lists]
        last_minute = sum(1 for x in lists if x[0] >= now - 60)
        variants: dict[str, dict[str, float | int]] = {}
        for _, _, variant in lists:
            variants.setdefault(variant, {"lists": 0, "actions": 0})["lists"] += 1
        for _, variant in actions:
            variants.setdefault(variant, {"lists": 0, "actions": 0})["actions"] += 1
        for entry in variants.values():
            entry["rate"] = round(entry["actions"] / entry["lists"], 4) if entry["lists"] else 0.0
        return {
            "window_s": self._window_s,
            "requests_last_minute": last_minute,
            "requests_in_window": len(lists),
            "latency_ms": {
                "p50": _round(percentile(latencies, 50)),
                "p95": _round(percentile(latencies, 95)),
            },
            "variants": variants,
        }


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 1)
