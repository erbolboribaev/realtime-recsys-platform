"""Numbers for the admin dashboard: ``GET /admin/stats``."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from recsys.monitoring import RequestWindow, heartbeat_age_s, read_counts

log = logging.getLogger("recsys.admin")

# A job counts as alive while its heartbeat is younger than this (they beat every second or so,
# the key itself expires after 30 s).
ALIVE_WITHIN_S = 15.0


@dataclass
class AdminSupport:
    redis: object
    window: RequestWindow = field(default_factory=RequestWindow)
    daily_budget_usd: float = 1.0
    token: str | None = None
    clock: Callable[[], float] = time.time


def _service(redis, name: str, clock: Callable[[], float]) -> dict:
    age = heartbeat_age_s(redis, name, clock)
    return {
        "alive": age is not None and age <= ALIVE_WITHIN_S,
        "seconds_since_beat": None if age is None else round(age, 1),
    }


def build_stats(admin: AdminSupport, api_status: dict) -> dict:
    """Never raises: when Redis is down the Redis-backed parts are reported as unavailable."""
    stats: dict = {
        "generated_at": datetime.fromtimestamp(admin.clock(), UTC).isoformat(timespec="seconds"),
        "services": {"api": {"alive": True}, "redis": {"alive": False}},
        "api": api_status,
        "traffic": admin.window.snapshot(),
        "llm": None,
        "events": None,
    }
    try:
        admin.redis.ping()
        stats["services"]["redis"] = {"alive": True}
        stats["services"]["feature_job"] = _service(admin.redis, "feature-job", admin.clock)
        stats["services"]["explain_worker"] = _service(admin.redis, "explain-worker", admin.clock)

        day = datetime.fromtimestamp(admin.clock(), UTC).strftime("%Y-%m-%d")
        spent_micro = int(admin.redis.get(f"llm:cost:{day}") or 0)
        answers = read_counts(admin.redis, "explain")
        stats["llm"] = {
            "spent_usd": round(spent_micro / 1e6, 4),
            "budget_usd": admin.daily_budget_usd,
            "answers": {
                # cached answers were written by the AI earlier, so they are shown on their own
                "ai": max(0, answers.get("llm", 0) - answers.get("cached", 0)),
                "template": answers.get("template", 0),
                "from_cache": answers.get("cached", 0),
            },
        }
        stats["events"] = read_counts(admin.redis, "feature-job")
    except Exception:
        log.warning("admin stats: Redis is unavailable", exc_info=True)
        stats["services"]["redis"] = {"alive": False}
        stats["services"].setdefault("feature_job", {"alive": False, "seconds_since_beat": None})
        stats["services"].setdefault("explain_worker", {"alive": False, "seconds_since_beat": None})
    return stats
