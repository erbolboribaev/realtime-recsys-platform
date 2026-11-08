"""Online features: per-user and per-item counters kept in Redis.

Every event updates a few small structures the ranker can read in microseconds:

    user:{id}          hash   views / carts / purchases, last_item, last_ts
    user:{id}:recent   list   the user's 20 most recent items, newest first
    item:{id}          hash   views / carts / purchases
    trending           zset   item -> weighted interaction score

Kafka delivers at-least-once, so a redelivered event must not be counted twice. Each
event first claims ``seen:{event_id}`` with ``SET NX``; only the claimant updates the
counters. If the process dies between the claim and the updates, that one event is lost
from the counters (they are approximate by design); the data lake keeps the exact history.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from recsys.ml.dataset import EVENT_WEIGHTS
from recsys.ml.feature_logic import RECENT_EVENT_WINDOW

RECENT_ITEMS = RECENT_EVENT_WINDOW

_FIELD = {"view": "views", "add_to_cart": "carts", "purchase": "purchases"}


def user_key(user_id: str) -> str:
    return f"user:{user_id}"


def item_key(item_id: str) -> str:
    return f"item:{item_id}"


def apply_event(redis, event: Mapping[str, Any], ttl_s: int = 86400) -> bool:
    """Update the feature store for one event. Returns False for a duplicate."""
    claimed = redis.set(f"seen:{event['event_id']}", 1, nx=True, ex=ttl_s)
    if not claimed:
        return False

    field = _FIELD[event["event_type"]]
    user, item = event["user_id"], event["item_id"]

    pipe = redis.pipeline()
    pipe.hincrby(user_key(user), field, 1)
    pipe.hset(
        user_key(user),
        mapping={"last_item": item, "last_ts": event["timestamp"].isoformat()},
    )
    pipe.lpush(f"{user_key(user)}:recent", item)
    pipe.ltrim(f"{user_key(user)}:recent", 0, RECENT_ITEMS - 1)
    pipe.hincrby(item_key(item), field, 1)
    pipe.zincrby("trending", EVENT_WEIGHTS[event["event_type"]], item)
    pipe.execute()
    return True


def read_user_features(redis, user_id: str) -> dict[str, Any]:
    """Counters and recent items for one user (what the ranker would read)."""
    raw = redis.hgetall(user_key(user_id))
    features: dict[str, Any] = {"views": 0, "carts": 0, "purchases": 0}
    for key, value in raw.items():
        key = key.decode() if isinstance(key, bytes) else key
        value = value.decode() if isinstance(value, bytes) else value
        features[key] = int(value) if key in features else value
    recent = redis.lrange(f"{user_key(user_id)}:recent", 0, RECENT_ITEMS - 1)
    features["recent_items"] = [i.decode() if isinstance(i, bytes) else i for i in recent]
    return features
