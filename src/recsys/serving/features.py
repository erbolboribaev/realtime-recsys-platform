"""Online feature assembly: Redis counters + live candidates -> ranker input.

Counters are written by ``recsys.features.stream_job``. The arithmetic itself lives in
``recsys.ml.feature_logic`` and is shared with training.
"""

from __future__ import annotations

import numpy as np

from recsys.features.online import item_key, user_key
from recsys.ml.feature_logic import (
    RECENT_DISTINCT_ITEMS,
    RECENT_EVENT_WINDOW,
    feature_matrix,
    similarity_feature,
)
from recsys.serving.bundle import Bundle

COUNT_FIELDS = ["views", "carts", "purchases"]


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else value


def read_item_counts(redis, item_ids: list[str]) -> np.ndarray:
    """Views / carts / purchases per item, ``(len(item_ids), 3)``; one Redis round trip."""
    if not item_ids:
        return np.zeros((0, 3))
    pipe = redis.pipeline(transaction=False)
    for item in item_ids:
        pipe.hmget(item_key(item), COUNT_FIELDS)
    rows = pipe.execute()
    return np.array([[int(v or 0) for v in row] for row in rows], dtype=np.float64)


def read_user_state(redis, user_id: str) -> tuple[float, list[str]]:
    """Total events of the user and the ids of their latest events (newest first)."""
    pipe = redis.pipeline(transaction=False)
    pipe.hmget(user_key(user_id), COUNT_FIELDS)
    pipe.lrange(f"{user_key(user_id)}:recent", 0, RECENT_EVENT_WINDOW - 1)
    counts, recent = pipe.execute()
    return float(sum(int(v or 0) for v in counts)), [_text(i) for i in recent]


def distinct_recent(item_ids: list[str], bundle: Bundle) -> list[int]:
    """Newest distinct items known to the model, as item indices."""
    picked: list[int] = []
    for item in item_ids:
        index = bundle.item_index.get(item)
        if index is not None and index not in picked:
            picked.append(index)
            if len(picked) == RECENT_DISTINCT_ITEMS:
                break
    return picked


def live_feature_matrix(
    bundle: Bundle,
    user_index: int,
    candidates: np.ndarray,
    item_counts: np.ndarray,
    recent_items: list[int],
    user_activity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Ranker input for one user's candidates, plus the raw ALS scores."""
    scores = bundle.item_factors[candidates] @ bundle.user_factors[user_index]
    similarity = similarity_feature(
        bundle.unit_item_factors[candidates], bundle.unit_item_factors[recent_items]
    )
    matrix = feature_matrix(bundle.features, scores, item_counts, similarity, user_activity)
    return matrix, scores
