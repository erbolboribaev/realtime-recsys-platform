"""Ranker feature definitions, shared by training and online serving.

Training builds features from historical events; serving builds them from Redis counters
and the live candidate list. Both call these functions, so the two paths cannot drift
apart (training-serving skew). ``tests/test_feature_parity.py`` checks that they agree.

``counts`` is an ``(n_candidates, 3)`` array: views, add-to-carts, purchases per item.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

ALL_FEATURES = [
    "als_score",
    "als_rank",
    "item_pop_log",
    "item_cart_rate",
    "item_purchase_rate",
    "user_activity_log",
    "recent_similarity",
]


# The online store keeps each user's latest events; "recent items" are the newest distinct
# items among them. Training uses exactly the same window so the two stay identical.
RECENT_EVENT_WINDOW = 20
RECENT_DISTINCT_ITEMS = 5


def similarity_feature(candidate_unit: np.ndarray, recent_unit: np.ndarray) -> np.ndarray:
    """Cosine similarity of each candidate to the mean of the user's recent items."""
    if len(recent_unit) == 0:
        return np.zeros(len(candidate_unit))
    return candidate_unit @ recent_unit.mean(axis=0)


def feature_matrix(
    names: Sequence[str],
    als_scores: np.ndarray,
    counts: np.ndarray,
    similarity: np.ndarray,
    user_activity: float,
) -> np.ndarray:
    """Feature rows for one user's candidates, in candidate order.

    ``user_activity`` is the user's total number of events (not yet log-transformed).
    """
    n = len(als_scores)
    views = np.maximum(counts[:, 0], 1.0)
    columns = {
        "als_score": als_scores,
        "als_rank": np.log1p(np.arange(n)),
        "item_pop_log": np.log1p(counts.sum(axis=1)),
        "item_cart_rate": counts[:, 1] / views,
        "item_purchase_rate": counts[:, 2] / views,
        "user_activity_log": np.full(n, np.log1p(user_activity)),
        "recent_similarity": similarity,
    }
    return np.column_stack([columns[name] for name in names]).astype(np.float64)
