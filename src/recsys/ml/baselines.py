"""Non-personalised baseline every real model has to beat."""

from __future__ import annotations

import numpy as np

from recsys.ml.dataset import Interactions


def top_k_unseen(scores: np.ndarray, seen: np.ndarray, k: int) -> list[int]:
    """Indices of the ``k`` highest scores, skipping items the user already has."""
    masked = scores.astype(np.float64).copy()
    masked[seen] = -np.inf
    k = min(k, int(np.isfinite(masked).sum()))
    if k <= 0:
        return []
    top = np.argpartition(-masked, k - 1)[:k]
    return [int(i) for i in top[np.argsort(-masked[top], kind="stable")]]


class PopularityModel:
    """Recommends the most interacted-with items to everyone."""

    def __init__(self, interactions: Interactions) -> None:
        self._interactions = interactions
        scores = np.zeros(interactions.n_items)
        for item, (_, values) in enumerate(interactions.by_item):
            scores[item] = values.sum()
        self._scores = scores

    def recommend(self, user: int, k: int) -> list[int]:
        return top_k_unseen(self._scores, self._interactions.seen_items(user), k)
