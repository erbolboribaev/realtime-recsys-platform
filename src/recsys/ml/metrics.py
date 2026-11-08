"""Ranking metrics for top-K recommendation."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence


def recall_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Share of the relevant items that appear in the top ``k``."""
    if not relevant:
        raise ValueError("relevant must not be empty")
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Binary-relevance NDCG: hits near the top count more than hits near rank ``k``."""
    if not relevant:
        raise ValueError("relevant must not be empty")
    dcg = sum(
        1.0 / math.log2(rank + 2) for rank, item in enumerate(recommended[:k]) if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg


def evaluate(
    recommend: Callable[[int, int], Sequence[int]],
    truth: Mapping[int, set[int]],
    k: int,
) -> dict[str, float]:
    """Average Recall@k and NDCG@k over every user that has ground truth."""
    if not truth:
        raise ValueError("no users to evaluate")
    recalls, ndcgs = [], []
    for user, relevant in truth.items():
        ranked = recommend(user, k)
        recalls.append(recall_at_k(ranked, relevant, k))
        ndcgs.append(ndcg_at_k(ranked, relevant, k))
    return {
        f"recall@{k}": sum(recalls) / len(recalls),
        f"ndcg@{k}": sum(ndcgs) / len(ndcgs),
        "users": float(len(truth)),
    }
