"""Second stage: build a labelled ranking dataset from ALS candidates.

Retrieval (ALS + the vector index) is cheap but coarse. The ranker re-orders its top
candidates using extra signals. Training data is built without leaking the future:

    history events  -> fit ALS, compute every feature
    label events    -> the *later* period; a candidate is positive if the user touched
                       that item (not seen in history) in this period

Training uses (history = period A, labels = period B). The honest test uses
(history = A + B, labels = period C): ALS and the features are rebuilt from older data,
and the ranker has never seen period C.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from recsys.ml.als import ALSModel
from recsys.ml.dataset import Interactions, build_interactions, future_items
from recsys.ml.feature_logic import (
    ALL_FEATURES,
    RECENT_DISTINCT_ITEMS,
    RECENT_EVENT_WINDOW,
    feature_matrix,
    similarity_feature,
)
from recsys.ml.metrics import ndcg_at_k, recall_at_k

Event = Mapping[str, Any]

FEATURES = ALL_FEATURES

# Features that carried real signal in the ablation (`python -m recsys.ml.rank --ablate`).
# The conversion rates and the per-user activity level added only noise: the latter is
# constant within a user, so it cannot help order one user's candidates.
CORE_FEATURES = ["als_score", "als_rank", "item_pop_log", "recent_similarity"]


@dataclass(frozen=True)
class RankingSet:
    """One row per (user, candidate item); rows of one user are contiguous."""

    X: np.ndarray
    y: np.ndarray
    users: np.ndarray  # user index per row
    items: np.ndarray  # item index per row
    truth: dict[int, set[int]]  # every future item per user (not only the candidates)
    interactions: Interactions
    model: ALSModel

    @property
    def groups(self) -> list[int]:
        """Rows per user, in row order (what LambdaRank needs)."""
        change = np.flatnonzero(np.diff(self.users)) + 1
        return np.diff(np.concatenate([[0], change, [len(self.users)]])).tolist()


def split_in_three(events: Sequence[Event], fractions: tuple[float, float] = (0.6, 0.2)):
    """Oldest ``fractions[0]`` -> A, next ``fractions[1]`` -> B, the rest -> C."""
    ordered = sorted(events, key=lambda e: e["timestamp"])
    first = int(len(ordered) * fractions[0])
    second = int(len(ordered) * (fractions[0] + fractions[1]))
    if not 0 < first < second < len(ordered):
        raise ValueError("not enough events for a three-way split")
    cut_a, cut_b = ordered[first]["timestamp"], ordered[second]["timestamp"]
    a = [e for e in ordered if e["timestamp"] < cut_a]
    b = [e for e in ordered if cut_a <= e["timestamp"] < cut_b]
    c = [e for e in ordered if e["timestamp"] >= cut_b]
    return a, b, c


def _item_counts(events: Sequence[Event], interactions: Interactions) -> np.ndarray:
    """Per item: views, add-to-carts and purchases in the history."""
    counts = np.zeros((interactions.n_items, 3))
    column = {"view": 0, "add_to_cart": 1, "purchase": 2}
    for event in events:
        item = interactions.item_index.get(event["item_id"])
        if item is not None:
            counts[item, column[event["event_type"]]] += 1
    return counts


def _user_stats(events: Sequence[Event], interactions: Interactions):
    """Event count per user, and each user's recent distinct items (newest first).

    "Recent" means the newest distinct items among the user's last
    ``RECENT_EVENT_WINDOW`` events, which is what the online store holds.
    """
    activity = np.zeros(interactions.n_users)
    in_window = np.zeros(interactions.n_users, dtype=int)
    recent: list[list[int]] = [[] for _ in range(interactions.n_users)]
    for event in sorted(events, key=lambda e: e["timestamp"], reverse=True):
        user = interactions.user_index.get(event["user_id"])
        item = interactions.item_index.get(event["item_id"])
        if user is None or item is None:
            continue
        activity[user] += 1
        if in_window[user] < RECENT_EVENT_WINDOW:
            in_window[user] += 1
            if item not in recent[user] and len(recent[user]) < RECENT_DISTINCT_ITEMS:
                recent[user].append(item)
    return activity, recent


def build_ranking_set(
    history: Sequence[Event],
    labels: Sequence[Event],
    n_candidates: int = 100,
    factors: int = 32,
    iterations: int = 10,
    seed: int = 0,
) -> RankingSet:
    interactions = build_interactions(history)
    als = ALSModel(factors=factors, iterations=iterations, seed=seed).fit(interactions)
    truth = future_items(labels, interactions)

    item_counts = _item_counts(history, interactions)
    user_activity, recent = _user_stats(history, interactions)
    norms = np.linalg.norm(als.item_factors, axis=1, keepdims=True) + 1e-12
    unit = als.item_factors / norms

    rows, labels_out, users_out, items_out = [], [], [], []
    for user in sorted(truth):
        candidates = als.recommend(user, n_candidates)
        if not candidates:
            continue
        cand = np.asarray(candidates)
        scores = als.item_factors[cand] @ als.user_factors[user]
        similarity = similarity_feature(unit[cand], unit[recent[user]])
        matrix = feature_matrix(
            FEATURES, scores, item_counts[cand], similarity, user_activity[user]
        )
        rows.extend(matrix)
        labels_out.extend(1.0 if int(item) in truth[user] else 0.0 for item in cand)
        users_out.extend([user] * len(cand))
        items_out.extend(int(item) for item in cand)

    if not rows:
        raise ValueError("no users with future activity: nothing to rank")
    return RankingSet(
        X=np.asarray(rows, dtype=np.float64),
        y=np.asarray(labels_out),
        users=np.asarray(users_out),
        items=np.asarray(items_out),
        truth=truth,
        interactions=interactions,
        model=als,
    )


def evaluate_ranking(rs: RankingSet, scores: np.ndarray, k: int = 10) -> dict[str, float]:
    """Mean Recall@k / NDCG@k when each user's candidates are ordered by ``scores``."""
    recalls, ndcgs = [], []
    for user in np.unique(rs.users):
        mask = rs.users == user
        order = np.argsort(-scores[mask], kind="stable")
        top = rs.items[mask][order][:k].tolist()
        relevant = rs.truth[int(user)]
        recalls.append(recall_at_k(top, relevant, k))
        ndcgs.append(ndcg_at_k(top, relevant, k))
    return {f"recall@{k}": float(np.mean(recalls)), f"ndcg@{k}": float(np.mean(ndcgs))}


def als_order(rs: RankingSet) -> np.ndarray:
    """Scores that reproduce the retrieval order (the baseline the ranker must beat)."""
    return rs.X[:, FEATURES.index("als_score")]
