"""Turn raw events into a training/evaluation dataset.

The lake is at-least-once, so the first step is always deduplication on ``event_id``.
Evaluation uses a *time-based* split: train on the past, test on the future, exactly the
way a deployed recommender is used. A random split would leak future behaviour into
training and overstate every metric.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

# Stronger actions are stronger evidence of interest.
EVENT_WEIGHTS: dict[str, float] = {"view": 1.0, "add_to_cart": 3.0, "purchase": 5.0}

Event = Mapping[str, Any]


def deduplicate(events: Iterable[Event]) -> list[Event]:
    """Drop repeated ``event_id`` values (keeps the first occurrence)."""
    seen: set[str] = set()
    unique: list[Event] = []
    for event in events:
        event_id = event["event_id"]
        if event_id in seen:
            continue
        seen.add(event_id)
        unique.append(event)
    return unique


@dataclass(frozen=True)
class Interactions:
    """Implicit-feedback matrix in compressed per-user / per-item form."""

    user_ids: list[str]
    item_ids: list[str]
    user_index: dict[str, int]
    item_index: dict[str, int]
    # For each user: the item indices and the strength of each interaction.
    by_user: list[tuple[np.ndarray, np.ndarray]]
    by_item: list[tuple[np.ndarray, np.ndarray]]

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def seen_items(self, user: int) -> np.ndarray:
        return self.by_user[user][0]


def build_interactions(events: Iterable[Event], item_ids: list[str] | None = None) -> Interactions:
    """Aggregate events into one strength per (user, item).

    Strength is ``log1p`` of the summed event weights, so ten views do not count ten times
    more than one purchase. Pass ``item_ids`` to keep the item axis aligned with a catalog.
    """
    totals: dict[tuple[str, str], float] = {}
    for event in events:
        key = (event["user_id"], event["item_id"])
        totals[key] = totals.get(key, 0.0) + EVENT_WEIGHTS[event["event_type"]]

    user_ids = sorted({user for user, _ in totals})
    if item_ids is None:
        item_ids = sorted({item for _, item in totals})
    user_index = {user: i for i, user in enumerate(user_ids)}
    item_index = {item: i for i, item in enumerate(item_ids)}

    per_user: list[dict[int, float]] = [{} for _ in user_ids]
    per_item: list[dict[int, float]] = [{} for _ in item_ids]
    for (user, item), total in totals.items():
        if item not in item_index:
            continue
        strength = math.log1p(total)
        per_user[user_index[user]][item_index[item]] = strength
        per_item[item_index[item]][user_index[user]] = strength

    def pack(rows: list[dict[int, float]]) -> list[tuple[np.ndarray, np.ndarray]]:
        packed = []
        for row in rows:
            indices = np.fromiter(row.keys(), dtype=np.int64, count=len(row))
            values = np.fromiter(row.values(), dtype=np.float64, count=len(row))
            packed.append((indices, values))
        return packed

    return Interactions(user_ids, item_ids, user_index, item_index, pack(per_user), pack(per_item))


@dataclass(frozen=True)
class Split:
    train: list[Event]
    test: list[Event]
    cutoff: datetime


def time_split(events: Iterable[Event], test_fraction: float = 0.2) -> Split:
    """Everything before the cutoff is train, everything after is test."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    ordered = sorted(events, key=lambda e: e["timestamp"])
    if len(ordered) < 2:
        raise ValueError("need at least two events to split")
    position = min(max(int(len(ordered) * (1.0 - test_fraction)), 1), len(ordered) - 1)
    cutoff = ordered[position]["timestamp"]
    train = [e for e in ordered if e["timestamp"] < cutoff]
    test = [e for e in ordered if e["timestamp"] >= cutoff]
    return Split(train, test, cutoff)


def future_items(test_events: Iterable[Event], interactions: Interactions) -> dict[int, set[int]]:
    """Items each *known* user touched in the test period that they had not seen in train.

    Re-recommending something a user already interacted with is not useful, so repeats are
    excluded from the ground truth (and masked at recommendation time).
    """
    truth: dict[int, set[int]] = {}
    for event in test_events:
        user = interactions.user_index.get(event["user_id"])
        item = interactions.item_index.get(event["item_id"])
        if user is None or item is None:
            continue
        if item in set(interactions.seen_items(user).tolist()):
            continue
        truth.setdefault(user, set()).add(item)
    return truth
