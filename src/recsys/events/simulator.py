"""Deterministic traffic simulator for an online store.

Users have sparse category preferences, items follow a long-tail popularity curve, and
each session is a small funnel: view -> (maybe) add_to_cart -> (maybe) purchase. The
preference signal is what makes personalisation measurable: a model that learns it must
beat a plain popularity baseline on the generated data.

The same seed always produces the same stream, which keeps tests and offline
evaluations reproducible.
"""

from __future__ import annotations

import heapq
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import islice

import numpy as np

from recsys.events.catalog import Catalog, generate_catalog
from recsys.events.schema import UserEvent


@dataclass(frozen=True)
class SimulatorConfig:
    n_users: int = 1000
    n_items: int = 2000
    n_categories: int = 20
    seed: int = 42
    session_rate_per_s: float = 2.0  # new sessions per simulated second
    mean_session_views: float = 4.0
    mean_gap_s: float = 8.0  # mean pause between views inside a session
    p_add_to_cart: float = 0.12
    p_purchase_given_cart: float = 0.35
    preference_concentration: float = 0.3  # Dirichlet alpha; lower = more peaked tastes
    start_time: datetime = field(default_factory=lambda: datetime(2026, 1, 1, tzinfo=UTC))


class EventSimulator:
    def __init__(self, config: SimulatorConfig | None = None) -> None:
        self.config = config or SimulatorConfig()
        cfg = self.config
        self._rng = np.random.default_rng(cfg.seed)

        self.catalog: Catalog = generate_catalog(cfg.n_items, cfg.n_categories, self._rng)

        # Per-user taste over categories, and a long-tail of how active each user is.
        self.user_preferences = self._rng.dirichlet(
            np.full(cfg.n_categories, cfg.preference_concentration), size=cfg.n_users
        )
        activity = 1.0 / (self._rng.permutation(cfg.n_users) + 1) ** 0.7
        self._user_weights = activity / activity.sum()

        # Items grouped by category with their popularity renormalised inside the group.
        self._category_items: list[np.ndarray] = []
        self._category_probs: list[np.ndarray] = []
        for category in range(cfg.n_categories):
            members = self.catalog.items_in_category(category)
            weights = self.catalog.popularity[members]
            self._category_items.append(members)
            self._category_probs.append(weights / weights.sum())

    def user_id(self, index: int) -> str:
        return f"user_{index:05d}"

    def _new_id(self) -> str:
        return str(uuid.UUID(bytes=self._rng.bytes(16), version=4))

    def _session_events(self, start: datetime) -> list[UserEvent]:
        cfg = self.config
        rng = self._rng

        user_index = int(rng.choice(cfg.n_users, p=self._user_weights))
        user_id = self.user_id(user_index)
        session_id = self._new_id()
        n_views = int(rng.geometric(1.0 / cfg.mean_session_views))

        events: list[UserEvent] = []
        clock = start

        def emit(item_index: int, event_type: str) -> None:
            events.append(
                UserEvent(
                    event_id=self._new_id(),
                    user_id=user_id,
                    session_id=session_id,
                    item_id=self.catalog.item_ids[item_index],
                    event_type=event_type,  # type: ignore[arg-type]
                    timestamp=clock,
                )
            )

        for _ in range(n_views):
            clock += timedelta(seconds=float(rng.exponential(cfg.mean_gap_s)) + 0.001)
            category = int(rng.choice(cfg.n_categories, p=self.user_preferences[user_index]))
            item_index = int(
                rng.choice(self._category_items[category], p=self._category_probs[category])
            )
            emit(item_index, "view")

            if rng.random() < cfg.p_add_to_cart:
                clock += timedelta(seconds=float(rng.exponential(cfg.mean_gap_s / 4)) + 0.001)
                emit(item_index, "add_to_cart")

                if rng.random() < cfg.p_purchase_given_cart:
                    clock += timedelta(seconds=float(rng.exponential(cfg.mean_gap_s / 4)) + 0.001)
                    emit(item_index, "purchase")

        return events

    def stream(self) -> Iterator[UserEvent]:
        """Endless stream of events in non-decreasing timestamp order.

        Sessions overlap in time, so events are buffered in a heap and released only once
        no future session can produce an earlier event (future sessions start no earlier
        than the current session clock).
        """
        heap: list[tuple[datetime, int, UserEvent]] = []
        sequence = 0
        session_clock = self.config.start_time

        while True:
            session_clock += timedelta(
                seconds=float(self._rng.exponential(1.0 / self.config.session_rate_per_s))
            )
            for event in self._session_events(session_clock):
                heapq.heappush(heap, (event.timestamp, sequence, event))
                sequence += 1
            while heap and heap[0][0] <= session_clock:
                yield heapq.heappop(heap)[2]

    def take(self, n: int) -> list[UserEvent]:
        return list(islice(self.stream(), n))
