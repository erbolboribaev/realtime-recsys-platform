"""Recommendation service: candidates -> features -> ranker -> response.

Request path:
    1. user vector (from the bundle) -> nearest items in the index, skipping everything
       the user has already seen (training-time history plus fresh events from Redis)
    2. features for those candidates from Redis counters and the live candidate list
    3. variant ``control`` keeps the ALS order, variant ``treatment`` re-orders by the ranker
    4. the response is recorded as an impression (best effort, never blocks the answer)

Unknown users (no vector yet) get the global trending list.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from recsys.events.schema import Impression
from recsys.ml.index import ItemIndex
from recsys.serving.ab import assign_variant
from recsys.serving.bundle import Bundle
from recsys.serving.features import (
    distinct_recent,
    live_feature_matrix,
    read_item_counts,
    read_user_state,
)
from recsys.serving.impressions import ImpressionSink, NullSink

log = logging.getLogger("recsys.serving")


@dataclass(frozen=True)
class Recommendation:
    request_id: str
    user_id: str
    variant: str
    source: str
    items: list[str]
    model_version: str
    candidates: int
    latency_ms: float
    timings_ms: dict[str, float] = field(default_factory=dict)


class DependencyError(RuntimeError):
    """A backing service (Redis, the vector index) could not be reached."""


class RecommendationService:
    def __init__(
        self,
        bundle: Bundle,
        index: ItemIndex,
        redis,
        scorer: Callable[[np.ndarray], np.ndarray],
        sink: ImpressionSink | None = None,
        experiment: str = "ranker-v1",
        treatment_share: float = 0.5,
        model_version: str = "als-lightgbm-v1",
        clock: Callable[[], float] = time.perf_counter,
        dependency_errors: tuple[type[Exception], ...] = (ConnectionError, TimeoutError),
    ) -> None:
        self.bundle = bundle
        self._index = index
        self._redis = redis
        self._scorer = scorer
        self._sink = sink or NullSink()
        self._experiment = experiment
        self._treatment_share = treatment_share
        self._model_version = model_version
        self._clock = clock
        self._dependency_errors = dependency_errors
        self.dropped_impressions = 0

    def _trending(self, k: int) -> list[str]:
        raw = self._redis.zrevrange("trending", 0, k - 1)
        return [i.decode() if isinstance(i, bytes) else i for i in raw]

    def recommend(self, user_id: str, k: int = 10) -> Recommendation:
        try:
            return self._recommend(user_id, k)
        except self._dependency_errors as error:
            raise DependencyError(f"{type(error).__name__}: {error}") from error

    def _recommend(self, user_id: str, k: int) -> Recommendation:
        started = self._clock()
        mark = started
        timings: dict[str, float] = {}

        def lap(name: str) -> None:
            nonlocal mark
            now = self._clock()
            timings[name] = (now - mark) * 1000.0
            mark = now

        request_id = str(uuid.uuid4())
        bundle = self.bundle

        items: list[str] = []
        variant, source, n_candidates = "cold_start", "trending", 0
        user = bundle.user_index.get(user_id)

        if user is not None:
            activity, recent_ids = read_user_state(self._redis, user_id)
            lap("user_state")
            seen = set(bundle.seen_items(user).tolist())
            seen.update(bundle.item_index[i] for i in recent_ids if i in bundle.item_index)

            found = self._index.search(bundle.user_factors[user], bundle.n_candidates, sorted(seen))
            lap("retrieval")
            if found:
                candidates = np.asarray(found)
                counts = read_item_counts(self._redis, [bundle.item_ids[i] for i in candidates])
                features, als_scores = live_feature_matrix(
                    bundle, user, candidates, counts, distinct_recent(recent_ids, bundle), activity
                )
                lap("features")
                variant = assign_variant(user_id, self._experiment, self._treatment_share)
                if variant == "treatment":
                    scores, source = self._scorer(features), "ranker"
                else:
                    scores, source = als_scores, "als"
                order = np.argsort(-scores, kind="stable")[:k]
                items = [bundle.item_ids[candidates[i]] for i in order]
                n_candidates = len(candidates)
                lap("rank")

        if not items:
            items = self._trending(k)
            lap("trending")

        self._record(request_id, user_id, variant, source, items)
        lap("impression")
        return Recommendation(
            request_id=request_id,
            user_id=user_id,
            variant=variant,
            source=source,
            items=items,
            model_version=self._model_version,
            candidates=n_candidates,
            latency_ms=(self._clock() - started) * 1000.0,
            timings_ms=timings,
        )

    def _record(self, request_id, user_id, variant, source, items) -> None:
        impression = Impression(
            request_id=request_id,
            user_id=user_id,
            variant=variant,
            source=source,
            model_version=self._model_version,
            items=items,
            timestamp=datetime.now(UTC),
        )
        try:
            self._sink.publish(impression)
        except Exception:  # an analytics outage must never break recommendations
            self.dropped_impressions += 1
            log.warning("could not record impression %s", request_id, exc_info=True)
