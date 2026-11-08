from datetime import UTC, datetime

import numpy as np
import pytest

from recsys.events.schema import Impression
from recsys.features.online import apply_event
from recsys.ml.index import NumpyItemIndex
from recsys.serving.service import RecommendationService
from test_features import FakeRedis
from test_serving_features import training_setup


class RecordingSink:
    def __init__(self):
        self.impressions: list[Impression] = []

    def publish(self, impression):
        self.impressions.append(impression)


class BrokenSink:
    def publish(self, impression):
        raise ConnectionError("kafka is down")


def popularity_scorer(features):
    """Column 2 is item_pop_log: a stand-in for the ONNX ranker."""
    return features[:, 2]


def make_service(share, sink=None, scorer=popularity_scorer):
    history, rs, bundle = training_setup()
    redis = FakeRedis()
    for event in sorted(history, key=lambda e: e["timestamp"]):
        apply_event(redis, event)
    index = NumpyItemIndex()
    index.build(bundle.item_factors, bundle.item_ids)
    service = RecommendationService(bundle, index, redis, scorer, sink=sink, treatment_share=share)
    return service, rs, bundle, redis


def test_control_returns_the_als_order():
    service, rs, bundle, _ = make_service(share=0.0)
    user = 3
    result = service.recommend(bundle.user_ids[user], k=10)
    expected = [bundle.item_ids[i] for i in rs.model.recommend(user, 10)]
    assert (result.variant, result.source) == ("control", "als")
    assert result.items == expected


def test_treatment_reorders_candidates_with_the_ranker():
    service, _, bundle, redis = make_service(share=1.0)
    result = service.recommend(bundle.user_ids[3], k=10)
    assert (result.variant, result.source) == ("treatment", "ranker")
    popularity = [
        sum(int(v or 0) for v in redis.hmget(f"item:{item}", ["views", "carts", "purchases"]))
        for item in result.items
    ]
    assert popularity == sorted(popularity, reverse=True)
    assert result.candidates == bundle.n_candidates


def test_never_recommends_items_the_user_has_seen():
    service, _, bundle, _ = make_service(share=1.0)
    for user in range(0, 60, 6):
        seen = {bundle.item_ids[i] for i in bundle.seen_items(user)}
        assert not seen & set(service.recommend(bundle.user_ids[user], k=20).items)


def test_an_item_viewed_after_training_disappears_from_the_next_response():
    service, _, bundle, redis = make_service(share=0.0)
    user_id = bundle.user_ids[5]
    top = service.recommend(user_id, k=5).items[0]
    apply_event(
        redis,
        {
            "event_id": "fresh-1",
            "user_id": user_id,
            "session_id": "s",
            "item_id": top,
            "event_type": "view",
            "timestamp": datetime(2026, 6, 1, tzinfo=UTC),
        },
    )
    assert top not in service.recommend(user_id, k=5).items


def test_unknown_users_get_trending_items():
    service, _, bundle, redis = make_service(share=1.0)
    result = service.recommend("brand-new-user", k=5)
    assert (result.variant, result.source, result.candidates) == ("cold_start", "trending", 0)
    assert result.items == redis.zrevrange("trending", 0, 4)


def test_k_is_respected_and_request_ids_are_unique():
    service, _, bundle, _ = make_service(share=0.5)
    first = service.recommend(bundle.user_ids[1], k=7)
    second = service.recommend(bundle.user_ids[1], k=7)
    assert len(first.items) == len(second.items) == 7
    assert first.request_id != second.request_id
    assert first.latency_ms >= 0


def test_every_response_is_recorded_as_an_impression():
    sink = RecordingSink()
    service, _, bundle, _ = make_service(share=1.0, sink=sink)
    result = service.recommend(bundle.user_ids[2], k=5)
    (impression,) = sink.impressions
    assert impression.request_id == result.request_id
    assert impression.items == result.items
    assert impression.variant == "treatment"
    assert impression.user_id == bundle.user_ids[2]


def test_a_broken_sink_does_not_break_recommendations():
    service, _, bundle, _ = make_service(share=0.0, sink=BrokenSink())
    result = service.recommend(bundle.user_ids[2], k=5)
    assert len(result.items) == 5
    assert service.dropped_impressions == 1


def test_scorer_receives_one_row_per_candidate_with_the_bundle_features():
    seen = {}

    def spy(features):
        seen["shape"] = features.shape
        return np.zeros(len(features))

    service, _, bundle, _ = make_service(share=1.0, scorer=spy)
    service.recommend(bundle.user_ids[4], k=5)
    assert seen["shape"] == (bundle.n_candidates, len(bundle.features))


def test_impression_requires_a_timezone():
    with pytest.raises(ValueError):
        Impression(
            request_id="r",
            user_id="u",
            variant="control",
            source="als",
            model_version="v",
            items=[],
            timestamp=datetime(2026, 1, 1),
        )


class DownRedis(FakeRedis):
    def pipeline(self, transaction=True):
        raise ConnectionError("redis is down")


def test_an_unreachable_backing_service_raises_dependency_error():
    from recsys.serving.service import DependencyError

    service, _, bundle, _ = make_service(share=0.0)
    service._redis = DownRedis()
    with pytest.raises(DependencyError, match="redis is down"):
        service.recommend(bundle.user_ids[1], k=5)


def test_programming_errors_are_not_reported_as_dependency_failures():
    def broken(features):
        raise KeyError("bug")

    service, _, bundle, _ = make_service(share=1.0, scorer=broken)
    with pytest.raises(KeyError):
        service.recommend(bundle.user_ids[1], k=5)


def test_each_stage_is_timed():
    service, _, bundle, _ = make_service(share=1.0)
    timings = service.recommend(bundle.user_ids[1], k=5).timings_ms
    assert {"user_state", "retrieval", "features", "rank", "impression"} <= timings.keys()
    assert all(value >= 0 for value in timings.values())
