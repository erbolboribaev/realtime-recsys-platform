import numpy as np
import pytest

from recsys.events.simulator import EventSimulator, SimulatorConfig
from recsys.features.online import apply_event
from recsys.ml.dataset import deduplicate
from recsys.ml.ranking import FEATURES, build_ranking_set, split_in_three
from recsys.serving.bundle import Bundle, load_bundle, save_bundle
from recsys.serving.features import (
    distinct_recent,
    live_feature_matrix,
    read_item_counts,
    read_user_state,
)
from test_features import FakeRedis


def training_setup():
    config = SimulatorConfig(n_users=100, n_items=200, n_categories=6, seed=21)
    events = deduplicate([e.model_dump() for e in EventSimulator(config).take(10_000)])
    a, b, _ = split_in_three(events)
    rs = build_ranking_set(a, b, n_candidates=25, factors=8, iterations=3)
    bundle = Bundle.from_training(rs.model, rs.interactions, list(FEATURES), 25)
    return a, rs, bundle


def test_online_features_equal_training_features():
    """The skew test: Redis-derived features must match what training computed."""
    history, rs, bundle = training_setup()
    redis = FakeRedis()
    for event in sorted(history, key=lambda e: e["timestamp"]):
        apply_event(redis, event)

    checked = 0
    for user in np.unique(rs.users)[:15]:
        rows = rs.users == user
        candidates = rs.items[rows]
        counts = read_item_counts(redis, [bundle.item_ids[i] for i in candidates])
        activity, recent_ids = read_user_state(redis, bundle.user_ids[user])
        recent = distinct_recent(recent_ids, bundle)
        live, _ = live_feature_matrix(bundle, int(user), candidates, counts, recent, activity)
        assert np.allclose(live, rs.X[rows]), f"features differ for user {user}"
        checked += 1
    assert checked == 15


def test_bundle_round_trips_through_disk(tmp_path):
    _, _, bundle = training_setup()
    save_bundle(tmp_path / "bundle", bundle)
    loaded = load_bundle(tmp_path / "bundle")
    assert loaded.user_ids == bundle.user_ids
    assert loaded.features == bundle.features
    assert np.allclose(loaded.item_factors, bundle.item_factors)
    assert all(
        np.array_equal(loaded.seen_items(u), bundle.seen_items(u)) for u in range(0, 100, 10)
    )


def test_bundle_records_each_users_seen_items():
    _, rs, bundle = training_setup()
    for user in range(0, rs.interactions.n_users, 10):
        assert set(bundle.seen_items(user).tolist()) == set(
            rs.interactions.seen_items(user).tolist()
        )


def test_bundle_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        Bundle(
            user_factors=np.zeros((2, 3)),
            item_factors=np.zeros((2, 3)),
            user_ids=["a"],
            item_ids=["x", "y"],
            features=[],
            n_candidates=1,
            seen_indptr=np.zeros(2, dtype=np.int64),
            seen_indices=np.zeros(0, dtype=np.int64),
        )


def test_distinct_recent_skips_unknown_and_repeated_items_and_stops_at_five():
    _, _, bundle = training_setup()
    known = bundle.item_ids[:8]
    ids = [known[0], "never-seen", known[0], *known[1:]]
    picked = distinct_recent(ids, bundle)
    assert picked == [bundle.item_index[i] for i in known[:5]]


def test_unknown_user_state_is_empty():
    activity, recent = read_user_state(FakeRedis(), "nobody")
    assert activity == 0.0 and recent == []


def test_publish_writes_a_loadable_bundle_and_an_onnx_ranker(tmp_path):
    pytest.importorskip("lightgbm")
    pytest.importorskip("onnxmltools")
    pytest.importorskip("onnxruntime")
    from recsys.ml.publish import publish
    from recsys.ml.rankers import onnx_scores

    config = SimulatorConfig(n_users=100, n_items=200, n_categories=6, seed=21)
    events = [e.model_dump() for e in EventSimulator(config).take(10_000)]
    summary = publish(events, tmp_path / "models", n_candidates=25, factors=8, iterations=3)

    bundle = load_bundle(tmp_path / "models")
    assert bundle.features == ["als_score", "als_rank", "item_pop_log", "recent_similarity"]
    assert (tmp_path / "models" / "ranker.onnx").exists()
    scores = onnx_scores(tmp_path / "models" / "ranker.onnx", np.zeros((3, len(bundle.features))))
    assert scores.shape == (3,)
    assert summary["users"] == len(bundle.user_ids)
