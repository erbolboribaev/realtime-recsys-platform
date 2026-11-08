import numpy as np
import pytest

from recsys.events.simulator import EventSimulator, SimulatorConfig
from recsys.ml.dataset import deduplicate
from recsys.ml.rankers import LightGBMRanker, LogisticRanker
from recsys.ml.ranking import (
    FEATURES,
    als_order,
    build_ranking_set,
    evaluate_ranking,
    split_in_three,
)


def small_events():
    config = SimulatorConfig(n_users=120, n_items=250, n_categories=6, seed=11)
    return deduplicate([e.model_dump() for e in EventSimulator(config).take(12_000)])


def small_set():
    a, b, _ = split_in_three(small_events())
    return build_ranking_set(a, b, n_candidates=30, factors=8, iterations=3)


def test_three_way_split_is_ordered_and_covers_everything():
    events = small_events()
    a, b, c = split_in_three(events)
    assert len(a) + len(b) + len(c) == len(events)
    assert max(e["timestamp"] for e in a) < min(e["timestamp"] for e in b)
    assert max(e["timestamp"] for e in b) < min(e["timestamp"] for e in c)


def test_three_way_split_rejects_tiny_input():
    with pytest.raises(ValueError):
        split_in_three(small_events()[:2])


def test_ranking_set_shapes_and_groups_line_up():
    rs = small_set()
    assert rs.X.shape == (len(rs.y), len(FEATURES))
    assert np.isfinite(rs.X).all()
    assert sum(rs.groups) == len(rs.y)
    assert len(rs.groups) == len(np.unique(rs.users))


def test_labels_match_the_future_and_candidates_are_unseen():
    rs = small_set()
    for user, item, label in zip(rs.users, rs.items, rs.y, strict=True):
        assert (label == 1.0) == (int(item) in rs.truth[int(user)])
        assert int(item) not in set(rs.interactions.seen_items(int(user)).tolist())


def test_ranking_set_needs_some_future_activity():
    a, _, _ = split_in_three(small_events())
    with pytest.raises(ValueError, match="nothing to rank"):
        build_ranking_set(a, [], n_candidates=10, factors=4, iterations=1)


def test_a_perfect_scorer_is_at_least_as_good_as_the_als_order():
    rs = small_set()
    perfect = evaluate_ranking(rs, rs.y, k=10)["ndcg@10"]
    baseline = evaluate_ranking(rs, als_order(rs), k=10)["ndcg@10"]
    assert perfect >= baseline


def test_logistic_ranker_learns_a_separable_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 3))
    y = (X[:, 0] + 0.1 * rng.normal(size=2000) > 0).astype(float)
    ranker = LogisticRanker().fit(X, y)
    scores = ranker.predict(X)
    assert scores[y == 1].mean() > scores[y == 0].mean()
    assert abs(ranker.weights[1]) > 5 * abs(ranker.weights[2])


def test_unfitted_logistic_ranker_fails_clearly():
    with pytest.raises(RuntimeError, match="fit"):
        LogisticRanker().predict(np.zeros((1, 2)))


def test_lightgbm_objective_is_validated():
    with pytest.raises(ValueError, match="objective"):
        LightGBMRanker("squared")


def test_lightgbm_ranker_fits_and_scores():
    pytest.importorskip("lightgbm")
    rs = small_set()
    ranker = LightGBMRanker("binary", n_estimators=30).fit(rs.X, rs.y)
    scores = ranker.predict(rs.X)
    assert scores.shape == (len(rs.y),)
    assert scores[rs.y == 1].mean() > scores[rs.y == 0].mean()


def test_lambdarank_requires_group_sizes():
    pytest.importorskip("lightgbm")
    rs = small_set()
    with pytest.raises(ValueError, match="group"):
        LightGBMRanker("lambdarank", n_estimators=5).fit(rs.X, rs.y)


def test_onnx_scores_match_lightgbm(tmp_path):
    pytest.importorskip("lightgbm")
    pytest.importorskip("onnxmltools")
    pytest.importorskip("onnxruntime")
    from recsys.ml.rankers import onnx_scores

    rs = small_set()
    ranker = LightGBMRanker("binary", n_estimators=30).fit(rs.X, rs.y)
    path = tmp_path / "ranker.onnx"
    ranker.export_onnx(path)
    assert np.abs(onnx_scores(path, rs.X) - ranker.predict(rs.X)).max() < 1e-4
