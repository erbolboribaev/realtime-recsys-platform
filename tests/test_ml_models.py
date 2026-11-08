import numpy as np

from recsys.events.simulator import EventSimulator, SimulatorConfig
from recsys.ml.als import ALSModel
from recsys.ml.baselines import PopularityModel, top_k_unseen
from recsys.ml.dataset import build_interactions, deduplicate, future_items, time_split
from recsys.ml.metrics import evaluate


def small_dataset():
    config = SimulatorConfig(n_users=150, n_items=300, n_categories=6, seed=3)
    events = [e.model_dump() for e in EventSimulator(config).take(12_000)]
    split = time_split(deduplicate(events), test_fraction=0.2)
    inter = build_interactions(split.train)
    return inter, future_items(split.test, inter)


def test_top_k_unseen_skips_seen_items_and_orders_by_score():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    assert top_k_unseen(scores, np.array([1]), k=2) == [3, 2]


def test_top_k_unseen_handles_fewer_candidates_than_k():
    assert top_k_unseen(np.array([1.0, 2.0]), np.array([0]), k=5) == [1]


def test_als_loss_goes_down_with_more_iterations():
    inter, _ = small_dataset()
    short = ALSModel(factors=8, iterations=1).fit(inter).loss()
    long = ALSModel(factors=8, iterations=8).fit(inter).loss()
    assert long < short


def test_als_never_recommends_items_the_user_already_has():
    inter, _ = small_dataset()
    model = ALSModel(factors=8, iterations=3).fit(inter)
    for user in range(0, inter.n_users, 25):
        seen = set(inter.seen_items(user).tolist())
        assert not seen & set(model.recommend(user, 20))


def test_als_is_deterministic_for_a_fixed_seed():
    inter, _ = small_dataset()
    first = ALSModel(factors=8, iterations=2, seed=1).fit(inter)
    second = ALSModel(factors=8, iterations=2, seed=1).fit(inter)
    assert np.allclose(first.item_factors, second.item_factors)


def test_als_beats_popularity_on_data_with_real_tastes():
    inter, truth = small_dataset()
    popularity = evaluate(PopularityModel(inter).recommend, truth, k=20)
    als = evaluate(ALSModel(factors=16, iterations=8).fit(inter).recommend, truth, k=20)
    assert als["recall@20"] > popularity["recall@20"]


def test_similar_items_excludes_the_query_item():
    inter, _ = small_dataset()
    model = ALSModel(factors=8, iterations=3).fit(inter)
    assert 5 not in model.similar_items(5, k=10)
