import math

import numpy as np
import pytest

from recsys.events.simulator import EventSimulator, SimulatorConfig
from recsys.experiments.ab_analysis import compare_means
from recsys.experiments.simulate_ab import (
    ClickModel,
    expected_clicks,
    expected_clicks_by_variant,
    per_round_rates,
    run_experiment,
    summarize_experiment,
)
from recsys.serving.ab import assign_variant


def test_identical_arms_show_no_effect():
    values = [0.0, 1.0] * 50
    result = compare_means(values, list(values))
    assert result["diff"] == 0.0
    assert result["p_value"] == pytest.approx(1.0)


def test_a_clear_difference_is_significant_and_the_interval_excludes_zero():
    control = [0.1, 0.2, 0.15, 0.12] * 50
    treatment = [0.3, 0.4, 0.35, 0.32] * 50
    result = compare_means(control, treatment)
    assert result["p_value"] < 1e-6
    assert result["ci_low"] > 0
    assert result["lift"] == pytest.approx(result["diff"] / result["mean_control"])


def test_a_difference_in_the_wrong_direction_has_a_negative_z():
    assert compare_means([1.0, 2.0, 3.0] * 10, [0.0, 1.0, 2.0] * 10)["z"] < 0


def test_zero_variance_arms_do_not_crash():
    result = compare_means([1.0, 1.0], [1.0, 1.0])
    assert result["z"] == 0.0 and result["p_value"] == pytest.approx(1.0)


def test_each_arm_needs_two_observations():
    with pytest.raises(ValueError):
        compare_means([1.0], [1.0, 2.0])


def world(n_users=300):
    config = SimulatorConfig(n_users=n_users, n_items=150, n_categories=5, seed=3)
    sim = EventSimulator(config)
    return sim, ClickModel(sim.user_preferences, sim.catalog)


def test_click_probability_is_higher_for_liked_items_and_for_top_positions():
    sim, model = world(50)
    user = 0
    relevance = model.relevance(user)
    best, worst = int(np.argmax(relevance)), int(np.argmin(relevance))
    assert model.click_probability(user, best, 0) > model.click_probability(user, worst, 0)
    assert model.click_probability(user, best, 0) > model.click_probability(user, best, 9)
    assert model.click_probability(user, best, 0) <= model.max_probability


def test_relevance_is_a_probability_distribution_per_user():
    _, model = world(20)
    assert model.relevance(3).sum() == pytest.approx(1.0)


def test_the_experiment_detects_a_genuinely_better_recommender():
    sim, model = world(300)
    user_ids = [sim.user_id(i) for i in range(300)]
    items = list(model.catalog.item_ids)
    rng_fake = np.random.default_rng(5)

    def call(user_id, k):
        variant = assign_variant(user_id, "exp")
        user = int(user_id.removeprefix("user_"))
        if variant == "treatment":  # knows the true tastes
            order = np.argsort(-model.relevance(user))[:k]
        else:  # random list
            order = rng_fake.choice(len(items), size=k, replace=False)
        return {
            "variant": variant,
            "items": [items[i] for i in order],
            "request_id": f"r-{user_id}",
        }

    engaged = []
    results = run_experiment(
        call, engaged.append, model, user_ids, rounds=3, k=10, rng=np.random.default_rng(1)
    )
    summary = summarize_experiment(results)
    for metric in ("clicked_request_rate", "clicks_per_request"):
        assert summary[metric]["diff"] > 0
        assert summary[metric]["p_value"] < 0.001
    assert engaged and all(e["request_id"].startswith("r-") for e in engaged)
    assert {e["event_type"] for e in engaged} <= {"view", "add_to_cart", "purchase"}


def test_the_experiment_finds_nothing_when_both_arms_are_identical():
    sim, model = world(300)
    user_ids = [sim.user_id(i) for i in range(300)]
    items = list(model.catalog.item_ids)

    def call(user_id, k):
        user = int(user_id.removeprefix("user_"))
        order = np.argsort(-model.relevance(user))[:k]
        return {
            "variant": assign_variant(user_id, "exp"),
            "items": [items[i] for i in order],
            "request_id": "r",
        }

    results = run_experiment(
        call, lambda event: None, model, user_ids, rounds=2, k=10, rng=np.random.default_rng(2)
    )
    p_value = summarize_experiment(results)["clicked_request_rate"]["p_value"]
    assert p_value > 0.01 and not math.isnan(p_value)


def test_expected_clicks_prefers_liking_items_first():
    _, model = world(30)
    order = np.argsort(-model.relevance(0))
    top = list(order[:5])
    assert expected_clicks(model, 0, top) > expected_clicks(model, 0, top[::-1])
    assert expected_clicks(model, 0, top) > expected_clicks(model, 0, list(order[-5:]))


def test_expected_clicks_by_variant_scores_each_list_without_sampling():
    sim, model = world(40)
    users = [sim.user_id(i) for i in range(40)]
    items = model.catalog.item_ids

    def call(user_id, k):
        user = users.index(user_id)
        best = np.argsort(-model.relevance(user))[:k]
        variant = "treatment" if user % 2 else "control"
        ids = [items[i] for i in (best if variant == "treatment" else best[::-1])]
        return {"variant": variant, "items": ids}

    out = expected_clicks_by_variant(call, model, users, 5)
    assert set(out) == {"control", "treatment"}
    assert np.mean(out["treatment"]) > np.mean(out["control"])


def test_per_round_rates_reports_one_value_per_round_and_variant():
    results = {
        "control": {"a": [(1, 1), (0, 0)], "b": [(0, 0), (0, 0)]},
        "treatment": {"c": [(1, 1), (1, 2)]},
        "cold_start": {"z": [(1, 1)]},
    }
    rates = per_round_rates(results)
    assert rates == {"control": [0.5, 0.0], "treatment": [1.0, 1.0]}
