import math

import pytest

from recsys.ml.metrics import evaluate, ndcg_at_k, recall_at_k


def test_recall_counts_share_of_relevant_items_found():
    assert recall_at_k([1, 2, 3, 4], {2, 4, 9}, k=4) == pytest.approx(2 / 3)


def test_recall_ignores_items_beyond_k():
    assert recall_at_k([1, 2, 3, 4], {4}, k=3) == 0.0


def test_ndcg_is_one_for_a_perfect_ranking_and_lower_when_hits_are_late():
    relevant = {1, 2}
    assert ndcg_at_k([1, 2, 3], relevant, k=3) == pytest.approx(1.0)
    late = ndcg_at_k([3, 1, 2], relevant, k=3)
    assert late == pytest.approx((1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3)))
    assert late < 1.0


def test_empty_relevant_set_is_rejected():
    with pytest.raises(ValueError):
        recall_at_k([1], set(), k=1)


def test_evaluate_averages_over_users():
    truth = {0: {1}, 1: {5}}
    ranked = {0: [1, 2], 1: [3, 4]}
    result = evaluate(lambda user, k: ranked[user], truth, k=2)
    assert result["recall@2"] == pytest.approx(0.5)
    assert result["users"] == 2.0
