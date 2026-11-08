import pytest

from recsys.serving.ab import assign_variant


def test_the_same_user_always_gets_the_same_variant():
    assert assign_variant("user_1", "exp") == assign_variant("user_1", "exp")


def test_split_is_close_to_the_requested_share():
    variants = [assign_variant(f"user_{i}", "exp", 0.3) for i in range(5000)]
    share = variants.count("treatment") / len(variants)
    assert 0.27 < share < 0.33


def test_extreme_shares_are_all_or_nothing():
    users = [f"user_{i}" for i in range(200)]
    assert {assign_variant(u, "exp", 0.0) for u in users} == {"control"}
    assert {assign_variant(u, "exp", 1.0) for u in users} == {"treatment"}


def test_a_new_experiment_name_reshuffles_users():
    users = [f"user_{i}" for i in range(500)]
    first = [assign_variant(u, "exp-a") for u in users]
    second = [assign_variant(u, "exp-b") for u in users]
    assert first != second


def test_share_must_be_a_probability():
    with pytest.raises(ValueError):
        assign_variant("u", "exp", 1.5)
