from datetime import UTC, datetime, timedelta

import pytest

from recsys.ml.dataset import (
    build_interactions,
    deduplicate,
    future_items,
    time_split,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def ev(n, user="u1", item="i1", kind="view", minutes=0):
    return {
        "event_id": f"e{n}",
        "user_id": user,
        "item_id": item,
        "event_type": kind,
        "timestamp": T0 + timedelta(minutes=minutes),
    }


def test_deduplicate_keeps_first_occurrence_of_each_event_id():
    events = [ev(1), ev(1), ev(2)]
    assert [e["event_id"] for e in deduplicate(events)] == ["e1", "e2"]


def test_stronger_events_give_stronger_interactions():
    events = [ev(1, item="a", kind="view"), ev(2, item="b", kind="purchase")]
    inter = build_interactions(events)
    items, strengths = inter.by_user[0]
    by_item = {inter.item_ids[i]: s for i, s in zip(items, strengths, strict=True)}
    assert by_item["b"] > by_item["a"]


def test_repeated_views_are_damped_not_linear():
    once = build_interactions([ev(1)]).by_user[0][1][0]
    ten = build_interactions([ev(n) for n in range(10)]).by_user[0][1][0]
    assert once < ten < 10 * once


def test_user_and_item_views_hold_the_same_pairs():
    events = [ev(1, "u1", "a"), ev(2, "u2", "a"), ev(3, "u2", "b")]
    inter = build_interactions(events)
    assert inter.n_users == 2 and inter.n_items == 2
    pairs_by_user = {(u, i) for u, (items, _) in enumerate(inter.by_user) for i in items}
    pairs_by_item = {(u, i) for i, (users, _) in enumerate(inter.by_item) for u in users}
    assert pairs_by_user == pairs_by_item


def test_time_split_never_puts_future_events_in_train():
    events = [ev(n, minutes=n) for n in range(10)]
    split = time_split(events, test_fraction=0.3)
    assert max(e["timestamp"] for e in split.train) < min(e["timestamp"] for e in split.test)
    assert len(split.train) + len(split.test) == 10


def test_time_split_rejects_bad_fraction():
    with pytest.raises(ValueError):
        time_split([ev(1), ev(2)], test_fraction=1.5)


def test_future_items_excludes_repeats_and_unknown_users():
    train = [ev(1, "u1", "a"), ev(2, "u1", "b")]
    inter = build_interactions(train, item_ids=["a", "b", "c", "d"])
    test = [
        ev(3, "u1", "a", minutes=5),  # already seen in train -> not a target
        ev(4, "u1", "c", minutes=6),
        ev(5, "ghost", "d", minutes=7),  # user never seen in train
    ]
    truth = future_items(test, inter)
    assert truth == {inter.user_index["u1"]: {inter.item_index["c"]}}
