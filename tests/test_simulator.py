from collections import Counter, defaultdict

import numpy as np

from recsys.events.simulator import EventSimulator, SimulatorConfig


def small_config(**overrides) -> SimulatorConfig:
    values = {"n_users": 200, "n_items": 400, "n_categories": 8, "seed": 7}
    values.update(overrides)
    return SimulatorConfig(**values)


def test_same_seed_gives_identical_stream():
    first = EventSimulator(small_config()).take(500)
    second = EventSimulator(small_config()).take(500)
    assert first == second


def test_different_seed_gives_different_stream():
    first = EventSimulator(small_config(seed=1)).take(200)
    second = EventSimulator(small_config(seed=2)).take(200)
    assert [e.item_id for e in first] != [e.item_id for e in second]


def test_events_come_out_in_timestamp_order():
    events = EventSimulator(small_config()).take(3000)
    timestamps = [event.timestamp for event in events]
    assert timestamps == sorted(timestamps)


def test_event_ids_are_unique():
    events = EventSimulator(small_config()).take(3000)
    assert len({event.event_id for event in events}) == len(events)


def test_ids_stay_inside_the_configured_universe():
    cfg = small_config()
    events = EventSimulator(cfg).take(2000)
    valid_items = {f"item_{i:05d}" for i in range(cfg.n_items)}
    valid_users = {f"user_{i:05d}" for i in range(cfg.n_users)}
    assert {e.item_id for e in events} <= valid_items
    assert {e.user_id for e in events} <= valid_users


def test_funnel_shrinks_at_every_step():
    counts = Counter(event.event_type for event in EventSimulator(small_config()).take(20000))
    assert counts["view"] > counts["add_to_cart"] > counts["purchase"] > 0


def test_purchase_is_always_preceded_by_a_cart_add_of_the_same_item():
    events = EventSimulator(small_config()).take(20000)
    carted: dict[tuple[str, str], int] = defaultdict(int)
    for event in events:
        key = (event.session_id, event.item_id)
        if event.event_type == "add_to_cart":
            carted[key] += 1
        elif event.event_type == "purchase":
            assert carted[key] > 0, "purchase without a prior add_to_cart"


def test_each_session_belongs_to_a_single_user():
    owners: dict[str, str] = {}
    for event in EventSimulator(small_config()).take(5000):
        assert owners.setdefault(event.session_id, event.user_id) == event.user_id


def test_users_prefer_their_favourite_categories():
    """The personalisation signal exists: views follow each user's taste, not just popularity."""
    cfg = small_config(n_users=50)
    simulator = EventSimulator(cfg)
    events = [e for e in simulator.take(40000) if e.event_type == "view"]

    categories = simulator.catalog.categories
    item_index = {item_id: i for i, item_id in enumerate(simulator.catalog.item_ids)}

    hits = 0
    total = 0
    for event in events:
        user = int(event.user_id.split("_")[1])
        favourite = int(np.argmax(simulator.user_preferences[user]))
        total += 1
        hits += int(categories[item_index[event.item_id]] == favourite)

    # With 8 categories a taste-blind stream would hit the favourite ~12.5% of the time.
    assert hits / total > 0.30
