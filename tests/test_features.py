from datetime import UTC, datetime
from types import SimpleNamespace

from recsys.events.schema import UserEvent
from recsys.features.online import RECENT_ITEMS, apply_event, read_user_features
from recsys.features.stream_job import run

T0 = datetime(2026, 1, 1, tzinfo=UTC)


class FakePipeline:
    def __init__(self, redis):
        self._redis = redis
        self._calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self._calls.append((name, args, kwargs))
            return self

        return record

    def execute(self):
        results = [
            getattr(self._redis, name)(*args, **kwargs) for name, args, kwargs in self._calls
        ]
        self._calls = []
        return results


class FakeRedis:
    """Just enough of redis-py for the feature code."""

    def __init__(self):
        self.kv, self.hashes, self.lists, self.zsets = {}, {}, {}, {}

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.kv:
            return None
        self.kv[key] = value
        return True

    def pipeline(self, transaction=True):
        return FakePipeline(self)

    def hincrby(self, key, field, amount):
        h = self.hashes.setdefault(key, {})
        h[field] = str(int(h.get(field, 0)) + amount)

    def hset(self, key, mapping):
        self.hashes.setdefault(key, {}).update({k: str(v) for k, v in mapping.items()})

    def hmget(self, key, fields):
        h = self.hashes.get(key, {})
        return [h.get(field) for field in fields]

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def lpush(self, key, value):
        self.lists.setdefault(key, []).insert(0, value)

    def ltrim(self, key, start, stop):
        self.lists[key] = self.lists.get(key, [])[start : stop + 1]

    def lrange(self, key, start, stop):
        return self.lists.get(key, [])[start : stop + 1]

    def zrevrange(self, key, start, stop):
        z = self.zsets.get(key, {})
        return sorted(z, key=lambda member: -z[member])[start : stop + 1]

    def zincrby(self, key, amount, member):
        z = self.zsets.setdefault(key, {})
        z[member] = z.get(member, 0) + amount


def ev(n, user="u1", item="i1", kind="view"):
    return {
        "event_id": f"e{n}",
        "user_id": user,
        "session_id": "s1",
        "item_id": item,
        "event_type": kind,
        "timestamp": T0,
    }


def test_counters_follow_event_types():
    redis = FakeRedis()
    for n, kind in enumerate(["view", "view", "add_to_cart", "purchase"]):
        assert apply_event(redis, ev(n, kind=kind))
    features = read_user_features(redis, "u1")
    assert (features["views"], features["carts"], features["purchases"]) == (2, 1, 1)
    assert features["last_item"] == "i1"


def test_a_redelivered_event_is_counted_once():
    redis = FakeRedis()
    assert apply_event(redis, ev(1)) is True
    assert apply_event(redis, ev(1)) is False
    assert read_user_features(redis, "u1")["views"] == 1
    assert redis.zsets["trending"]["i1"] == 1.0


def test_recent_items_are_newest_first_and_capped():
    redis = FakeRedis()
    for n in range(RECENT_ITEMS + 5):
        apply_event(redis, ev(n, item=f"i{n}"))
    recent = read_user_features(redis, "u1")["recent_items"]
    assert len(recent) == RECENT_ITEMS
    assert recent[0] == f"i{RECENT_ITEMS + 4}"


def test_trending_weights_stronger_events_more():
    redis = FakeRedis()
    apply_event(redis, ev(1, item="a", kind="view"))
    apply_event(redis, ev(2, item="b", kind="purchase"))
    assert redis.zsets["trending"]["b"] > redis.zsets["trending"]["a"]


def test_unknown_user_has_zero_counters():
    features = read_user_features(FakeRedis(), "nobody")
    assert features == {"views": 0, "carts": 0, "purchases": 0, "recent_items": []}


class FakeConsumer:
    def __init__(self, batches):
        self._batches = list(batches)
        self.commits = 0
        self.applied_at_commit = []

    def poll(self, timeout_ms):
        return self._batches.pop(0) if self._batches else {}

    def commit(self):
        self.commits += 1


def message(event: dict, offset: int):
    payload = UserEvent(**event).model_dump_json().encode()
    return SimpleNamespace(value=payload, offset=offset)


def test_job_applies_events_commits_after_each_batch_and_skips_bad_messages():
    redis = FakeRedis()
    consumer = FakeConsumer(
        [
            {"p": [message(ev(1), 0), message(ev(2), 1), SimpleNamespace(value=b"{}", offset=2)]},
            {"p": [message(ev(2), 3), message(ev(3, user="u2"), 4)]},  # e2 is a redelivery
        ]
    )
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] > 2

    stats = run(consumer, redis, should_stop)
    assert stats == {"applied": 3, "duplicates": 1, "invalid": 1}
    assert consumer.commits == 2
    assert read_user_features(redis, "u1")["views"] == 2
    assert read_user_features(redis, "u2")["views"] == 1
