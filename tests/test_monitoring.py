from recsys.monitoring import RequestWindow, beat, count, heartbeat_age_s, percentile, read_counts
from recsys.serving.admin import AdminSupport, build_stats


class FakeRedis:
    def __init__(self):
        self.kv = {}
        self.hashes = {}

    def set(self, key, value, nx=False, ex=None):
        self.kv[key] = value
        return True

    def get(self, key):
        value = self.kv.get(key)
        return value.encode() if isinstance(value, str) else value

    def hincrby(self, key, field, amount):
        h = self.hashes.setdefault(key, {})
        h[field] = h.get(field, 0) + amount

    def hgetall(self, key):
        return {k.encode(): str(v).encode() for k, v in self.hashes.get(key, {}).items()}

    def ping(self):
        return True


class BrokenRedis(FakeRedis):
    def set(self, *a, **k):
        raise ConnectionError("down")

    def hincrby(self, *a, **k):
        raise ConnectionError("down")

    def ping(self):
        raise ConnectionError("down")


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def test_percentile_nearest_rank():
    assert percentile([], 95) is None
    assert percentile([5], 50) == 5
    values = list(range(1, 101))
    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 95


def test_heartbeat_age_and_missing():
    redis, clock = FakeRedis(), Clock(1000.0)
    assert heartbeat_age_s(redis, "job", clock) is None
    beat(redis, "job", clock=clock)
    clock.now = 1007.5
    assert heartbeat_age_s(redis, "job", clock) == 7.5


def test_monitoring_never_raises_when_redis_is_down():
    broken = BrokenRedis()
    beat(broken, "job")
    count(broken, "explain", "llm")


def test_counts_round_trip_and_ignore_non_positive():
    redis = FakeRedis()
    count(redis, "explain", "llm", 2)
    count(redis, "explain", "llm")
    count(redis, "explain", "template", 0)
    assert read_counts(redis, "explain") == {"llm": 3}


def test_window_counts_actions_by_the_variant_of_their_list():
    clock = Clock()
    window = RequestWindow(clock=clock)
    window.record_list("r1", "treatment", 20.0)
    window.record_list("r2", "control", 40.0)
    window.record_list("r3", "treatment", 30.0)
    window.record_action("r1")
    window.record_action("r1")
    window.record_action("unknown")
    window.record_action(None)
    snap = window.snapshot()
    assert snap["requests_in_window"] == 3
    assert snap["variants"]["treatment"] == {"lists": 2, "actions": 2, "rate": 1.0}
    assert snap["variants"]["control"]["actions"] == 0
    assert snap["latency_ms"]["p95"] == 40.0


def test_window_drops_old_requests():
    clock = Clock(1000.0)
    window = RequestWindow(window_s=300, clock=clock)
    window.record_list("old", "control", 10.0)
    clock.now = 1400.0
    window.record_list("new", "control", 12.0)
    snap = window.snapshot()
    assert snap["requests_in_window"] == 1
    assert snap["requests_last_minute"] == 1


def test_build_stats_reports_services_and_spend():
    redis, clock = FakeRedis(), Clock(1_700_000_000.0)
    beat(redis, "feature-job", clock=clock)
    clock.now += 40  # the explain worker never beat; feature job went quiet
    redis.kv["llm:cost:2023-11-14"] = 250_000
    count(redis, "explain", "llm", 4)
    count(redis, "explain", "template", 1)
    admin = AdminSupport(redis, RequestWindow(clock=clock), daily_budget_usd=1.0, clock=clock)
    stats = build_stats(admin, {"status": "ok"})
    assert stats["services"]["redis"]["alive"] is True
    assert stats["services"]["feature_job"]["alive"] is False
    assert stats["services"]["explain_worker"]["seconds_since_beat"] is None
    assert stats["llm"]["spent_usd"] == 0.25
    assert stats["llm"]["answers"] == {"ai": 4, "template": 1, "from_cache": 0}


def test_build_stats_survives_redis_outage():
    admin = AdminSupport(BrokenRedis())
    stats = build_stats(admin, {"status": "ok"})
    assert stats["services"]["redis"]["alive"] is False
    assert stats["services"]["feature_job"]["alive"] is False
    assert stats["llm"] is None
