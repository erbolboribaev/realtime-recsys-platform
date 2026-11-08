import json
from types import SimpleNamespace

import pytest

from recsys.events.simulator import EventSimulator, SimulatorConfig
from recsys.llm.catalog import SimulatedCatalog
from recsys.llm.explain import Explainer, Explanation
from recsys.llm.fallback import store_request, template_for_request
from recsys.llm.gateway import Completion, LLMError
from recsys.llm.queue import ExplainRequest, load_result, result_key, store_result
from recsys.llm.sse import format_event, sse_events
from recsys.llm.worker import ExplainWorker, run, run_forever
from test_features import FakeRedis as BaseRedis


class FakeRedis(BaseRedis):
    def get(self, key):
        return self.kv.get(key)


SIM = EventSimulator(SimulatorConfig(n_items=200, n_categories=10, seed=4))
CATALOG = SimulatedCatalog(SIM.catalog)
ITEMS = list(SIM.catalog.item_ids[:3])


def request(request_id="r1", user="u1", items=None):
    return ExplainRequest(request_id=request_id, user_id=user, items=items or ITEMS)


class Model:
    def __init__(self, text=None, error=None):
        self.text, self.error, self.calls = text, error, 0

    def complete(self, messages, max_tokens=150, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return Completion(self.text or "ok", "m", 1, 1, 5, False, 1.0)


def worker(model, redis=None):
    redis = redis or FakeRedis()
    return ExplainWorker(redis, CATALOG, Explainer(model)), redis


# ---- queue and result store


def test_request_roundtrips_through_kafka_bytes_and_rejects_bad_input():
    original = request()
    assert ExplainRequest.model_validate_json(original.kafka_value()) == original
    assert original.kafka_key() == b"u1"
    with pytest.raises(ValueError):
        ExplainRequest(request_id="r", user_id="u", items=[])


def test_results_are_stored_once_and_leave_out_the_rejected_model_text():
    redis = FakeRedis()
    first = Explanation("hello", "llm", raw_text="secret draft")
    assert store_result(redis, "r1", first) is True
    assert store_result(redis, "r1", Explanation("other", "llm")) is False
    stored = load_result(redis, "r1")
    assert stored["text"] == "hello" and "raw_text" not in stored
    assert load_result(redis, "missing") is None
    assert result_key("r1") == "explain:r1"


# ---- worker


def test_worker_explains_a_request_and_stores_the_result():
    model = Model("Since you browse these, look at the suggestions.")
    w, redis = worker(model)
    assert w.handle(request()) == "done"
    assert load_result(redis, "r1")["source"] == "llm" and model.calls == 1


def test_worker_uses_the_users_recent_history_from_redis():
    redis = FakeRedis()
    redis.lists["user:u1:recent"] = [ITEMS[0]]
    seen = {}

    class Spy(Model):
        def complete(self, messages, **kwargs):
            seen["user"] = messages[1]["content"]
            return super().complete(messages, **kwargs)

    w, _ = worker(Spy("fine"), redis)
    w.handle(request())
    assert "Recently browsed categories:" in seen["user"] and "nothing yet" not in seen["user"]


def test_a_redelivered_request_is_not_sent_to_the_model_again():
    model = Model("fine")
    w, _ = worker(model)
    assert w.handle(request()) == "done"
    assert w.handle(request()) == "duplicate"
    assert model.calls == 1


def test_when_the_model_is_down_the_user_still_gets_a_template_explanation():
    w, redis = worker(Model(error=LLMError("down")))
    assert w.handle(request()) == "done"
    stored = load_result(redis, "r1")
    assert stored["source"] == "template" and stored["reason"].startswith("llm unavailable")


def message(req, offset):
    value = req if isinstance(req, bytes) else req.kafka_value()
    return SimpleNamespace(value=value, offset=offset)


class FakeConsumer:
    def __init__(self, batches, fail=None):
        self.batches, self.commits, self.fail, self.closed = list(batches), 0, fail, False

    def close(self):
        self.closed = True

    def poll(self, timeout_ms):
        if self.fail:
            raise self.fail
        return self.batches.pop(0) if self.batches else {}

    def commit(self):
        self.commits += 1


def test_run_commits_after_each_batch_and_skips_invalid_messages():
    w, redis = worker(Model("fine"))
    consumer = FakeConsumer(
        [
            {"p": [message(request("a"), 0), message(b"{}", 1), message(request("b"), 2)]},
            {"p": [message(request("a"), 3)]},
        ]
    )
    state = {"polls": 0}

    def stop():
        state["polls"] += 1
        return state["polls"] > 3

    stats = run(consumer, w, stop)
    assert stats == {"done": 2, "duplicate": 1, "invalid": 1}
    assert consumer.commits == 2
    assert load_result(redis, "a") and load_result(redis, "b")


# ---- SSE


def parse(stream):
    return "".join(stream)


def test_format_event_is_valid_sse():
    text = format_event("explanation", {"text": "hi"})
    assert text == 'event: explanation\ndata: {"text": "hi"}\n\n'
    assert json.loads(text.split("data: ")[1]) == {"text": "hi"}


def test_sse_waits_with_heartbeats_then_delivers_the_explanation():
    answers = iter([None, None, {"text": "ready", "source": "llm"}])
    naps = []
    out = parse(sse_events(lambda: next(answers), sleep=naps.append, interval_s=0.5))
    assert out.startswith("retry: 3000")
    assert out.count("event: pending") == 1 and out.count(": waiting") == 1
    assert out.endswith('event: explanation\ndata: {"text": "ready", "source": "llm"}\n\n')
    assert naps == [0.5, 0.5]


def test_sse_ends_with_a_timeout_event_when_nothing_arrives():
    now = {"t": 0.0}

    def sleep(seconds):
        now["t"] += seconds

    out = parse(
        sse_events(lambda: None, timeout_s=2.0, interval_s=1.0, sleep=sleep, clock=lambda: now["t"])
    )
    assert out.endswith("}\n\n") and "event: timeout" in out and "event: explanation" not in out


def test_sse_returns_immediately_when_the_result_already_exists():
    out = parse(sse_events(lambda: {"text": "done"}, sleep=lambda s: pytest.fail("slept")))
    assert "event: explanation" in out and "pending" not in out


def test_run_forever_reconnects_after_the_kafka_client_crashes():
    w, redis = worker(Model("fine"))
    broken = FakeConsumer([], fail=ValueError("Invalid file descriptor: -1"))
    healthy = FakeConsumer([{"p": [message(request("after-crash"), 0)]}])
    consumers = iter([broken, healthy])
    naps, polls = [], {"n": 0}

    def stop():
        polls["n"] += 1
        return healthy.commits >= 1  # stop once the healthy consumer finished a batch

    totals = run_forever(lambda: next(consumers), w, stop, sleep=naps.append, retry_delay_s=2.0)
    assert totals["restarts"] == 1 and totals["done"] == 1
    assert naps == [2.0] and broken.closed and healthy.closed
    assert load_result(redis, "after-crash")


def test_sse_uses_the_fallback_explanation_instead_of_a_timeout():
    now = {"t": 0.0}

    def sleep(seconds):
        now["t"] += seconds

    out = "".join(
        sse_events(
            lambda: None,
            timeout_s=1.0,
            interval_s=1.0,
            sleep=sleep,
            clock=lambda: now["t"],
            fallback=lambda: {"text": "template", "source": "template"},
        )
    )
    assert "event: explanation" in out and "event: timeout" not in out


def test_sse_still_times_out_when_the_fallback_has_nothing():
    out = "".join(
        sse_events(lambda: None, timeout_s=0.0, sleep=lambda s: None, fallback=lambda: None)
    )
    assert "event: timeout" in out


def test_template_for_request_uses_the_remembered_items_and_recent_history():
    redis = FakeRedis()
    redis.lists["user:u1:recent"] = [ITEMS[0]]
    store_request(redis, request("r9"))
    result = template_for_request(redis, CATALOG, "r9")
    assert result["source"] == "template" and result["cost_micro_usd"] == 0
    assert "might like" in result["text"]
    assert template_for_request(redis, CATALOG, "unknown") is None
