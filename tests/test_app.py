import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from recsys.serving.app import create_app  # noqa: E402
from test_service import make_service  # noqa: E402


def client(share=1.0):
    service, _, bundle, _ = make_service(share)
    return TestClient(create_app(service)), bundle


def test_health_reports_what_is_loaded():
    http, bundle = client()
    body = http.get("/health").json()
    assert body["status"] == "ok"
    assert body["users"] == len(bundle.user_ids)
    assert body["items"] == len(bundle.item_ids)


def test_recommend_returns_items_and_metadata():
    http, bundle = client()
    response = http.get("/recommend", params={"user_id": bundle.user_ids[0], "k": 5})
    body = response.json()
    assert response.status_code == 200
    assert len(body["items"]) == 5
    assert body["variant"] == "treatment"
    assert {"request_id", "model_version", "latency_ms", "candidates"} <= body.keys()


def test_k_defaults_to_ten():
    http, bundle = client()
    assert len(http.get("/recommend", params={"user_id": bundle.user_ids[0]}).json()["items"]) == 10


@pytest.mark.parametrize(
    "params", [{}, {"user_id": ""}, {"user_id": "u", "k": 0}, {"user_id": "u", "k": 51}]
)
def test_invalid_requests_are_rejected(params):
    http, _ = client()
    assert http.get("/recommend", params=params).status_code == 422


def test_a_failing_backing_service_returns_503():
    from recsys.serving.service import DependencyError

    service, _, bundle, _ = make_service(1.0)

    def down(user_id, k):
        raise DependencyError("redis down")

    service.recommend = down
    http = TestClient(create_app(service))
    response = http.get("/recommend", params={"user_id": bundle.user_ids[0]})
    assert response.status_code == 503
    assert "redis down" in response.json()["detail"]


def test_response_includes_stage_timings():
    http, bundle = client()
    body = http.get("/recommend", params={"user_id": bundle.user_ids[0]}).json()
    assert "rank" in body["timings_ms"]


class FakePublisher:
    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    def publish(self, request):
        if self.fail:
            raise RuntimeError("kafka down")
        self.sent.append(request)


class ResultStore:
    def __init__(self, values=None):
        self.values = values or {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, nx=False, ex=None):
        self.values[key] = value
        return True

    def pipeline(self, transaction=True):
        return self

    def hmget(self, key, fields):
        return self

    def lrange(self, key, start, stop):
        return self

    def execute(self):
        return [[None, None, None], []]


def explain_client(publisher, store=None, timeout_s=0.0):
    from recsys.serving.app import ExplainSupport

    service, _, bundle, _ = make_service(1.0)
    support = ExplainSupport(publisher, store or ResultStore(), timeout_s)
    return TestClient(create_app(service, support)), bundle


def test_explain_flag_queues_a_request_and_returns_the_stream_url():
    publisher = FakePublisher()
    http, bundle = explain_client(publisher)
    body = http.get("/recommend", params={"user_id": bundle.user_ids[0], "explain": "true"}).json()
    assert body["explain_url"] == f"/explain/{body['request_id']}"
    assert publisher.sent[0].request_id == body["request_id"]
    assert publisher.sent[0].items == body["items"]


def test_without_the_flag_nothing_is_queued():
    publisher = FakePublisher()
    http, bundle = explain_client(publisher)
    body = http.get("/recommend", params={"user_id": bundle.user_ids[0]}).json()
    assert "explain_url" not in body and publisher.sent == []


def test_a_queue_failure_does_not_break_the_recommendation():
    http, bundle = explain_client(FakePublisher(fail=True))
    response = http.get("/recommend", params={"user_id": bundle.user_ids[0], "explain": "true"})
    assert response.status_code == 200 and "explain_url" not in response.json()
    assert len(response.json()["items"]) == 10


def test_explain_stream_delivers_a_stored_explanation_as_sse():
    import json

    stored = json.dumps({"text": "Because you browse X", "source": "llm"})
    http, _ = explain_client(FakePublisher(), ResultStore({"explain:abc-1": stored}))
    response = http.get("/explain/abc-1")
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: explanation" in response.text and "Because you browse X" in response.text


def test_explain_stream_times_out_cleanly_when_no_result_arrives():
    http, _ = explain_client(FakePublisher(), timeout_s=0.0)
    assert "event: timeout" in http.get("/explain/missing-id").text


def test_explain_is_404_when_explanations_are_not_enabled():
    http, _ = client()
    assert http.get("/explain/abc").status_code == 404


def test_explain_rejects_odd_request_ids():
    http, _ = explain_client(FakePublisher())
    assert http.get("/explain/" + "a" * 65).status_code == 422


def test_explain_stream_falls_back_to_a_template_when_the_worker_is_silent():
    from recsys.events.simulator import EventSimulator, SimulatorConfig
    from recsys.llm.catalog import SimulatedCatalog
    from recsys.serving.app import ExplainSupport

    sim = EventSimulator(SimulatorConfig(n_items=200, n_categories=10, seed=4))
    service, _, bundle, _ = make_service(1.0)
    store = ResultStore()
    support = ExplainSupport(
        FakePublisher(), store, timeout_s=0.0, catalog=SimulatedCatalog(sim.catalog)
    )
    http = TestClient(create_app(service, support))
    body = http.get("/recommend", params={"user_id": bundle.user_ids[0], "explain": "true"}).json()
    stream = http.get(body["explain_url"]).text
    assert "event: explanation" in stream and '"source": "template"' in stream
    assert "event: timeout" not in stream


class FakeEvents:
    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    def publish(self, event):
        if self.fail:
            raise RuntimeError("kafka down")
        self.sent.append(event)


class StoreCatalog:
    def describe(self, item_id):
        from recsys.llm.catalog import ItemInfo

        return ItemInfo(item_id, f"Thing {item_id}", "Books", 9.0) if item_id != "missing" else None


def storefront_client(events=None, web_dir=None):
    from recsys.serving.app import Storefront

    service, _, bundle, _ = make_service(1.0)
    store = Storefront(StoreCatalog(), events or FakeEvents(), web_dir)
    return TestClient(create_app(service, storefront=store)), bundle


def test_items_returns_facts_for_known_ids():
    http, _ = storefront_client()
    body = http.get("/items", params={"ids": "a,missing,b"}).json()
    assert [x["item_id"] for x in body["items"]] == ["a", "b"]
    assert body["items"][0]["name"] == "Thing a"


def test_items_needs_ids():
    http, _ = storefront_client()
    assert http.get("/items").status_code == 422


def test_action_is_published_as_an_event():
    events = FakeEvents()
    http, _ = storefront_client(events)
    response = http.post(
        "/events",
        json={"user_id": "u1", "item_id": "i1", "event_type": "add_to_cart", "request_id": "r1"},
    )
    assert response.status_code == 202
    assert events.sent[0].event_type == "add_to_cart"
    assert events.sent[0].request_id == "r1"
    assert response.json()["event_id"] == events.sent[0].event_id


def test_unknown_action_type_is_rejected():
    http, _ = storefront_client()
    body = {"user_id": "u1", "item_id": "i1", "event_type": "like"}
    assert http.post("/events", json=body).status_code == 422


def test_event_stream_outage_returns_503():
    http, _ = storefront_client(FakeEvents(fail=True))
    body = {"user_id": "u1", "item_id": "i1", "event_type": "view"}
    assert http.post("/events", json=body).status_code == 503


def test_storefront_routes_are_absent_without_a_storefront():
    http, _ = client()
    assert http.get("/items", params={"ids": "a"}).status_code == 404


def test_web_files_are_served_and_do_not_shadow_the_api(tmp_path):
    (tmp_path / "index.html").write_text("<h1>shop</h1>", encoding="utf-8")
    http, bundle = storefront_client(web_dir=str(tmp_path))
    assert "shop" in http.get("/").text
    assert http.get("/health").json()["status"] == "ok"
    assert http.get("/recommend", params={"user_id": bundle.user_ids[0]}).status_code == 200
