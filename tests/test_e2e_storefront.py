"""End to end: a real browser, the real API app and the real recommendation service.

Only the outside world is replaced: Redis is an in-memory fake, there is no Kafka, and the
feature job is a loopback that applies each posted action to that same Redis (exactly what
the stream job does after Kafka). So a click really changes the next list.

Needs ``pip install playwright && playwright install chromium``. Without them the tests are
skipped; set ``E2E_REQUIRED=1`` (CI does) to turn a missing browser into a failure instead.
"""

import os
import socket
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
uvicorn = pytest.importorskip("uvicorn")
sync_api = pytest.importorskip("playwright.sync_api")

from recsys.features.online import apply_event  # noqa: E402
from recsys.llm.catalog import ItemInfo  # noqa: E402
from recsys.monitoring import RequestWindow, beat  # noqa: E402
from recsys.serving.admin import AdminSupport  # noqa: E402
from recsys.serving.app import Storefront, create_app  # noqa: E402
from test_monitoring import FakeRedis as AdminRedis  # noqa: E402
from test_service import make_service  # noqa: E402

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


class Catalog:
    def describe(self, item_id):
        return ItemInfo(item_id, f"Product {item_id}", "Books", 25.0)


class LoopbackEvents:
    """Stands in for Kafka plus the feature job: apply the action to the feature store."""

    def __init__(self, redis):
        self._redis = redis
        self.sent = []

    def publish(self, event):
        self.sent.append(event)
        apply_event(self._redis, event.model_dump())


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture()
def stack():
    service, _, bundle, feature_redis = make_service(1.0)
    events = LoopbackEvents(feature_redis)
    admin_redis = AdminRedis()
    admin = AdminSupport(admin_redis, RequestWindow())
    app = create_app(service, storefront=Storefront(Catalog(), events, str(WEB_DIR)), admin=admin)
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "the test server did not start"
    yield {
        "url": f"http://127.0.0.1:{port}",
        "shopper": bundle.user_ids[3],
        "events": events,
        "admin_redis": admin_redis,
    }
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture()
def page():
    with sync_api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except Exception as error:  # the browser binary is not installed
            if os.environ.get("E2E_REQUIRED"):
                raise
            pytest.skip(f"Chromium is not available: {error}")
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        context.set_default_timeout(15_000)
        yield context.new_page()
        browser.close()


def card_ids(page) -> list[str]:
    return page.eval_on_selector_all(".card", "els => els.map(e => e.dataset.item)")


def test_buying_an_item_sends_the_event_and_changes_the_list(stack, page):
    page.goto(f"{stack['url']}/?shopper={stack['shopper']}")
    page.wait_for_selector(".card")
    before = card_ids(page)
    assert len(before) == 9

    page.locator(".card").first.get_by_role("button", name="Buy").click()
    assert page.locator("#acts li").first.inner_text().startswith("Bought")

    page.wait_for_selector(".card .new", timeout=15_000)  # an item entered the list
    after = card_ids(page)
    assert before[0] not in after, "an item that was just bought should not be recommended again"
    assert len(after) == 9

    sent = stack["events"].sent
    assert len(sent) == 1
    assert (sent[0].event_type, sent[0].item_id, sent[0].user_id) == (
        "purchase",
        before[0],
        stack["shopper"],
    )
    assert sent[0].request_id, "the action must carry the id of the list it came from"


def test_dashboard_shows_the_traffic_and_a_stopped_job(stack, page):
    page.goto(f"{stack['url']}/?shopper={stack['shopper']}")
    page.wait_for_selector(".card")
    page.locator(".card").first.get_by_role("button", name="View").click()

    page.goto(f"{stack['url']}/admin.html")
    page.wait_for_selector(".svc")
    assert page.locator(".svc").count() == 4
    assert page.locator(".svc.up").count() >= 2  # the API and the store answer
    stopped = page.locator(".svc.down")
    assert "Event processor" in stopped.first.inner_text()  # nothing has reported yet

    page.wait_for_selector(".ab-row")
    assert "lists" in page.locator(".ab-row").first.inner_text()

    beat(stack["admin_redis"], "feature-job")
    page.wait_for_function(
        "() => [...document.querySelectorAll('.svc.up')]"
        ".some(e => e.innerText.includes('Event processor'))",
        timeout=10_000,
    )
