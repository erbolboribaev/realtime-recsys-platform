import pytest
from pydantic import ValidationError

from recsys.llm.catalog import ItemInfo
from recsys.serving.storefront import (
    MAX_ITEMS_PER_REQUEST,
    NullEventPublisher,
    build_event,
    describe_items,
)


class FakeCatalog:
    def describe(self, item_id):
        if item_id.startswith("item_"):
            return ItemInfo(item_id, f"Thing {item_id}", "Books", 12.5)
        return None


def test_describe_keeps_order_and_skips_unknown_ids():
    found = describe_items(FakeCatalog(), ["item_2", "nope", "item_1"])
    assert [x["item_id"] for x in found] == ["item_2", "item_1"]
    assert found[0] == {
        "item_id": "item_2",
        "name": "Thing item_2",
        "category": "Books",
        "price": 12.5,
    }


def test_describe_is_capped():
    ids = [f"item_{i}" for i in range(MAX_ITEMS_PER_REQUEST + 20)]
    assert len(describe_items(FakeCatalog(), ids)) == MAX_ITEMS_PER_REQUEST


def test_built_event_is_a_valid_user_event():
    event = build_event("user_00001", "item_5", "purchase", "req-1")
    assert event.user_id == "user_00001"
    assert event.event_type == "purchase"
    assert event.request_id == "req-1"
    assert event.session_id == "web-user_00001"
    assert event.timestamp.tzinfo is not None


def test_each_event_gets_its_own_id():
    assert build_event("u", "i", "view").event_id != build_event("u", "i", "view").event_id


def test_unknown_event_type_is_rejected():
    with pytest.raises(ValidationError):
        build_event("u", "i", "like")  # type: ignore[arg-type]


def test_null_publisher_accepts_events():
    assert NullEventPublisher().publish(build_event("u", "i", "view")) is None
