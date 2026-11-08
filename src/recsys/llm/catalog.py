"""Human-readable item facts for prompts.

The simulated catalog only has an id, a category index and a price, so names are built from
them. A real shop would answer ``describe`` from its product database instead; everything
else in the LLM layer depends only on the ``describe`` / ``ItemInfo`` shape.
"""

from __future__ import annotations

from dataclasses import dataclass

from recsys.events.catalog import Catalog

CATEGORY_NAMES = (
    "Running shoes", "Sports watches", "Backpacks", "Headphones", "Laptops",
    "Smartphones", "Cameras", "Coffee makers", "Cookware", "Books",
    "Board games", "Yoga mats", "Bicycles", "Sunglasses", "Jackets",
    "Desk lamps", "Keyboards", "Monitors", "Water bottles", "Tents",
)  # fmt: skip


SINGULAR = (
    "Running shoe", "Sports watch", "Backpack", "Headphone", "Laptop",
    "Smartphone", "Camera", "Coffee maker", "Cookware set", "Book",
    "Board game", "Yoga mat", "Bicycle", "Pair of sunglasses", "Jacket",
    "Desk lamp", "Keyboard", "Monitor", "Water bottle", "Tent",
)  # fmt: skip
assert len(SINGULAR) == len(CATEGORY_NAMES)


@dataclass(frozen=True)
class ItemInfo:
    item_id: str
    name: str
    category: str
    price: float


class SimulatedCatalog:
    def __init__(self, catalog: Catalog) -> None:
        self._catalog = catalog
        self._index = {item: i for i, item in enumerate(catalog.item_ids)}

    def describe(self, item_id: str) -> ItemInfo | None:
        i = self._index.get(item_id)
        if i is None:
            return None
        index = int(self._catalog.categories[i]) % len(CATEGORY_NAMES)
        return ItemInfo(
            item_id=item_id,
            name=f"{SINGULAR[index]} model {i}",
            category=CATEGORY_NAMES[index],
            price=float(self._catalog.prices[i]),
        )

    @property
    def n_categories(self) -> int:
        return min(self._catalog.n_categories, len(CATEGORY_NAMES))

    def item_ids_in(self, category: int) -> list[str]:
        return [self._catalog.item_ids[i] for i in self._catalog.items_in_category(category)]
