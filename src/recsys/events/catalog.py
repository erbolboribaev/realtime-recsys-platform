"""Synthetic product catalog: categories, prices and a long-tail popularity curve."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Catalog:
    item_ids: tuple[str, ...]
    categories: np.ndarray  # shape (n_items,), category index of each item
    prices: np.ndarray  # shape (n_items,)
    popularity: np.ndarray  # shape (n_items,), sums to 1
    n_categories: int

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def items_in_category(self, category: int) -> np.ndarray:
        return np.flatnonzero(self.categories == category)


def generate_catalog(n_items: int, n_categories: int, rng: np.random.Generator) -> Catalog:
    """Build a catalog where a few items are very popular and most are rarely seen.

    Popularity follows a Zipf-like curve over a random ranking, which is the shape real
    e-commerce traffic has and the reason popularity is a strong baseline for the models.
    """
    if n_items < n_categories:
        raise ValueError("need at least one item per category")

    # First n_categories items are assigned one per category so no category is empty.
    categories = np.concatenate(
        [np.arange(n_categories), rng.integers(0, n_categories, n_items - n_categories)]
    )
    rng.shuffle(categories)

    ranks = rng.permutation(n_items) + 1
    popularity = 1.0 / ranks**0.8
    popularity = popularity / popularity.sum()

    prices = np.round(rng.lognormal(mean=3.5, sigma=0.6, size=n_items), 2)

    return Catalog(
        item_ids=tuple(f"item_{i:05d}" for i in range(n_items)),
        categories=categories,
        prices=prices,
        popularity=popularity,
        n_categories=n_categories,
    )
