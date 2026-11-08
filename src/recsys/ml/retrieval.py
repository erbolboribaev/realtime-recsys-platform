"""Candidate retrieval: user vector in, nearest unseen items out."""

from __future__ import annotations

from recsys.ml.als import ALSModel
from recsys.ml.dataset import Interactions
from recsys.ml.index import ItemIndex


class Retriever:
    def __init__(self, model: ALSModel, interactions: Interactions, index: ItemIndex) -> None:
        if model.user_factors is None or model.item_factors is None:
            raise ValueError("model must be fitted")
        self._model = model
        self._interactions = interactions
        self._index = index

    def publish(self) -> None:
        """Load the model's item factors into the index."""
        self._index.build(self._model.item_factors, self._interactions.item_ids)

    def recommend(self, user: int, k: int) -> list[int]:
        vector = self._model.user_factors[user]
        seen = self._interactions.seen_items(user).tolist()
        return self._index.search(vector, k, exclude=seen)
