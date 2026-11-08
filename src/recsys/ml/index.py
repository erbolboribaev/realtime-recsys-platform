"""Item vector indexes for the retrieval stage.

Retrieval turns "score every item for this user" into "ask an index for the nearest items".
ALS scores are dot products, so both indexes rank by dot product and return exactly the
same order as the model itself; that equality is what the tests and the training script
verify. ``NumpyItemIndex`` is the exact reference. ``QdrantItemIndex`` is the production
index and, with an approximate-search setup, would be compared against the reference.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from recsys.ml.baselines import top_k_unseen


class ItemIndex(Protocol):
    def build(self, vectors: np.ndarray, item_ids: Sequence[str]) -> None: ...

    def search(self, vector: np.ndarray, k: int, exclude: Sequence[int] = ()) -> list[int]: ...


class NumpyItemIndex:
    """Exact brute-force dot-product search. Reference implementation and test double."""

    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None

    def build(self, vectors: np.ndarray, item_ids: Sequence[str]) -> None:
        self._vectors = np.asarray(vectors, dtype=np.float64)

    def search(self, vector: np.ndarray, k: int, exclude: Sequence[int] = ()) -> list[int]:
        if self._vectors is None:
            raise RuntimeError("index is empty: call build() first")
        scores = self._vectors @ np.asarray(vector, dtype=np.float64)
        return top_k_unseen(scores, np.asarray(exclude, dtype=np.int64), k)


class QdrantItemIndex:
    """Qdrant collection holding one point per item (point id = item index).

    ``client`` is any ``qdrant_client.QdrantClient``; pass ``QdrantClient(":memory:")`` in
    tests and ``QdrantClient(url=...)`` against the compose stack.
    """

    def __init__(self, client, collection: str = "items") -> None:
        self._client = client
        self._collection = collection

    def build(self, vectors: np.ndarray, item_ids: Sequence[str]) -> None:
        from qdrant_client import models

        if len(vectors) != len(item_ids):
            raise ValueError("vectors and item_ids must have the same length")

        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        self._client.create_collection(
            self._collection,
            vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.DOT),
        )
        batch = 500
        for start in range(0, len(vectors), batch):
            stop = min(start + batch, len(vectors))
            self._client.upsert(
                self._collection,
                points=[
                    models.PointStruct(
                        id=i,
                        vector=vectors[i].astype(float).tolist(),
                        payload={"item_id": item_ids[i]},
                    )
                    for i in range(start, stop)
                ],
            )

    def search(self, vector: np.ndarray, k: int, exclude: Sequence[int] = ()) -> list[int]:
        from qdrant_client import models

        query_filter = (
            models.Filter(must_not=[models.HasIdCondition(has_id=[int(i) for i in exclude])])
            if len(exclude)
            else None
        )
        response = self._client.query_points(
            self._collection,
            query=np.asarray(vector, dtype=float).tolist(),
            limit=k,
            query_filter=query_filter,
        )
        return [int(point.id) for point in response.points]
