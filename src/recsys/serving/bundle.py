"""Everything the API needs from a training run, saved to one folder.

factors.npz   user and item factors from the final ALS fit
meta.json     user ids, item ids, ranker feature names, candidate count
ranker.onnx   the exported ranker (written by ``recsys.ml.publish``)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Bundle:
    user_factors: np.ndarray
    item_factors: np.ndarray
    user_ids: list[str]
    item_ids: list[str]
    features: list[str]
    n_candidates: int
    # Items each user had interacted with at training time (CSR layout), so the API can
    # skip them without storing every user's full history in Redis.
    seen_indptr: np.ndarray
    seen_indices: np.ndarray
    user_index: dict[str, int] = field(init=False)
    item_index: dict[str, int] = field(init=False)
    unit_item_factors: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if len(self.user_ids) != len(self.user_factors):
            raise ValueError("user_ids and user_factors disagree in length")
        if len(self.item_ids) != len(self.item_factors):
            raise ValueError("item_ids and item_factors disagree in length")
        if len(self.seen_indptr) != len(self.user_ids) + 1:
            raise ValueError("seen_indptr must have one entry per user plus one")
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}
        self.item_index = {i: n for n, i in enumerate(self.item_ids)}
        norms = np.linalg.norm(self.item_factors, axis=1, keepdims=True) + 1e-12
        self.unit_item_factors = self.item_factors / norms

    def seen_items(self, user_index: int) -> np.ndarray:
        return self.seen_indices[self.seen_indptr[user_index] : self.seen_indptr[user_index + 1]]

    @classmethod
    def from_training(cls, model, interactions, features: list[str], n_candidates: int) -> Bundle:
        """Build a bundle from a fitted ALS model and the interactions it was trained on."""
        indptr = [0]
        indices: list[np.ndarray] = []
        for items, _ in interactions.by_user:
            indices.append(items.astype(np.int64))
            indptr.append(indptr[-1] + len(items))
        return cls(
            user_factors=model.user_factors,
            item_factors=model.item_factors,
            user_ids=list(interactions.user_ids),
            item_ids=list(interactions.item_ids),
            features=list(features),
            n_candidates=n_candidates,
            seen_indptr=np.asarray(indptr, dtype=np.int64),
            seen_indices=np.concatenate(indices) if indices else np.zeros(0, dtype=np.int64),
        )


def save_bundle(directory: str | Path, bundle: Bundle) -> None:
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        folder / "factors.npz",
        user_factors=bundle.user_factors,
        item_factors=bundle.item_factors,
        seen_indptr=bundle.seen_indptr,
        seen_indices=bundle.seen_indices,
    )
    meta = {
        "user_ids": bundle.user_ids,
        "item_ids": bundle.item_ids,
        "features": bundle.features,
        "n_candidates": bundle.n_candidates,
    }
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def load_bundle(directory: str | Path) -> Bundle:
    folder = Path(directory)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    with np.load(folder / "factors.npz") as arrays:
        return Bundle(
            user_factors=arrays["user_factors"],
            item_factors=arrays["item_factors"],
            user_ids=meta["user_ids"],
            item_ids=meta["item_ids"],
            features=meta["features"],
            n_candidates=meta["n_candidates"],
            seen_indptr=arrays["seen_indptr"],
            seen_indices=arrays["seen_indices"],
        )
