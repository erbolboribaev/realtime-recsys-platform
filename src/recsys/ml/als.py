"""Implicit-feedback ALS (Hu, Koren, Volinsky 2008) in plain NumPy.

Each observed (user, item) pair has a confidence ``1 + alpha * strength`` that the user
likes the item; unobserved pairs count as weak negatives. Alternating least squares solves
for user factors with item factors fixed, then the reverse. Every step is an exact
regularised least-squares solve, so the loss never goes up.
"""

from __future__ import annotations

import numpy as np

from recsys.ml.baselines import top_k_unseen
from recsys.ml.dataset import Interactions


class ALSModel:
    def __init__(
        self,
        factors: int = 32,
        regularization: float = 0.1,
        alpha: float = 20.0,
        iterations: int = 10,
        seed: int = 0,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self._interactions: Interactions | None = None

    def _solve_side(
        self, fixed: np.ndarray, rows: list[tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        gram = fixed.T @ fixed
        eye = self.regularization * np.eye(self.factors)
        out = np.zeros((len(rows), self.factors))
        for i, (indices, strengths) in enumerate(rows):
            if len(indices) == 0:
                continue
            confidence = 1.0 + self.alpha * strengths
            vectors = fixed[indices]
            a = gram + (vectors * (confidence - 1.0)[:, None]).T @ vectors + eye
            b = (vectors * confidence[:, None]).sum(axis=0)
            out[i] = np.linalg.solve(a, b)
        return out

    def loss(self) -> float:
        """Weighted squared error + regularisation over *all* pairs (small data only)."""
        assert self.user_factors is not None and self.item_factors is not None
        inter = self._interactions
        assert inter is not None
        prediction = self.user_factors @ self.item_factors.T
        preference = np.zeros_like(prediction)
        confidence = np.ones_like(prediction)
        for user, (items, strengths) in enumerate(inter.by_user):
            preference[user, items] = 1.0
            confidence[user, items] = 1.0 + self.alpha * strengths
        error = (confidence * (preference - prediction) ** 2).sum()
        reg = self.regularization * ((self.user_factors**2).sum() + (self.item_factors**2).sum())
        return float(error + reg)

    def fit(self, interactions: Interactions) -> ALSModel:
        rng = np.random.default_rng(self.seed)
        self._interactions = interactions
        self.item_factors = rng.normal(0, 0.1, (interactions.n_items, self.factors))
        self.user_factors = np.zeros((interactions.n_users, self.factors))
        for _ in range(self.iterations):
            self.user_factors = self._solve_side(self.item_factors, interactions.by_user)
            self.item_factors = self._solve_side(self.user_factors, interactions.by_item)
        return self

    def recommend(self, user: int, k: int) -> list[int]:
        assert self.user_factors is not None and self.item_factors is not None
        assert self._interactions is not None
        scores = self.item_factors @ self.user_factors[user]
        return top_k_unseen(scores, self._interactions.seen_items(user), k)

    def similar_items(self, item: int, k: int) -> list[int]:
        """Nearest items by cosine similarity (basis for the vector index in Qdrant)."""
        assert self.item_factors is not None
        norms = np.linalg.norm(self.item_factors, axis=1) + 1e-12
        scores = (self.item_factors @ self.item_factors[item]) / (norms * norms[item])
        return top_k_unseen(scores, np.array([item]), k)
