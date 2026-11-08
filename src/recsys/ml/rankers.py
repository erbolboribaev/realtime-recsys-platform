"""Rankers: a dependency-free logistic model and a LightGBM model.

Both score one (user, candidate) row at a time (pointwise learning to rank): the higher
the score, the more likely the user interacts with the item.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class LogisticRanker:
    """L2-regularised logistic regression fitted with Newton's method (NumPy only)."""

    def __init__(self, l2: float = 1.0, iterations: int = 25) -> None:
        self.l2 = l2
        self.iterations = iterations
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self.weights: np.ndarray | None = None

    def _design(self, X: np.ndarray) -> np.ndarray:
        scaled = (X - self._mean) / self._scale
        return np.column_stack([np.ones(len(X)), scaled])

    def fit(self, X: np.ndarray, y: np.ndarray, groups=None) -> LogisticRanker:
        self._mean = X.mean(axis=0)
        self._scale = X.std(axis=0) + 1e-9
        design = self._design(X)
        penalty = self.l2 * np.eye(design.shape[1])
        penalty[0, 0] = 0.0  # never penalise the intercept
        w = np.zeros(design.shape[1])
        for _ in range(self.iterations):
            p = 1.0 / (1.0 + np.exp(-np.clip(design @ w, -30, 30)))
            gradient = design.T @ (p - y) + penalty @ w
            hessian = design.T @ (design * (p * (1 - p) + 1e-9)[:, None]) + penalty
            step = np.linalg.solve(hessian, gradient)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.weights = w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("fit() first")
        return self._design(X) @ self.weights


class LightGBMRanker:
    """LightGBM ranker.

    ``objective="binary"`` is pointwise and exports to ONNX with the same
    ``onnxmltools`` call the fraud-detection project uses. ``"lambdarank"`` optimises NDCG
    directly per user group; it is for offline comparison (ONNX export is not supported).
    """

    def __init__(self, objective: str = "binary", n_estimators: int = 400, seed: int = 0) -> None:
        if objective not in {"binary", "lambdarank"}:
            raise ValueError("objective must be 'binary' or 'lambdarank'")
        self.objective = objective
        self.n_estimators = n_estimators
        self.seed = seed
        self.model = None
        self.n_features = 0

    def fit(self, X: np.ndarray, y: np.ndarray, groups=None) -> LightGBMRanker:
        import lightgbm as lgb

        self.n_features = X.shape[1]
        # Small trees, large leaves and strong L2: the signal here is weak, so a flexible
        # model mostly learns noise (the first, unregularised version did worse than ALS).
        common = {
            "n_estimators": self.n_estimators,
            "learning_rate": 0.03,
            "num_leaves": 7,
            "min_child_samples": 200,
            "reg_lambda": 10.0,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "random_state": self.seed,
            "verbose": -1,
        }
        if self.objective == "lambdarank" and groups is None:
            raise ValueError("lambdarank needs the group sizes")

        # Hold out the last 15% of users (whole groups) to stop training early.
        fit_kwargs: dict = {}
        train_rows, train_groups = len(y), groups
        if groups is not None and len(groups) >= 20:
            cut_group = int(len(groups) * 0.85)
            train_rows = int(sum(groups[:cut_group]))
            train_groups = list(groups[:cut_group])
            fit_kwargs["eval_set"] = [(X[train_rows:], y[train_rows:])]
            fit_kwargs["callbacks"] = [lgb.early_stopping(30, verbose=False)]
            if self.objective == "lambdarank":
                fit_kwargs["eval_group"] = [list(groups[cut_group:])]
        X_fit, y_fit = X[:train_rows], y[:train_rows]

        if self.objective == "binary":
            self.model = lgb.LGBMClassifier(**common).fit(X_fit, y_fit, **fit_kwargs)
        else:
            self.model = lgb.LGBMRanker(objective="lambdarank", **common).fit(
                X_fit, y_fit.astype(int), group=train_groups, **fit_kwargs
            )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit() first")
        if self.objective == "binary":
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def export_onnx(self, path: str | Path) -> None:
        if self.objective != "binary":
            raise ValueError("ONNX export is only supported for the binary objective")
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType

        onnx_model = onnxmltools.convert_lightgbm(
            self.model,
            initial_types=[("input", FloatTensorType([None, self.n_features]))],
            zipmap=False,
        )
        Path(path).write_bytes(onnx_model.SerializeToString())


class OnnxRanker:
    """An exported ranker loaded once and reused for every request."""

    def __init__(self, path: str | Path) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Probability of the positive class for each row."""
        probabilities = self._session.run(None, {self._input: X.astype(np.float32)})[1]
        return np.asarray(probabilities)[:, 1]


def onnx_scores(path: str | Path, X: np.ndarray) -> np.ndarray:
    """Score rows with an exported ONNX ranker (loads the model on every call)."""
    return OnnxRanker(path)(X)
