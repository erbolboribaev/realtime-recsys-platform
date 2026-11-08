"""Experiment tracking with MLflow (local file store by default)."""

from __future__ import annotations

import os
from collections.abc import Mapping


def sanitize_metric_name(name: str) -> str:
    """MLflow rejects ``@`` in metric names, so ``recall@50`` becomes ``recall_at_50``."""
    return name.replace("@", "_at_")


def uses_file_store(tracking_uri: str) -> bool:
    """True for a plain folder or ``file:`` URI, false for a database or a remote server."""
    return "://" not in tracking_uri or tracking_uri.startswith("file:")


def log_run(
    experiment: str,
    run_name: str,
    params: Mapping[str, object],
    metrics: Mapping[str, float],
    tracking_uri: str,
) -> str:
    """Record one training run and return its MLflow run id."""
    if uses_file_store(tracking_uri):
        # Recent MLflow versions refuse the local-folder backend unless explicitly allowed.
        # Fine for a single-developer project; point MLFLOW_TRACKING_URI at a database
        # (e.g. sqlite:///mlflow.db, needs sqlalchemy + alembic) or a server to move off it.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({key: str(value) for key, value in params.items()})
        mlflow.log_metrics({sanitize_metric_name(key): float(v) for key, v in metrics.items()})
        return run.info.run_id
