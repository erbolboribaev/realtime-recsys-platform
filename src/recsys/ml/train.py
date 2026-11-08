"""Train ALS, compare it with the popularity baseline, and publish it.

Run: ``python -m recsys.ml.train [--track] [--index]``

* ``--track``  log parameters and metrics to MLflow (needs ``pip install -e ".[ml]"``)
* ``--index``  load item vectors into Qdrant and verify that retrieval through the index
               returns exactly what the model itself would recommend (needs ``make up``)
"""

from __future__ import annotations

import argparse

from recsys.config import Settings
from recsys.events.simulator import EventSimulator, SimulatorConfig
from recsys.ml.als import ALSModel
from recsys.ml.baselines import PopularityModel
from recsys.ml.dataset import build_interactions, deduplicate, future_items, time_split
from recsys.ml.evaluate import load_lake_events
from recsys.ml.index import NumpyItemIndex
from recsys.ml.metrics import evaluate
from recsys.ml.retrieval import Retriever


def index_agreement(model, interactions, index, k: int, sample: int = 100) -> float:
    """Share of sampled users whose indexed top-k equals the model's own top-k."""
    retriever = Retriever(model, interactions, index)
    retriever.publish()
    users = list(range(0, interactions.n_users, max(interactions.n_users // sample, 1)))
    same = sum(retriever.recommend(u, k) == model.recommend(u, k) for u in users)
    return same / len(users)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["simulator", "lake"], default="simulator")
    parser.add_argument("--events", type=int, default=100_000)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--factors", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--index", action="store_true")
    args = parser.parse_args()
    settings = Settings.from_env()

    if args.source == "lake":
        events = load_lake_events()
    else:
        events = [e.model_dump() for e in EventSimulator(SimulatorConfig()).take(args.events)]

    split = time_split(deduplicate(events), test_fraction=0.2)
    interactions = build_interactions(split.train)
    truth = future_items(split.test, interactions)

    model = ALSModel(factors=args.factors, iterations=args.iterations).fit(interactions)
    baseline = evaluate(PopularityModel(interactions).recommend, truth, args.k)
    als = evaluate(model.recommend, truth, args.k)
    key = f"recall@{args.k}"
    print(f"popularity {key}={baseline[key]:.4f}   als {key}={als[key]:.4f}")

    if args.index:
        from qdrant_client import QdrantClient

        from recsys.ml.index import QdrantItemIndex

        qdrant = QdrantItemIndex(
            QdrantClient(url=settings.qdrant_url, check_compatibility=False),
            settings.qdrant_collection,
        )
        agreement = index_agreement(model, interactions, qdrant, args.k)
        print(f"Qdrant retrieval matches the model for {agreement:.0%} of sampled users")
        exact = index_agreement(model, interactions, NumpyItemIndex(), args.k)
        print(f"reference index agreement: {exact:.0%}")

    if args.track:
        from recsys.ml.tracking import log_run

        metrics = {f"als_{k}": v for k, v in als.items()}
        metrics.update({f"popularity_{k}": v for k, v in baseline.items()})
        run_id = log_run(
            "als-retrieval",
            f"als-f{args.factors}-i{args.iterations}",
            {"factors": args.factors, "iterations": args.iterations, "events": len(events)},
            metrics,
            settings.mlflow_tracking_uri,
        )
        print(f"logged MLflow run {run_id} to {settings.mlflow_tracking_uri}")


if __name__ == "__main__":
    main()
