"""Offline evaluation: popularity baseline vs ALS on a time-based split.

Run: ``python -m recsys.ml.evaluate`` (simulated events, no infrastructure needed)
or  ``python -m recsys.ml.evaluate --source lake`` (reads the Parquet lake from Phase 1).
"""

from __future__ import annotations

import argparse
import time

from recsys.events.simulator import EventSimulator, SimulatorConfig
from recsys.ml.als import ALSModel
from recsys.ml.baselines import PopularityModel
from recsys.ml.dataset import (
    build_interactions,
    deduplicate,
    future_items,
    time_split,
)
from recsys.ml.metrics import evaluate


def load_lake_events() -> list[dict]:
    import pyarrow.parquet as pq

    from recsys.config import Settings
    from recsys.sink.storage import lake_root, s3_filesystem

    settings = Settings.from_env()
    fs = s3_filesystem(settings)
    events: list[dict] = []
    for path in sorted(fs.glob(f"{lake_root(settings)}/dt=*/*.parquet")):
        with fs.open(path, "rb") as handle:
            events.extend(pq.read_table(handle).to_pylist())
    return events


def run(events: list[dict], k: int, factors: int, iterations: int) -> dict[str, dict[str, float]]:
    events = deduplicate(events)
    split = time_split(events, test_fraction=0.2)
    interactions = build_interactions(split.train)
    truth = future_items(split.test, interactions)

    popularity = PopularityModel(interactions)
    started = time.perf_counter()
    als = ALSModel(factors=factors, iterations=iterations).fit(interactions)
    print(f"ALS trained in {time.perf_counter() - started:.1f}s")

    print(
        f"train events={len(split.train)} test events={len(split.test)} "
        f"users={interactions.n_users} items={interactions.n_items} eval users={len(truth)}"
    )
    return {
        "popularity": evaluate(popularity.recommend, truth, k),
        "als": evaluate(als.recommend, truth, k),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["simulator", "lake"], default="simulator")
    parser.add_argument("--events", type=int, default=100_000)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--factors", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    if args.source == "lake":
        events = load_lake_events()
    else:
        events = [e.model_dump() for e in EventSimulator(SimulatorConfig()).take(args.events)]

    results = run(events, args.k, args.factors, args.iterations)
    print()
    for name, metrics in results.items():
        line = "  ".join(f"{key}={value:.4f}" for key, value in metrics.items() if key != "users")
        print(f"{name:<11} {line}")
    gate = results["als"][f"recall@{args.k}"] > results["popularity"][f"recall@{args.k}"]
    print(f"\nPhase 2 gate (ALS Recall@{args.k} > popularity): {'PASS' if gate else 'FAIL'}")


if __name__ == "__main__":
    main()
