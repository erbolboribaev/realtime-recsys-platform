"""Train the production models and write everything the API needs.

Run: ``python -m recsys.ml.publish [--index]``   (needs ``pip install -e ".[ml,rank]"``)

Output folder (default ``models/``): ``ranker.onnx``, ``factors.npz``, ``meta.json``.
``--index`` also loads the item vectors into Qdrant (needs ``make up``).

The ranker is trained on two periods (A -> B and A+B -> C), each built from older history
than its labels. The serving ALS is then refitted on all events, which matches how the
offline test was run (history first, labels later).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from recsys.config import Settings
from recsys.events.simulator import EventSimulator, SimulatorConfig
from recsys.ml.als import ALSModel
from recsys.ml.dataset import build_interactions, deduplicate
from recsys.ml.rankers import LightGBMRanker
from recsys.ml.ranking import CORE_FEATURES, FEATURES, build_ranking_set, split_in_three
from recsys.serving.bundle import Bundle, save_bundle


def publish(
    events: Sequence[dict[str, Any]],
    out_dir: str | Path,
    n_candidates: int = 100,
    factors: int = 32,
    iterations: int = 10,
) -> dict[str, Any]:
    events = deduplicate(events)
    period_a, period_b, period_c = split_in_three(events)
    cols = [FEATURES.index(name) for name in CORE_FEATURES]

    first = build_ranking_set(period_a, period_b, n_candidates, factors, iterations)
    second = build_ranking_set(period_a + period_b, period_c, n_candidates, factors, iterations)
    X = np.vstack([first.X[:, cols], second.X[:, cols]])
    y = np.concatenate([first.y, second.y])
    ranker = LightGBMRanker("binary").fit(X, y, first.groups + second.groups)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ranker.export_onnx(out / "ranker.onnx")

    interactions = build_interactions(events)
    als = ALSModel(factors=factors, iterations=iterations).fit(interactions)
    bundle = Bundle.from_training(als, interactions, CORE_FEATURES, n_candidates)
    save_bundle(out, bundle)
    return {
        "ranker_rows": len(y),
        "users": interactions.n_users,
        "items": interactions.n_items,
        "bundle": bundle,
        "model": als,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["simulator", "lake"], default="simulator")
    parser.add_argument("--events", type=int, default=100_000)
    parser.add_argument("--out", default="models")
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--index", action="store_true", help="load item vectors into Qdrant")
    args = parser.parse_args()

    if args.source == "lake":
        from recsys.ml.evaluate import load_lake_events

        events = load_lake_events()
    else:
        events = [e.model_dump() for e in EventSimulator(SimulatorConfig()).take(args.events)]

    summary = publish(events, args.out, args.candidates)
    print(
        f"published to {args.out}/: ranker trained on {summary['ranker_rows']} rows, "
        f"{summary['users']} users, {summary['items']} items"
    )

    if args.index:
        from qdrant_client import QdrantClient

        from recsys.ml.index import QdrantItemIndex

        settings = Settings.from_env()
        bundle = summary["bundle"]
        QdrantItemIndex(
            QdrantClient(url=settings.qdrant_url, check_compatibility=False),
            settings.qdrant_collection,
        ).build(bundle.item_factors, bundle.item_ids)
        collection = settings.qdrant_collection
        print(f"indexed {len(bundle.item_ids)} items in Qdrant collection {collection}")


if __name__ == "__main__":
    main()
