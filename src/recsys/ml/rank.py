"""Offline ranking experiment: does the second stage beat plain ALS ordering?

Run: ``python -m recsys.ml.rank``            (logistic ranker, no extra dependencies)
     ``python -m recsys.ml.rank --model lightgbm --onnx models/ranker.onnx``
"""

from __future__ import annotations

import argparse

import numpy as np

from recsys.events.simulator import EventSimulator, SimulatorConfig
from recsys.ml.dataset import deduplicate
from recsys.ml.rankers import LightGBMRanker, LogisticRanker, onnx_scores
from recsys.ml.ranking import (
    CORE_FEATURES,
    FEATURES,
    als_order,
    build_ranking_set,
    evaluate_ranking,
    split_in_three,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["logistic", "lightgbm", "lambdarank"], default="logistic"
    )
    parser.add_argument("--events", type=int, default=100_000)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--features", choices=["core", "all"], default="core")
    parser.add_argument("--ablate", action="store_true", help="drop one feature at a time")
    parser.add_argument("--onnx", help="export the LightGBM ranker here (binary objective)")
    args = parser.parse_args()

    events = [e.model_dump() for e in EventSimulator(SimulatorConfig()).take(args.events)]
    period_a, period_b, period_c = split_in_three(deduplicate(events))

    train = build_ranking_set(period_a, period_b, args.candidates)
    test = build_ranking_set(period_a + period_b, period_c, args.candidates)
    print(
        f"train rows={len(train.y)} (positives {train.y.mean():.1%}), "
        f"test rows={len(test.y)}, users={len(test.truth)}"
    )

    names = CORE_FEATURES if args.features == "core" else FEATURES
    cols = [FEATURES.index(name) for name in names]
    train_x, test_x = train.X[:, cols], test.X[:, cols]
    print(f"features: {', '.join(names)}")

    if args.ablate:
        reference = evaluate_ranking(
            test, LogisticRanker().fit(train_x, train.y).predict(test_x), 10
        )
        print(f"\nablation (logistic), all selected features: ndcg@10={reference['ndcg@10']:.4f}")
        for i, name in enumerate(names):
            keep = [j for j in range(len(names)) if j != i]
            model = LogisticRanker().fit(train_x[:, keep], train.y)
            score = evaluate_ranking(test, model.predict(test_x[:, keep]), 10)["ndcg@10"]
            print(f"  without {name:<20} ndcg@10={score:.4f}")
        print()

    if args.model == "logistic":
        ranker = LogisticRanker().fit(train_x, train.y)
    else:
        ranker = LightGBMRanker("lambdarank" if args.model == "lambdarank" else "binary")
        ranker.fit(train_x, train.y, train.groups)

    k = args.k
    ceiling = evaluate_ranking(test, test.y + 0 * als_order(test), k=args.candidates)
    baseline = evaluate_ranking(test, als_order(test), k)
    ranked = evaluate_ranking(test, ranker.predict(test_x), k)
    print(
        f"candidates contain this share of future items: recall@{args.candidates}="
        f"{ceiling[f'recall@{args.candidates}']:.4f}"
    )
    for name, metrics in (("ALS order", baseline), (f"{args.model} ranker", ranked)):
        print(f"{name:<18} " + "  ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
    gate = ranked[f"ndcg@{k}"] > baseline[f"ndcg@{k}"]
    print(f"\nPhase 3 gate (ranker NDCG@{k} > ALS): {'PASS' if gate else 'FAIL'}")

    if args.model == "logistic":
        for name, weight in sorted(
            zip(names, ranker.weights[1:], strict=True), key=lambda t: -abs(t[1])
        ):
            print(f"  weight {name:<20} {weight:+.3f}")

    if args.onnx:
        if args.model != "lightgbm":
            raise SystemExit("--onnx needs --model lightgbm")
        from pathlib import Path

        Path(args.onnx).parent.mkdir(parents=True, exist_ok=True)
        ranker.export_onnx(args.onnx)
        parity = np.abs(onnx_scores(args.onnx, test_x) - ranker.predict(test_x)).max()
        print(f"ONNX written to {args.onnx}; max |ONNX - LightGBM| = {parity:.2e}")


if __name__ == "__main__":
    main()
