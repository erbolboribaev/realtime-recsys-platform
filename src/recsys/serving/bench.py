"""Latency benchmark for a running API.

Run: ``python -m recsys.serving.bench --requests 500``  (API must be up on --url)

Reports the server-measured latency (what the service itself spent) and the client
round trip, per variant and per stage. The Phase 3 gate is server p95 < 100 ms.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

GATE_P95_MS = 100.0


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (``pct`` in 0..100) of a non-empty list."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * pct // 100))  # ceil without importing math
    return ordered[int(rank) - 1]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "n": float(len(values)),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def main() -> None:
    import httpx

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--users", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=30)
    args = parser.parse_args()

    # 127.0.0.1 by default: on Windows "localhost" can try IPv6 first and add latency.
    client = httpx.Client(base_url=args.url, timeout=10.0)

    def call(i: int) -> dict:
        began = time.perf_counter()
        body = client.get(
            "/recommend", params={"user_id": f"user_{i % args.users:05d}", "k": args.k}
        ).json()
        body["client_ms"] = (time.perf_counter() - began) * 1000.0
        return body

    for i in range(args.warmup):  # ONNX session, connection pools, Kafka producer
        call(i)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(call, range(args.requests)))

    by_variant: dict[str, list[float]] = defaultdict(list)
    by_stage: dict[str, list[float]] = defaultdict(list)
    for body in results:
        by_variant[body["variant"]].append(body["latency_ms"])
        for stage, value in body.get("timings_ms", {}).items():
            by_stage[stage].append(value)

    print(f"{'':<14}{'n':>6}{'mean':>9}{'p50':>9}{'p95':>9}{'p99':>9}  (ms)")
    rows = {"ALL (server)": [r["latency_ms"] for r in results]}
    rows.update({f"{k} (server)": v for k, v in sorted(by_variant.items())})
    rows["ALL (client)"] = [r["client_ms"] for r in results]
    for name, values in rows.items():
        s = summarize(values)
        print(
            f"{name:<14}{int(s['n']):>6}{s['mean']:>9.1f}{s['p50']:>9.1f}{s['p95']:>9.1f}{s['p99']:>9.1f}"
        )

    print("\nper stage (p50 / p95 ms):")
    for stage, values in by_stage.items():
        print(f"  {stage:<12}{percentile(values, 50):>8.1f}{percentile(values, 95):>8.1f}")

    p95 = summarize([r["latency_ms"] for r in results])["p95"]
    verdict = "PASS" if p95 < GATE_P95_MS else "FAIL"
    print(f"\nPhase 3 gate (server p95 < {GATE_P95_MS:.0f} ms): {verdict}")


if __name__ == "__main__":
    main()
