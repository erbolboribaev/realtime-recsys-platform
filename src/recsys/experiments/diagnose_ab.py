"""Noise-free check of the two arms: expected clicks per list under the click model.

Asks the running API for one list per user (no clicks are sent back, so Redis is not
modified and there is no feedback loop), then scores each list with the click model's
*probabilities* instead of sampling clicks. The remaining randomness is only in who is
in which arm, so a few hundred users are enough to see a real quality gap.

Run: ``python -m recsys.experiments.diagnose_ab``   (API on --url)
"""

from __future__ import annotations

import argparse

from recsys.events.simulator import EventSimulator, SimulatorConfig
from recsys.experiments.ab_analysis import compare_means
from recsys.experiments.simulate_ab import ClickModel, expected_clicks_by_variant


def main() -> None:
    import httpx

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--users", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--click-scale", type=float, default=2.0)
    args = parser.parse_args()

    world = EventSimulator(SimulatorConfig(n_users=args.users))
    model = ClickModel(world.user_preferences, world.catalog, scale=args.click_scale)
    user_ids = [world.user_id(i) for i in range(args.users)]
    http = httpx.Client(base_url=args.url, timeout=10.0)

    def call(user_id: str, k: int) -> dict:
        response = http.get("/recommend", params={"user_id": user_id, "k": k})
        response.raise_for_status()
        return response.json()

    by_variant = expected_clicks_by_variant(call, model, user_ids, args.k)
    s = compare_means(by_variant["control"], by_variant["treatment"])
    print(
        f"expected clicks per list  (users: control={int(s['n_control'])}, "
        f"treatment={int(s['n_treatment'])})"
    )
    print(f"  control (als)      {s['mean_control']:.4f}")
    print(f"  treatment (ranker) {s['mean_treatment']:.4f}   lift {s['lift']:+.1%}")
    print(
        f"  diff {s['diff']:+.4f}  95% CI [{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]  "
        f"p={s['p_value']:.4f}"
    )


if __name__ == "__main__":
    main()
