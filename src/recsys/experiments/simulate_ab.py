"""Online A/B experiment against a running API, with simulated users.

Each simulated user asks for recommendations, looks at the list top to bottom (higher
positions are noticed more often) and clicks items they would genuinely like. "Genuinely
like" uses the simulator's hidden category tastes and item popularity: the same process
that generated the training events. A click becomes a view (sometimes a cart add and a
purchase) in the Redis feature store, so later requests see the user's new history.

This measures whether the ranker beats ALS *inside the simulated world*. The click model is
an assumption (``--click-scale``); the point is the machinery (assignment, attribution,
significance testing), not a claim about real shoppers.

Run: ``python -m recsys.experiments.simulate_ab --rounds 5``   (API on --url, Redis up)
"""

from __future__ import annotations

import argparse
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from recsys.events.catalog import Catalog
from recsys.events.simulator import EventSimulator, SimulatorConfig
from recsys.experiments.ab_analysis import compare_means


@dataclass(frozen=True)
class ClickModel:
    preferences: np.ndarray  # (n_users, n_categories) true tastes
    catalog: Catalog
    scale: float = 2.0
    position_decay: float = 0.25
    max_probability: float = 0.9

    def relevance(self, user: int) -> np.ndarray:
        """Probability that the user's next organic view is each item."""
        popularity = self.catalog.popularity
        category_totals = np.bincount(
            self.catalog.categories, weights=popularity, minlength=self.catalog.n_categories
        )
        within_category = popularity / category_totals[self.catalog.categories]
        return self.preferences[user][self.catalog.categories] * within_category

    def click_probability(self, user: int, item: int, position: int) -> float:
        examined = 1.0 / (1.0 + self.position_decay * position)
        return min(self.max_probability, self.scale * self.relevance(user)[item]) * examined


Call = Callable[[str, int], dict]
Engage = Callable[[dict], None]


def run_experiment(
    call: Call,
    engage: Engage,
    model: ClickModel,
    user_ids: list[str],
    rounds: int,
    k: int,
    rng: np.random.Generator,
    p_cart: float = 0.12,
    p_purchase: float = 0.35,
) -> dict[str, dict[str, list[tuple[int, int]]]]:
    """Returns ``{variant: {user_id: [(any_click, n_clicks), ...one per round]}}``."""
    item_index = {item: i for i, item in enumerate(model.catalog.item_ids)}
    user_index = {user: i for i, user in enumerate(user_ids)}
    results: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))

    for _ in range(rounds):
        for user_id in rng.permutation(user_ids):
            user_id = str(user_id)
            response = call(user_id, k)
            clicked = [
                item
                for position, item in enumerate(response["items"])
                if rng.random()
                < model.click_probability(user_index[user_id], item_index[item], position)
            ]
            results[response["variant"]][user_id].append((int(bool(clicked)), len(clicked)))

            for item in clicked:
                kinds = ["view"]
                if rng.random() < p_cart:
                    kinds.append("add_to_cart")
                    if rng.random() < p_purchase:
                        kinds.append("purchase")
                for kind in kinds:
                    engage(
                        {
                            "event_id": str(uuid.uuid4()),
                            "user_id": user_id,
                            "session_id": response["request_id"],
                            "item_id": item,
                            "event_type": kind,
                            "timestamp": datetime.now(UTC),
                            "request_id": response["request_id"],
                        }
                    )
    return results


def summarize_experiment(results) -> dict[str, dict[str, float]]:
    """Compare treatment with control on per-user click rates (users are independent)."""
    metrics = {"clicked_request_rate": 0, "clicks_per_request": 1}
    out = {}
    for name, position in metrics.items():
        arms = {
            variant: [float(np.mean([r[position] for r in rounds])) for rounds in users.values()]
            for variant, users in results.items()
            if variant in {"control", "treatment"}
        }
        out[name] = compare_means(arms["control"], arms["treatment"])
    return out


def expected_clicks(model: ClickModel, user: int, item_indices: list[int]) -> float:
    """Expected number of clicks on a list: the click model's probabilities, no randomness."""
    return float(
        sum(
            model.click_probability(user, item, position)
            for position, item in enumerate(item_indices)
        )
    )


def expected_clicks_by_variant(
    call: Call, model: ClickModel, user_ids: list[str], k: int
) -> dict[str, list[float]]:
    """One request per user, no engagement: the click model's expected clicks per list.

    Removes click-sampling noise and the feedback loop, so it isolates how good the lists are
    *under the click model*. ``{variant: [expected clicks per user]}``.
    """
    item_index = {item: i for i, item in enumerate(model.catalog.item_ids)}
    user_index = {user: i for i, user in enumerate(user_ids)}
    out: dict[str, list[float]] = defaultdict(list)
    for user_id in user_ids:
        response = call(user_id, k)
        items = [item_index[item] for item in response["items"]]
        out[response["variant"]].append(expected_clicks(model, user_index[user_id], items))
    return out


def per_round_rates(results, metric: int = 0) -> dict[str, list[float]]:
    """Mean per-user metric in each round, per variant: shows drift (feedback loops)."""
    out = {}
    for variant, users in results.items():
        if variant not in {"control", "treatment"}:
            continue
        n_rounds = min(len(rounds) for rounds in users.values())
        out[variant] = [
            float(np.mean([rounds[r][metric] for rounds in users.values()]))
            for r in range(n_rounds)
        ]
    return out


def main() -> None:
    import httpx
    import redis as redis_lib

    from recsys.config import Settings
    from recsys.features.online import apply_event

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--users", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--click-scale", type=float, default=2.0)
    args = parser.parse_args()

    world = EventSimulator(SimulatorConfig(n_users=args.users))  # same seed as the models
    model = ClickModel(world.user_preferences, world.catalog, scale=args.click_scale)
    user_ids = [world.user_id(i) for i in range(args.users)]

    http = httpx.Client(base_url=args.url, timeout=10.0)
    store = redis_lib.Redis.from_url(Settings.from_env().redis_url)

    def call(user_id: str, k: int) -> dict:
        response = http.get("/recommend", params={"user_id": user_id, "k": k})
        response.raise_for_status()  # a 503 body has no "items": fail loudly, not with KeyError
        return response.json()

    results = run_experiment(
        call,
        lambda event: apply_event(store, event),
        model,
        user_ids,
        args.rounds,
        args.k,
        np.random.default_rng(args.seed),
    )

    summary = summarize_experiment(results)
    for name, s in summary.items():
        print(
            f"\n{name}  (users: control={int(s['n_control'])}, treatment={int(s['n_treatment'])})"
        )
        print(f"  control   {s['mean_control']:.4f}")
        print(f"  treatment {s['mean_treatment']:.4f}   lift {s['lift']:+.1%}")
        print(f"  diff {s['diff']:+.4f}  95% CI [{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]")
        print(f"  z={s['z']:.2f}  p={s['p_value']:.4f}")

    rates = per_round_rates(results)
    if rates.get("control") and rates.get("treatment"):
        print("\nclicked_request_rate by round (watch for drift):")
        print("  round  control  treatment")
        for r, (c, t) in enumerate(zip(rates["control"], rates["treatment"], strict=True), 1):
            print(f"  {r:>5}  {c:.4f}   {t:.4f}")


if __name__ == "__main__":
    main()
