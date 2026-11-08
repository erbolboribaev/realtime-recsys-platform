"""See explanations for a real user: recommendations from the API, history from Redis.

Run: ``python -m recsys.llm.explain_demo --user user_00001``
(API on --url, Redis up, GROQ_API_KEY / LLM_API_KEY set)
"""

from __future__ import annotations

import argparse

import httpx
import redis as redis_lib

from recsys.config import Settings
from recsys.events.simulator import EventSimulator, SimulatorConfig
from recsys.llm.catalog import SimulatedCatalog
from recsys.llm.explain import Explainer
from recsys.llm.gateway import build_gateway
from recsys.serving.features import read_user_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="user_00001")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    settings = Settings.from_env()
    store = redis_lib.Redis.from_url(settings.redis_url)
    catalog = SimulatedCatalog(EventSimulator(SimulatorConfig()).catalog)
    explainer = Explainer(build_gateway(settings, store))

    reply = httpx.get(
        f"{args.url}/recommend", params={"user_id": args.user, "k": args.k}, timeout=10
    )
    reply.raise_for_status()
    body = reply.json()
    _, recent_ids = read_user_state(store, args.user)

    recommended = [i for i in map(catalog.describe, body["items"]) if i]
    history = [i for i in map(catalog.describe, recent_ids) if i]

    print(f"user {args.user}  variant={body['variant']}  source={body['source']}")
    print("recently viewed:", ", ".join(i.name for i in history[:5]) or "-")
    print("recommended:")
    for item in recommended:
        print(f"  {item.name:<28} {item.category:<16} ${item.price:.2f}")
    result = explainer.explain(history, recommended)
    print(f"\nexplanation [{result.source}{' ' + result.reason if result.reason else ''}]:")
    print(f"  {result.text}")


if __name__ == "__main__":
    main()
