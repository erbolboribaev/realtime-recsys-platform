"""Measure explanation quality on a fixed set of cases, and fail when it drops.

Each case is a shopper (recent items) plus the products we would recommend. We ask the
explainer for an explanation and score what the *model* said (before the guardrail turns a
bad answer into a template):

* ``accepted``      the answer passed the guardrail (no invented price or product, sane length)
* ``on_topic``      it mentions the browsed category or one of the recommended products
* ``clean``         it avoids claims we have no data for (looks, sound, quality, deals)
* ``focused``       it names at most three products, as the prompt asks
* ``fallback``      the share that ended up as a template (model down or answer rejected)

Run live (needs a key; costs a fraction of a cent):
    python -m recsys.llm.evals --n 30
The scoring is covered by tests with fake models, so CI checks the harness without a key.
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass, field

import numpy as np

from recsys.llm.catalog import SINGULAR, ItemInfo, SimulatedCatalog
from recsys.llm.explain import _MODEL, Explainer

# Qualities and offers the prompt forbids: we have no data about any of them.
UNSUPPORTED_CLAIM = re.compile(
    r"\b(stylish|sleek|elegant|great|amazing|awesome|best|perfect|excellent|fantastic|"
    r"comfortable|durable|powerful|premium|high[- ]quality|top[- ]rated|"
    r"sound|battery|discount|sale|deal|cheap|bargain|free shipping)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Case:
    history: list[ItemInfo]
    recommended: list[ItemInfo]
    main_category: str


@dataclass
class Report:
    n: int
    accepted: float
    on_topic: float
    clean: float
    focused: float  # mentions at most three products, as the prompt asks
    fallback: float
    avg_cost_micro_usd: float
    failures: list[str] = field(default_factory=list)


DEFAULT_MINIMUMS = {"accepted": 0.90, "on_topic": 0.90, "clean": 0.90, "focused": 0.90}


def build_cases(catalog: SimulatedCatalog, n: int, seed: int = 0) -> list[Case]:
    """Deterministic shoppers: 3 recent items of one main category (+1 stray), 5 suggestions."""
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n):
        main, other = rng.choice(catalog.n_categories, size=2, replace=False)
        in_main = catalog.item_ids_in(int(main))
        in_other = catalog.item_ids_in(int(other))
        if len(in_main) < 6 or len(in_other) < 3:
            continue
        picks = [str(i) for i in rng.choice(in_main, size=6, replace=False)]
        stray = str(rng.choice(in_other))
        history = [catalog.describe(i) for i in [*picks[:3], stray]]
        recommended = [
            catalog.describe(i) for i in [*picks[3:6], *map(str, rng.choice(in_other, 2))]
        ]
        cases.append(Case(history, recommended, history[0].category))
    return cases


def _on_topic(text: str, case: Case) -> bool:
    lowered = text.lower()
    names = [case.main_category.lower(), *(item.name.lower() for item in case.recommended)]
    singular = [s.lower() for s in SINGULAR if case.main_category.lower().startswith(s.lower()[:6])]
    return any(name in lowered for name in [*names, *singular])


def run_evals(explainer: Explainer, cases: list[Case], delay_s: float = 0.0) -> Report:
    accepted = on_topic = clean = focused = fallback = answered = 0
    cost = 0
    failures: list[str] = []
    kinds: dict[str, int] = {}

    def note(kind: str, detail: str) -> None:
        """Keep up to three examples of each kind of problem."""
        if kinds.get(kind, 0) < 3:
            kinds[kind] = kinds.get(kind, 0) + 1
            failures.append(f"{kind}: {detail}")

    unavailable: dict[str, int] = {}
    for n, case in enumerate(cases):
        if delay_s and n:
            time.sleep(delay_s)  # stay under the provider's rate limit
        result = explainer.explain(case.history, case.recommended)
        cost += result.cost_micro_usd
        if result.source == "template":
            fallback += 1
        if not result.raw_text:
            unavailable[result.reason] = unavailable.get(result.reason, 0) + 1
            continue  # the model produced nothing to judge
        answered += 1
        if result.source == "llm":
            accepted += 1
        else:
            note("rejected", f"({result.reason}) {result.raw_text[:220]}")
        if _on_topic(result.raw_text, case):
            on_topic += 1
        else:
            note("off topic", result.raw_text[:220])
        if len(set(_MODEL.findall(result.raw_text))) <= 3:
            focused += 1
        else:
            note("too many products", result.raw_text[:220])
        claim = UNSUPPORTED_CLAIM.search(result.raw_text)
        if claim is None:
            clean += 1
        else:
            note(f"unsupported claim '{claim.group(0)}'", result.raw_text[:220])

    for reason, count in sorted(unavailable.items(), key=lambda item: -item[1]):
        failures.insert(0, f"no answer x{count}: {reason[:200]}")

    def share(count: int, total: int) -> float:
        return count / total if total else 0.0

    return Report(
        n=len(cases),
        accepted=share(accepted, answered),
        on_topic=share(on_topic, answered),
        clean=share(clean, answered),
        focused=share(focused, answered),
        fallback=share(fallback, len(cases)),
        avg_cost_micro_usd=cost / len(cases) if cases else 0.0,
        failures=failures,
    )


def violations(report: Report, minimums: dict[str, float] | None = None) -> list[str]:
    """Metrics below their minimum; empty means the evals pass."""
    problems = []
    for name, floor in (minimums or DEFAULT_MINIMUMS).items():
        value = getattr(report, name)
        if value < floor:
            problems.append(f"{name} {value:.0%} is below the minimum {floor:.0%}")
    if report.n == 0:
        problems.append("no cases were evaluated")
    return problems


def main() -> None:
    import redis as redis_lib

    from recsys.config import Settings
    from recsys.events.simulator import EventSimulator, SimulatorConfig
    from recsys.llm.gateway import build_gateway

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    args = parser.parse_args()

    settings = Settings.from_env()
    gateway = build_gateway(settings, redis_lib.Redis.from_url(settings.redis_url))
    catalog = SimulatedCatalog(EventSimulator(SimulatorConfig()).catalog)
    report = run_evals(Explainer(gateway), build_cases(catalog, args.n, args.seed), args.delay)

    print(f"cases {report.n}   model {gateway._client.model}")
    print(f"  accepted by guardrail   {report.accepted:.0%}")
    print(f"  on topic                {report.on_topic:.0%}")
    print(f"  free of unsupported claims {report.clean:.0%}")
    print(f"  at most 3 products      {report.focused:.0%}")
    print(f"  fell back to template   {report.fallback:.0%}")
    print(f"  avg cost per explanation ${report.avg_cost_micro_usd / 1e6:.6f}")
    for line in report.failures[:10]:
        print("  example:", line)
    problems = violations(report)
    for line in problems:
        print("FAIL:", line)
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
