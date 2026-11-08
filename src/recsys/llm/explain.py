"""Short, grounded explanations of why items were recommended.

``Explainer.explain`` never raises and never invents facts:

1. The prompt contains only what we know (recent categories, recommended items with
   category and price) and tells the model to use nothing else.
2. The answer is checked (length, only listed names and prices). A reply that fails the check
   is discarded.
3. When the model is unavailable, over budget or fails the check, a deterministic template
   is returned instead, so the product always has an explanation.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from recsys.llm.catalog import ItemInfo
from recsys.llm.gateway import Completion, LLMError

log = logging.getLogger("recsys.llm.explain")

MAX_CHARS = 400
MAX_LISTED = 3  # the model only sees (and may only name) this many products

SYSTEM_PROMPT = (
    "You write one or two short, friendly sentences telling a shopper why these products "
    "are suggested. Use ONLY the facts provided: the categories they recently browsed and "
    "the listed products.\n"
    "Rules:\n"
    "1. Mention at most three products in total, copying each name exactly.\n"
    "2. Do not judge the products. Never use words like great, best, perfect, amazing, "
    "stylish or ideal, and never say how a product looks, sounds or performs.\n"
    "3. Never mention prices, discounts or reviews.\n"
    "4. Only connect what the shopper browsed to the suggested categories, for example: "
    "'Since you have been browsing X, here are some Y you might want to look at.'\n"
    "5. No quotes, lists or emojis."
)


class Gateway(Protocol):
    def complete(self, messages: list[dict], max_tokens: int = ..., **kwargs) -> Completion: ...


@dataclass(frozen=True)
class Explanation:
    text: str
    source: str  # "llm" or "template"
    reason: str = ""  # why the template was used
    cost_micro_usd: int = 0
    cached: bool = False
    raw_text: str = ""  # what the model said, even when the guardrail dropped it (for evals)


def build_messages(history: list[ItemInfo], recommended: list[ItemInfo]) -> list[dict]:
    seen = Counter(item.category for item in history)
    browsed = ", ".join(f"{name} (x{n})" for name, n in seen.most_common(3)) or "nothing yet"
    lines = [
        f"- {item.name} | category: {item.category} | price: ${item.price:.2f}"
        for item in recommended[:MAX_LISTED]
    ]
    user = f"Recently browsed categories: {browsed}\nSuggested products:\n" + "\n".join(lines)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def template_explanation(history: list[ItemInfo], recommended: list[ItemInfo]) -> str:
    if not recommended:
        return "Here are some popular items to get you started."
    top = Counter(item.category for item in history).most_common(1)
    names = ", ".join(item.name for item in recommended[:2])
    if top:
        return f"Because you have been browsing {top[0][0].lower()}, you might like {names}."
    return f"These are popular right now: {names}."


_MODEL = re.compile(r"\bmodel (\d+)", re.IGNORECASE)
_PRICE = re.compile(r"\$\s?(\d+(?:[.,]\d+)?)")


def grounding_problem(text: str, recommended: list[ItemInfo]) -> str | None:
    """Why the text must not be shown, or None when it passes."""
    if not text or len(text) > MAX_CHARS:
        return "empty or too long"
    allowed_prices = {f"{item.price:.2f}" for item in recommended[:MAX_LISTED]}
    allowed_prices |= {f"{item.price:.0f}" for item in recommended[:MAX_LISTED]}
    for match in _PRICE.findall(text):
        if match.replace(",", ".") not in allowed_prices:
            return f"unlisted price ${match}"
    if re.search(r"item_\d+", text):
        return "raw item id leaked"
    listed = {str(int(item.item_id.rsplit("_", 1)[1])) for item in recommended[:MAX_LISTED]}
    for number in _MODEL.findall(text):
        if number not in listed:
            return f"unlisted product model {number}"
    return None


class Explainer:
    def __init__(self, gateway: Gateway, max_tokens: int = 400) -> None:
        self._gateway = gateway
        self._max_tokens = max_tokens

    def explain(self, history: list[ItemInfo], recommended: list[ItemInfo]) -> Explanation:
        fallback = template_explanation(history, recommended)
        if not recommended:
            return Explanation(fallback, "template", "nothing to explain")
        try:
            result = self._gateway.complete(
                build_messages(history, recommended), max_tokens=self._max_tokens
            )
        except LLMError as error:
            log.info("falling back to the template: %s", error)
            return Explanation(fallback, "template", f"llm unavailable: {error}")

        problem = grounding_problem(result.text, recommended)
        if problem:
            log.warning("discarding an ungrounded explanation (%s)", problem)
            return Explanation(
                fallback,
                "template",
                f"rejected: {problem}",
                result.cost_micro_usd,
                raw_text=result.text,
            )
        return Explanation(
            result.text,
            "llm",
            cost_micro_usd=result.cost_micro_usd,
            cached=result.cached,
            raw_text=result.text,
        )
