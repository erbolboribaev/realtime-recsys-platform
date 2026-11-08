import numpy as np

from recsys.events.simulator import EventSimulator, SimulatorConfig
from recsys.llm.catalog import CATEGORY_NAMES, SINGULAR, ItemInfo, SimulatedCatalog
from recsys.llm.explain import (
    MAX_CHARS,
    Explainer,
    build_messages,
    grounding_problem,
    template_explanation,
)
from recsys.llm.gateway import BudgetExceeded, Completion, LLMError


def info(n, category="Running shoes", price=59.9):
    return ItemInfo(f"item_{n:05d}", f"{category[:-1]} model {n}", category, price)


HISTORY = [info(1), info(2), info(3, "Sports watches", 120.0)]
RECS = [info(10, "Running shoes", 59.9), info(11, "Sports watches", 129.5)]


class FakeGateway:
    def __init__(self, text=None, error=None):
        self.text, self.error, self.calls = text, error, []

    def complete(self, messages, max_tokens=150, **kwargs):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return Completion(self.text, "m", 10, 5, 7, False, 1.0)


def test_catalog_describes_items_from_the_simulated_catalog():
    sim = EventSimulator(SimulatorConfig(n_items=100, n_categories=20, seed=1))
    catalog = SimulatedCatalog(sim.catalog)
    item = catalog.describe(sim.catalog.item_ids[7])
    assert item.category in CATEGORY_NAMES and item.price == float(sim.catalog.prices[7])
    assert item.name.endswith("model 7")
    for index, singular in enumerate(CATEGORY_NAMES):
        assert not singular.startswith("Sunglasse ") and SINGULAR[index]
    assert catalog.describe("nope") is None


def test_prompt_contains_only_known_facts_and_the_grounding_rules():
    system, user = build_messages(HISTORY, RECS)
    assert "ONLY the facts" in system["content"]
    assert "Running shoes (x2)" in user["content"] and "Sports watches (x1)" in user["content"]
    assert "Running shoe model 10" in user["content"] and "$129.50" in user["content"]


def test_prompt_handles_a_user_without_history():
    assert "nothing yet" in build_messages([], RECS)[1]["content"]


def test_a_grounded_llm_answer_is_returned_as_is():
    gateway = FakeGateway("Since you browse running shoes, Running shoe model 10 fits.")
    result = Explainer(gateway).explain(HISTORY, RECS)
    assert result.source == "llm" and "model 10" in result.text and result.cost_micro_usd == 7


def test_the_template_is_used_when_the_llm_fails_or_the_budget_is_spent():
    for error in (LLMError("down"), BudgetExceeded("cap")):
        result = Explainer(FakeGateway(error=error)).explain(HISTORY, RECS)
        assert result.source == "template" and "running shoes" in result.text.lower()
        assert result.reason.startswith("llm unavailable")


def test_an_answer_with_an_invented_price_is_rejected():
    result = Explainer(FakeGateway("Great deal at $9.99 today!")).explain(HISTORY, RECS)
    assert result.source == "template" and "unlisted price" in result.reason
    assert result.raw_text == "Great deal at $9.99 today!"  # kept so evals can judge it
    assert result.cost_micro_usd == 7  # the call was paid for even though the text was dropped


def test_listed_prices_and_rounded_prices_pass_the_check():
    assert grounding_problem("It costs $59.90.", RECS) is None
    assert grounding_problem("About $60.", RECS) is None
    assert grounding_problem("About $61.", RECS) is not None


def test_leaked_ids_empty_and_overlong_answers_are_rejected():
    assert grounding_problem("Try item_00010 now", RECS)
    assert grounding_problem("", RECS)
    assert grounding_problem("x" * (MAX_CHARS + 1), RECS)


def test_nothing_to_explain_does_not_call_the_model():
    gateway = FakeGateway("unused")
    result = Explainer(gateway).explain(HISTORY, [])
    assert result.source == "template" and gateway.calls == []


def test_template_mentions_the_top_category_when_there_is_history():
    assert "running shoes" in template_explanation(HISTORY, RECS)
    assert "popular" in template_explanation([], RECS)
    assert np.isfinite(len(template_explanation([], [])))


def test_a_product_model_that_was_not_recommended_is_rejected():
    assert grounding_problem("You may like Running shoe model 10.", RECS) is None
    assert "model 999" in grounding_problem("Try Headphone model 999 too.", RECS)


def test_sunglasses_and_watches_get_correct_singular_names():
    assert "Pair of sunglasses" in SINGULAR and "Sports watch" in SINGULAR
    assert all(not name.endswith("e model") for name in SINGULAR)


def test_the_prompt_forbids_judging_products_and_limits_how_many_are_named():
    system = build_messages(HISTORY, RECS)[0]["content"]
    assert "at most three products" in system
    assert "great" in system and "never" in system.lower()


def test_the_model_only_sees_three_products_and_cannot_name_others():
    many = [info(n) for n in range(20, 25)]
    user = build_messages(HISTORY, many)[1]["content"]
    assert "model 22" in user and "model 23" not in user and "model 24" not in user
    assert grounding_problem("Look at Running shoe model 22.", many) is None
    assert "model 23" in grounding_problem("Also Running shoe model 23.", many)
