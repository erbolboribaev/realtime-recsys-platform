import re

from recsys.events.simulator import EventSimulator, SimulatorConfig
from recsys.llm.catalog import SimulatedCatalog
from recsys.llm.evals import (
    UNSUPPORTED_CLAIM,
    Report,
    build_cases,
    run_evals,
    violations,
)
from recsys.llm.explain import Explainer
from recsys.llm.gateway import Completion, LLMError


def catalog():
    return SimulatedCatalog(
        EventSimulator(SimulatorConfig(n_items=400, n_categories=10, seed=2)).catalog
    )


class Model:
    """A fake model: ``answer(messages) -> text``."""

    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    def complete(self, messages, max_tokens=150, **kwargs):
        self.calls += 1
        return Completion(self.answer(messages), "m", 10, 5, 10, False, 1.0)


def first_product(messages):
    return re.search(r"- (.+?) \| category", messages[1]["content"]).group(1)


def good(messages):
    return f"Since you have been browsing similar items, take a look at {first_product(messages)}."


def test_cases_are_deterministic_and_well_formed():
    a, b = build_cases(catalog(), 12, seed=1), build_cases(catalog(), 12, seed=1)
    assert [c.recommended[0].item_id for c in a] == [c.recommended[0].item_id for c in b]
    assert len(a) == 12
    for case in a:
        assert len(case.history) == 4 and len(case.recommended) == 5
        assert len({i.item_id for i in case.history + case.recommended[:3]}) >= 6


def test_a_well_behaved_model_passes_every_metric():
    report = run_evals(Explainer(Model(good)), build_cases(catalog(), 10))
    assert (report.accepted, report.on_topic, report.clean, report.focused) == (1, 1, 1, 1)
    assert report.fallback == 0 and report.avg_cost_micro_usd == 10
    assert violations(report) == []


def test_unsupported_claims_are_counted_even_though_the_guardrail_would_pass_them():
    model = Model(lambda m: f"The stylish {first_product(m)} has great sound.")
    report = run_evals(Explainer(model), build_cases(catalog(), 10))
    assert report.accepted == 1.0  # the guardrail alone does not catch quality claims
    assert report.clean == 0.0
    assert any("clean" in v for v in violations(report))
    assert any("unsupported claim 'stylish'" in f for f in report.failures)


def test_invented_prices_are_rejected_by_the_guardrail_and_reported():
    model = Model(lambda m: f"Grab {first_product(m)} for only $0.99!")
    report = run_evals(Explainer(model), build_cases(catalog(), 10))
    assert report.accepted == 0.0 and report.fallback == 1.0
    assert any("accepted" in v for v in violations(report))


def test_off_topic_answers_are_counted():
    report = run_evals(Explainer(Model(lambda m: "Have a nice day.")), build_cases(catalog(), 8))
    assert report.on_topic == 0.0


def test_a_model_outage_means_fallbacks_and_no_scores_to_judge():
    class Down:
        def complete(self, *args, **kwargs):
            raise LLMError("down")

    report = run_evals(Explainer(Down()), build_cases(catalog(), 5))
    assert report.fallback == 1.0 and report.accepted == 0.0
    assert "no answer x5" in report.failures[0] and "down" in report.failures[0]


def test_violations_flags_an_empty_run_and_accepts_custom_minimums():
    assert violations(Report(0, 0, 0, 0, 0, 0, 0)) != []
    ok = Report(5, 0.8, 0.8, 0.8, 0.8, 0.0, 0)
    assert violations(ok, {"accepted": 0.75}) == []
    assert violations(ok)  # default minimum is 0.90


def test_claim_pattern_catches_common_marketing_words_and_ignores_plain_text():
    assert UNSUPPORTED_CLAIM.search("These headphones have amazing sound")
    assert UNSUPPORTED_CLAIM.search("A real bargain")
    assert not UNSUPPORTED_CLAIM.search("Since you browse headphones, try Headphone model 5.")


def test_answers_naming_more_than_three_products_are_flagged():
    def many(messages):
        names = re.findall(r"- (.+?) \| category", messages[1]["content"])
        return "Look at " + ", ".join(names) + ", Keyboard model 9001 and Yoga mat model 9002."

    report = run_evals(Explainer(Model(many)), build_cases(catalog(), 6))
    assert report.focused == 0.0
    assert report.accepted == 0.0  # the guardrail also drops products the model never saw
    assert any("too many products" in f for f in report.failures)
    assert any("focused" in v for v in violations(report))
