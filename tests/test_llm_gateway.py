from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from recsys.config import Settings
from recsys.llm.gateway import (
    BudgetExceeded,
    ChatResult,
    CostTracker,
    LLMError,
    LLMGateway,
    OpenAICompatibleClient,
    Price,
    build_client,
    build_gateway,
)


class FakeRedis:
    def __init__(self):
        self.kv = {}

    def get(self, key):
        return self.kv.get(key)

    def set(self, key, value, ex=None):
        self.kv[key] = value

    def incrby(self, key, amount):
        self.kv[key] = int(self.kv.get(key, 0)) + amount

    def expire(self, key, seconds):
        pass


def reply(text="ok", prompt=100, completion=20, status=200, headers=None):
    body = {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }
    return SimpleNamespace(status_code=status, headers=headers or {}, json=lambda: body)


class Script:
    """A fake ``post``: plays back responses (or raises exceptions) in order."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.calls = []

    def __call__(self, url, *, headers, json, timeout):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def make(*steps, cap_usd=1.0, prices=None):
    redis = FakeRedis()
    post = Script(*steps)
    sleeps = []
    client = OpenAICompatibleClient(
        "key", "test-model", "https://example.test/v1", post=post, sleep=sleeps.append
    )
    now = datetime(2026, 10, 29, tzinfo=UTC)
    tracker = CostTracker(redis, cap_usd, clock=lambda: now)
    gateway = LLMGateway(client, redis, tracker, prices=prices or {client.model: Price(1.0, 2.0)})
    return gateway, post, redis, sleeps


MESSAGES = [{"role": "user", "content": "hi"}]


def test_completion_returns_text_tokens_and_cost():
    gateway, post, _, _ = make(reply("  hello  ", prompt=100, completion=20))
    result = gateway.complete(MESSAGES)
    assert result.text == "hello"
    assert (result.prompt_tokens, result.completion_tokens) == (100, 20)
    assert result.cost_micro_usd == 140  # 100 * 1.0 + 20 * 2.0 micro-dollars
    assert not result.cached
    sent = post.calls[0]
    assert sent["url"].endswith("/chat/completions")
    assert sent["headers"]["Authorization"] == "Bearer key"
    assert sent["json"]["messages"] == MESSAGES


def test_identical_requests_are_served_from_cache_for_free():
    gateway, post, _, _ = make(reply("first"))
    first = gateway.complete(MESSAGES)
    second = gateway.complete(MESSAGES)
    assert second.text == "first" and second.cached and second.cost_micro_usd == 0
    assert len(post.calls) == 1
    assert gateway.stats.cache_hits == 1 and first.cost_micro_usd > 0


def test_different_messages_or_parameters_do_not_share_cache_entries():
    gateway, post, _, _ = make(reply("a"), reply("b"), reply("c"))
    gateway.complete(MESSAGES)
    gateway.complete([{"role": "user", "content": "other"}])
    gateway.complete(MESSAGES, temperature=0.9)
    assert len(post.calls) == 3


def test_cache_can_be_bypassed():
    gateway, post, _, _ = make(reply("a"), reply("b"))
    gateway.complete(MESSAGES, use_cache=False)
    assert gateway.complete(MESSAGES, use_cache=False).text == "b"
    assert len(post.calls) == 2


def test_spend_is_accumulated_and_blocks_once_the_cap_is_reached():
    # cap = 200 micro-dollars; each call costs 140
    gateway, post, _, _ = make(reply(), reply(), cap_usd=0.0002)
    gateway.complete([{"role": "user", "content": "1"}])  # spent 140 < 200: still allowed
    gateway.complete([{"role": "user", "content": "2"}])  # spent 280 >= 200 afterwards
    with pytest.raises(BudgetExceeded):
        gateway.complete([{"role": "user", "content": "3"}])
    assert len(post.calls) == 2  # the blocked request never reached the provider
    assert gateway.stats.budget_blocks == 1


def test_cached_answers_are_still_served_after_the_budget_is_spent():
    gateway, _, _, _ = make(reply("cached answer"), cap_usd=0.0001)
    gateway.complete(MESSAGES)  # spends 140 > cap of 100
    assert gateway.complete(MESSAGES).text == "cached answer"


def test_the_cost_counter_resets_on_a_new_day():
    redis = FakeRedis()
    day = {"now": datetime(2026, 10, 29, tzinfo=UTC)}
    tracker = CostTracker(redis, 0.0001, clock=lambda: day["now"])
    tracker.add(500)
    with pytest.raises(BudgetExceeded):
        tracker.check()
    day["now"] = datetime(2026, 10, 30, tzinfo=UTC)
    tracker.check()


def test_transient_errors_are_retried_with_backoff_then_succeed():
    gateway, post, _, sleeps = make(reply(status=503), ConnectionError("reset"), reply("recovered"))
    assert gateway.complete(MESSAGES).text == "recovered"
    assert len(post.calls) == 3 and sleeps == [0.5, 1.0]


def test_rate_limit_honours_retry_after():
    gateway, _, _, sleeps = make(reply(status=429, headers={"retry-after": "3"}), reply("ok"))
    gateway.complete(MESSAGES)
    assert sleeps == [3.0]


def test_gives_up_after_the_retry_limit_and_does_not_charge():
    gateway, post, redis, _ = make(reply(status=500), reply(status=500), reply(status=500))
    with pytest.raises(LLMError):
        gateway.complete(MESSAGES)
    assert len(post.calls) == 3 and gateway.stats.errors == 1
    assert not any(k.startswith("llm:cost") for k in redis.kv)


def test_client_errors_are_not_retried():
    gateway, post, _, sleeps = make(reply(status=401))
    with pytest.raises(LLMError, match="401"):
        gateway.complete(MESSAGES)
    assert len(post.calls) == 1 and sleeps == []


def test_malformed_provider_response_is_an_llm_error():
    broken = SimpleNamespace(status_code=200, headers={}, json=lambda: {"choices": []})
    gateway, _, _, _ = make(broken)
    with pytest.raises(LLMError, match="malformed"):
        gateway.complete(MESSAGES)


def test_unknown_model_price_counts_as_free_instead_of_crashing():
    gateway, _, _, _ = make(reply(), prices={"some-other-model": Price(1, 1)})
    assert gateway.complete(MESSAGES).cost_micro_usd == 0


def test_client_requires_an_api_key():
    with pytest.raises(ValueError):
        OpenAICompatibleClient("", "m", "https://example.test/v1")


def test_build_gateway_uses_provider_defaults_and_the_groq_key_fallback():
    settings = Settings(groq_api_key="k")
    post = Script(reply())
    gateway = build_gateway(settings, FakeRedis(), post=post)
    result = gateway.complete(MESSAGES)
    assert result.model == "openai/gpt-oss-20b"
    assert post.calls[0]["json"]["reasoning_effort"] == "low"
    assert post.calls[0]["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert post.calls[0]["headers"]["Authorization"] == "Bearer k"
    assert result.cost_micro_usd > 0  # the built-in Groq price table applies


def test_any_openai_compatible_server_works_with_url_model_and_prices_from_settings():
    settings = Settings(
        llm_provider="openai-compatible",
        llm_api_key="k",
        llm_base_url="http://localhost:11434/v1",
        llm_model="my-model",
        llm_price_input_per_m=1.0,
        llm_price_output_per_m=2.0,
    )
    post = Script(reply(prompt=10, completion=5))
    result = build_gateway(settings, FakeRedis(), post=post).complete(MESSAGES)
    assert post.calls[0]["url"] == "http://localhost:11434/v1/chat/completions"
    assert post.calls[0]["json"]["model"] == "my-model"
    assert result.cost_micro_usd == 20  # 10 * 1.0 + 5 * 2.0


def test_misconfiguration_fails_fast_with_a_clear_message():
    with pytest.raises(ValueError, match="unknown LLM_PROVIDER"):
        build_client(Settings(llm_provider="nope", llm_api_key="k"))
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        build_client(Settings(llm_provider="openai-compatible", llm_api_key="k"))
    with pytest.raises(ValueError, match="key"):
        build_client(Settings(llm_provider="groq"))


class OtherVendorClient:
    """A provider with a different wire format only has to return a ChatResult."""

    model = "vendor-x"

    def __init__(self):
        self.calls = 0

    def chat(self, messages, max_tokens, temperature):
        self.calls += 1
        return ChatResult(text="from another vendor", prompt_tokens=50, completion_tokens=10)


def test_the_gateway_works_with_any_client_that_implements_the_interface():
    redis = FakeRedis()
    client = OtherVendorClient()
    gateway = LLMGateway(
        client,
        redis,
        CostTracker(redis, 1.0, clock=lambda: datetime(2026, 10, 29, tzinfo=UTC)),
        prices={"vendor-x": Price(2.0, 4.0)},
    )
    first = gateway.complete(MESSAGES)
    assert first.text == "from another vendor" and first.cost_micro_usd == 140
    assert gateway.complete(MESSAGES).cached and client.calls == 1


def test_settings_read_llm_variables_from_the_environment():
    settings = Settings.from_env(
        {"LLM_PROVIDER": "openai-compatible", "LLM_DAILY_BUDGET_USD": "0.25", "LLM_API_KEY": "abc"}
    )
    assert settings.llm_provider == "openai-compatible"
    assert settings.llm_api_key == "abc" and settings.llm_daily_budget_usd == 0.25


def test_provider_error_message_is_included_so_failures_can_be_diagnosed():
    body = '{"error":{"message":"The model `x` does not exist","code":"model_not_found"}}'
    gateway, _, _, _ = make(SimpleNamespace(status_code=404, headers={}, text=body))
    with pytest.raises(LLMError, match="model_not_found"):
        gateway.complete(MESSAGES)


class BrokenRedis(FakeRedis):
    def __init__(self, fail_on):
        super().__init__()
        self.fail_on = fail_on

    def get(self, key):
        if "get" in self.fail_on and key.startswith("llm:cache"):
            raise ConnectionError("redis down")
        return super().get(key)

    def set(self, key, value, ex=None):
        if "set" in self.fail_on:
            raise ConnectionError("redis down")
        super().set(key, value, ex)

    def incrby(self, key, amount):
        if "incrby" in self.fail_on:
            raise ConnectionError("redis down")
        super().incrby(key, amount)


def gateway_on(redis, *steps):
    client = OpenAICompatibleClient(
        "key", "test-model", "https://example.test/v1", post=Script(*steps), sleep=lambda _: None
    )
    tracker = CostTracker(redis, 1.0, clock=lambda: datetime(2026, 10, 29, tzinfo=UTC))
    return LLMGateway(client, redis, tracker, prices={"test-model": Price(1.0, 2.0)})


def test_a_cache_outage_does_not_stop_answers():
    gateway = gateway_on(BrokenRedis({"get", "set"}), reply("still works"))
    assert gateway.complete(MESSAGES).text == "still works"


def test_a_cost_write_failure_still_returns_the_paid_answer():
    gateway = gateway_on(BrokenRedis({"incrby"}), reply("paid for"))
    assert gateway.complete(MESSAGES).text == "paid for"


def test_an_unreadable_cost_counter_refuses_the_call_with_an_llm_error():
    class NoCounter(FakeRedis):
        def get(self, key):
            if key.startswith("llm:cost"):
                raise ConnectionError("redis down")
            return super().get(key)

    gateway = gateway_on(NoCounter(), reply())
    with pytest.raises(LLMError, match="cost counter"):
        gateway.complete(MESSAGES)


def test_an_empty_completion_is_an_error_not_a_cached_blank_answer():
    empty = reply("   ")
    gateway, _, redis, _ = make(empty, empty, empty)
    with pytest.raises(LLMError, match="empty completion"):
        gateway.complete(MESSAGES)
    assert not any(k.startswith("llm:cache") for k in redis.kv)


def test_reasoning_effort_is_configurable_and_skipped_for_other_models():
    settings = Settings(llm_api_key="k", llm_model="some-chat-model", llm_provider="groq")
    post = Script(reply())
    build_gateway(settings, FakeRedis(), post=post).complete(MESSAGES)
    assert "reasoning_effort" not in post.calls[0]["json"]
    settings = Settings(llm_api_key="k", llm_reasoning_effort="high")
    post = Script(reply())
    build_gateway(settings, FakeRedis(), post=post).complete(MESSAGES)
    assert post.calls[0]["json"]["reasoning_effort"] == "high"
