"""One door to the LLM provider: retries, a response cache and a daily cost cap.

Everything that talks to a model goes through ``LLMGateway.complete``. Callers never see
HTTP, retries or pricing; they get a ``Completion`` or one of two errors:

* ``BudgetExceeded``: today's spend reached the cap, no request was sent.
* ``LLMError``: the provider failed after the retries (callers fall back to a template).

Providers sit behind the small ``ChatClient`` interface (``model`` + ``chat()``), so
switching vendors means writing one client class and registering it in ``PROVIDERS``;
the gateway, cache, budget and everything built on top stay untouched. The bundled client
speaks the OpenAI-compatible protocol (Groq, OpenAI, Together, OpenRouter, Ollama, vLLM),
needs no vendor SDK, and takes the HTTP call as a parameter so tests need no network.

Money is counted in micro-dollars (integers) so Redis ``INCRBY`` stays exact. A price of
$0.05 per million tokens is 0.05 micro-dollars per token.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

log = logging.getLogger("recsys.llm")

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class LLMError(Exception):
    """The provider could not produce an answer."""


class BudgetExceeded(LLMError):
    """The daily cost cap is reached; no request was sent."""


@dataclass(frozen=True)
class Price:
    """USD per million tokens (equals micro-dollars per token)."""

    input: float
    output: float


# Public list prices of the Groq models with self-serve pricing; verify at
# console.groq.com/docs/models and override with LLM_PRICE_INPUT_PER_M / _OUTPUT_PER_M.
# (llama-3.x are listed "contact sales" there, and an account without that access gets
# HTTP 404 model_not_found, so they are not used as defaults.)
GROQ_PRICES = {
    "openai/gpt-oss-20b": Price(0.075, 0.30),
    "openai/gpt-oss-120b": Price(0.15, 0.60),
}


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_micro_usd: int
    cached: bool
    latency_ms: float


@dataclass(frozen=True)
class ChatResult:
    """What every provider client returns, whatever the vendor's wire format is."""

    text: str
    prompt_tokens: int
    completion_tokens: int


class ChatClient(Protocol):
    model: str

    def chat(self, messages: list[dict], max_tokens: int, temperature: float) -> ChatResult: ...


class Response(Protocol):
    status_code: int
    headers: Any

    def json(self) -> Any: ...


Post = Callable[..., Response]


def httpx_post(url: str, *, headers: dict, json: dict, timeout: float) -> Response:
    """Default transport. Network failures become ``ConnectionError`` (retryable)."""
    import httpx

    try:
        return httpx.post(url, headers=headers, json=json, timeout=timeout)
    except httpx.TransportError as error:
        raise ConnectionError(str(error)) from error


class OpenAICompatibleClient:
    """``/chat/completions`` over HTTP with bounded retries and exponential backoff."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout_s: float = 15.0,
        max_retries: int = 2,
        extra: dict | None = None,
        post: Post = httpx_post,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise ValueError("the LLM API key is empty")
        self.api_key = api_key
        self.model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._timeout = timeout_s
        self._max_retries = max_retries
        self._post = post
        self._sleep = sleep
        self._extra = dict(extra or {})  # provider-specific request fields
        self.retries = 0

    def chat(self, messages: list[dict], max_tokens: int, temperature: float) -> ChatResult:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **self._extra,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last = "no attempt made"
        for attempt in range(self._max_retries + 1):
            if attempt:
                self.retries += 1
            wait = min(8.0, 0.5 * 2**attempt)
            try:
                response = self._post(self._url, headers=headers, json=body, timeout=self._timeout)
            except (TimeoutError, ConnectionError, OSError) as error:
                last = f"{type(error).__name__}: {error}"
            else:
                if response.status_code == 200:
                    return _parse(response.json())
                last = f"HTTP {response.status_code}{_detail(response)}"
                if response.status_code not in RETRYABLE_STATUS:
                    raise LLMError(last)  # bad key, bad request: retrying cannot help
                wait = _retry_after(response, wait)
            if attempt < self._max_retries:
                log.warning("LLM call failed (%s), retrying in %.1fs", last, wait)
                self._sleep(wait)
        raise LLMError(f"gave up after {self._max_retries + 1} attempts: {last}")


def _detail(response: Response) -> str:
    """The provider's own error message (e.g. "model not found"), shortened."""
    text = str(getattr(response, "text", "") or "").strip().replace("\n", " ")
    return f": {text[:300]}" if text else ""


def _parse(data) -> ChatResult:
    try:
        text = (data["choices"][0]["message"]["content"] or "").strip()
        usage = data.get("usage") or {}
        if not text:
            raise LLMError(
                "empty completion (a reasoning model may have spent max_tokens on thinking)"
            )
        return ChatResult(
            text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        )
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as error:
        raise LLMError(f"malformed provider response: {error!r}") from error


def _retry_after(response: Response, default: float) -> float:
    try:
        return min(30.0, max(0.0, float(response.headers.get("retry-after", default))))
    except (TypeError, ValueError):
        return default


class CostTracker:
    """Today's spend in Redis (one counter per UTC day), compared with a cap."""

    def __init__(
        self,
        redis,
        daily_cap_usd: float,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._redis = redis
        self._cap_micro = int(daily_cap_usd * 1_000_000)
        self._clock = clock

    def _key(self) -> str:
        return f"llm:cost:{self._clock():%Y-%m-%d}"

    def spent_micro(self) -> int:
        return int(self._redis.get(self._key()) or 0)

    def check(self) -> None:
        spent = self.spent_micro()
        if spent >= self._cap_micro:
            raise BudgetExceeded(
                f"daily LLM budget reached: ${spent / 1e6:.4f} of ${self._cap_micro / 1e6:.2f}"
            )

    def add(self, micro_usd: int) -> None:
        if micro_usd > 0:
            self._redis.incrby(self._key(), micro_usd)
            self._redis.expire(self._key(), 3 * 86400)


@dataclass
class GatewayStats:
    calls: int = 0
    cache_hits: int = 0
    errors: int = 0
    budget_blocks: int = 0
    spent_micro_usd: int = field(default=0)


class LLMGateway:
    def __init__(
        self,
        client: ChatClient,
        redis,
        tracker: CostTracker,
        cache_ttl_s: int = 86400,
        prices: dict[str, Price] | None = None,
    ) -> None:
        self._client = client
        self._redis = redis
        self._tracker = tracker
        self._ttl = cache_ttl_s
        self._prices = GROQ_PRICES if prices is None else prices
        self.stats = GatewayStats()

    def _cache_key(self, messages, max_tokens, temperature) -> str:
        raw = json.dumps([self._client.model, messages, max_tokens, temperature], sort_keys=True)
        return "llm:cache:" + hashlib.sha256(raw.encode()).hexdigest()

    def _cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> int:
        price = self._prices.get(model)
        if price is None:
            log.warning("no price for model %s; counting it as free", model)
            return 0
        micro = prompt_tokens * price.input + completion_tokens * price.output
        return max(1, round(micro)) if prompt_tokens + completion_tokens else 0

    def complete(
        self,
        messages: list[dict],
        max_tokens: int = 150,
        temperature: float = 0.2,
        use_cache: bool = True,
    ) -> Completion:
        started = time.perf_counter()
        key = self._cache_key(messages, max_tokens, temperature)

        if use_cache:
            hit = self._cache_get(key)
            if hit:
                self.stats.cache_hits += 1
                data = json.loads(hit)
                return Completion(
                    **{
                        **data,
                        "cached": True,
                        "cost_micro_usd": 0,
                        "latency_ms": (time.perf_counter() - started) * 1000,
                    }
                )

        try:
            self._tracker.check()
        except BudgetExceeded:
            self.stats.budget_blocks += 1
            raise
        except Exception as error:  # cannot read the spend: refuse rather than risk the cap
            self.stats.errors += 1
            raise LLMError(f"cost counter unavailable: {type(error).__name__}") from error

        try:
            answer = self._client.chat(messages, max_tokens, temperature)
        except LLMError:
            self.stats.errors += 1
            raise
        self.stats.calls += 1
        text, prompt_tokens, completion_tokens = (
            answer.text,
            answer.prompt_tokens,
            answer.completion_tokens,
        )

        cost = self._cost(self._client.model, prompt_tokens, completion_tokens)
        try:
            self._tracker.add(cost)
        except Exception:  # the answer is already paid for: still return it
            log.warning("could not record LLM cost of %d micro-dollars", cost, exc_info=True)
        self.stats.spent_micro_usd += cost

        result = Completion(
            text=text,
            model=self._client.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_micro_usd=cost,
            cached=False,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        if use_cache and text:
            self._cache_set(
                key,
                json.dumps(
                    {
                        "text": text,
                        "model": result.model,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "cost_micro_usd": cost,
                        "cached": False,
                        "latency_ms": result.latency_ms,
                    }
                ),
            )
        return result

    def _cache_get(self, key: str):
        try:
            return self._redis.get(key)
        except Exception:  # a cache outage must not stop answers
            log.warning("LLM cache read failed", exc_info=True)
            return None

    def _cache_set(self, key: str, value: str) -> None:
        try:
            self._redis.set(key, value, ex=self._ttl)
        except Exception:
            log.warning("LLM cache write failed", exc_info=True)


@dataclass(frozen=True)
class ProviderSpec:
    base_url: str
    default_model: str
    prices: dict[str, Price]


PROVIDERS = {
    "groq": ProviderSpec("https://api.groq.com/openai/v1", "openai/gpt-oss-20b", GROQ_PRICES),
    # Any OpenAI-compatible server: set LLM_BASE_URL and LLM_MODEL (and the prices, or the
    # cost cap will count the calls as free).
    "openai-compatible": ProviderSpec("", "", {}),
}


def build_client(settings, post: Post = httpx_post) -> tuple[ChatClient, dict[str, Price]]:
    """Create the configured provider client and its price table."""
    spec = PROVIDERS.get(settings.llm_provider)
    if spec is None:
        raise ValueError(
            f"unknown LLM_PROVIDER {settings.llm_provider!r}; choose from {sorted(PROVIDERS)}"
        )
    base_url = settings.llm_base_url or spec.base_url
    model = settings.llm_model or spec.default_model
    if not base_url or not model:
        raise ValueError("this provider needs LLM_BASE_URL and LLM_MODEL")
    api_key = settings.llm_api_key or (
        settings.groq_api_key if settings.llm_provider == "groq" else ""
    )
    prices = dict(spec.prices)
    if settings.llm_price_input_per_m or settings.llm_price_output_per_m:
        prices[model] = Price(settings.llm_price_input_per_m, settings.llm_price_output_per_m)
    extra = {}
    effort = settings.llm_reasoning_effort or ("low" if "gpt-oss" in model else "")
    if effort:  # reasoning models spend max_tokens on thinking; keep it short
        extra["reasoning_effort"] = effort
    client = OpenAICompatibleClient(
        api_key, model, base_url, timeout_s=settings.llm_timeout_s, extra=extra, post=post
    )
    return client, prices


def build_gateway(settings, redis, post: Post = httpx_post) -> LLMGateway:
    client, prices = build_client(settings, post)
    tracker = CostTracker(redis, settings.llm_daily_budget_usd)
    return LLMGateway(client, redis, tracker, cache_ttl_s=settings.llm_cache_ttl_s, prices=prices)
