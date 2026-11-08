"""Check the Groq key and the gateway against the real API (costs a fraction of a cent).

Run: ``python -m recsys.llm.smoke``   (LLM_API_KEY or GROQ_API_KEY set; Redis up)
The second, identical request must come from the cache and cost nothing.
"""

from __future__ import annotations

import redis as redis_lib

from recsys.config import Settings
from recsys.llm.gateway import LLMError, build_gateway


def main() -> None:
    settings = Settings.from_env()
    store = redis_lib.Redis.from_url(settings.redis_url)
    try:
        gateway = build_gateway(settings, store)
    except ValueError as error:  # missing key, unknown provider, ...
        raise SystemExit(f"{error} (keys go in the environment, never in git)") from error
    messages = [
        {"role": "system", "content": "You write one short sentence."},
        {"role": "user", "content": "Why might a runner like a sports watch?"},
    ]
    try:
        for label in ("first call", "second call"):
            result = gateway.complete(messages, max_tokens=400)
            print(
                f"{label}: cached={result.cached} model={result.model} "
                f"tokens={result.prompt_tokens}+{result.completion_tokens} "
                f"cost=${result.cost_micro_usd / 1e6:.6f} latency={result.latency_ms:.0f} ms"
            )
            print(f"  {result.text}")
    except LLMError as error:
        raise SystemExit(f"LLM call failed: {error}") from error


if __name__ == "__main__":
    main()
