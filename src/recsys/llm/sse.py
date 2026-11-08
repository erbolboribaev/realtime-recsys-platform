"""Server-Sent Events for a result that is produced elsewhere (the explanation worker).

``sse_events`` is a plain generator so it can be tested without a web framework; the API
wraps it in a streaming response. The browser keeps one HTTP connection open and receives:

    event: pending       once, as soon as the request is known
    : waiting            comment lines while the worker is busy (keeps proxies from closing)
    event: explanation   the finished explanation (then the stream ends)
    event: timeout       nothing arrived in time and no fallback exists (the client may retry)

Note this delivers the finished text as soon as it exists; it does not stream model tokens.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator


def format_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def sse_events(
    load: Callable[[], dict | None],
    timeout_s: float = 15.0,
    interval_s: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    fallback: Callable[[], dict | None] | None = None,
) -> Iterator[str]:
    deadline = clock() + timeout_s
    yield "retry: 3000\n\n"
    first = True
    while True:
        result = load()
        if result is not None:
            yield format_event("explanation", result)
            return
        if clock() >= deadline:
            substitute = fallback() if fallback else None
            if substitute is not None:  # same event name, so clients need no special case
                yield format_event("explanation", substitute)
                return
            yield format_event("timeout", {"detail": f"no explanation within {timeout_s:g}s"})
            return
        yield format_event("pending", {}) if first else ": waiting\n\n"
        first = False
        sleep(interval_s)
