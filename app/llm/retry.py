"""Exponential backoff for transient LLM failures."""

import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable

from app.llm.errors import TransientLLMError

MAX_RETRIES = 3
BASE_DELAY_SECONDS = 0.5
MAX_DELAY_SECONDS = 8.0

Sleep = Callable[[float], Awaitable[None]]


def backoff_delay(attempt: int) -> float:
    """Delay before retry number `attempt` (0-based): 0.5s, 1s, 2s... plus jitter, capped."""
    delay = BASE_DELAY_SECONDS * (2**attempt)
    return min(delay + random.uniform(0, BASE_DELAY_SECONDS), MAX_DELAY_SECONDS)


async def with_retries[T](
    call: Callable[[], Awaitable[T]],
    *,
    max_retries: int = MAX_RETRIES,
    sleep: Sleep = asyncio.sleep,
) -> T:
    """Run `call`, retrying only `TransientLLMError`, at most `max_retries` extra times.

    Every other exception (including `QuotaExceededError`) propagates immediately.
    """
    attempt = 0
    while True:
        try:
            return await call()
        except TransientLLMError:
            if attempt >= max_retries:
                raise
            await sleep(backoff_delay(attempt))
            attempt += 1


async def stream_with_retries(
    open_stream: Callable[[], AsyncIterator[str]],
    *,
    max_retries: int = MAX_RETRIES,
    sleep: Sleep = asyncio.sleep,
) -> AsyncIterator[str]:
    """Retry a stream only while nothing has been yielded yet.

    Once a fragment reached the caller, restarting would duplicate text, so a
    later transient failure propagates.
    """
    attempt = 0
    while True:
        yielded = False
        try:
            async for fragment in open_stream():
                yielded = True
                yield fragment
            return
        except TransientLLMError:
            if yielded or attempt >= max_retries:
                raise
            await sleep(backoff_delay(attempt))
            attempt += 1
