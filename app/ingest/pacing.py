"""Client-side pacing for the embeddings API (requests and tokens per minute)."""

import asyncio
import math
import time
from collections import deque
from collections.abc import Awaitable, Callable

WINDOW_SECONDS = 60.0

# English averages ~4 characters per token; 3 overestimates on purpose so the
# client stays under the provider's tokens-per-minute limit.
CHARS_PER_TOKEN_ESTIMATE = 3


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE))


class MinuteRateLimiter:
    """Sliding 60-second window limiting both request count and estimated tokens."""

    def __init__(
        self,
        *,
        requests_per_minute: int,
        tokens_per_minute: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if requests_per_minute <= 0 or tokens_per_minute <= 0:
            raise ValueError("rate limits must be positive")
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self._clock = clock
        self._sleep = sleep
        self._events: deque[tuple[float, int]] = deque()

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= WINDOW_SECONDS:
            self._events.popleft()

    async def acquire(self, tokens: int) -> None:
        """Wait until one request of `tokens` fits in the window, then record it."""
        tokens = min(tokens, self.tokens_per_minute)
        while True:
            now = self._clock()
            self._prune(now)
            used = sum(t for _, t in self._events)
            if (
                len(self._events) < self.requests_per_minute
                and used + tokens <= self.tokens_per_minute
            ):
                self._events.append((now, tokens))
                return
            oldest = self._events[0][0]
            await self._sleep(max(0.01, WINDOW_SECONDS - (now - oldest)))
