import pytest

from app.ingest.pacing import MinuteRateLimiter, estimate_tokens


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


async def test_requests_beyond_rpm_wait_for_the_window() -> None:
    """Protects EMBED_RPM; rejects sending more requests per minute than allowed."""
    clock = FakeClock()
    limiter = MinuteRateLimiter(
        requests_per_minute=2, tokens_per_minute=1_000, clock=clock, sleep=clock.sleep
    )

    await limiter.acquire(1)
    await limiter.acquire(1)
    assert clock.sleeps == []
    await limiter.acquire(1)

    assert clock.now == pytest.approx(60.0)


async def test_tokens_beyond_tpm_wait_for_the_window() -> None:
    """Protects the tokens-per-minute budget; rejects bursts that exceed EMBED_TPM."""
    clock = FakeClock()
    limiter = MinuteRateLimiter(
        requests_per_minute=100, tokens_per_minute=1_000, clock=clock, sleep=clock.sleep
    )

    await limiter.acquire(700)
    await limiter.acquire(700)

    assert clock.now == pytest.approx(60.0)


async def test_oversized_request_is_capped_instead_of_blocking_forever() -> None:
    """Protects against deadlock; rejects waiting forever for a request above the TPM limit."""
    clock = FakeClock()
    limiter = MinuteRateLimiter(
        requests_per_minute=10, tokens_per_minute=100, clock=clock, sleep=clock.sleep
    )

    await limiter.acquire(10_000)

    assert clock.sleeps == []


def test_token_estimate_is_conservative() -> None:
    """Protects TPM headroom; rejects estimating fewer tokens than ~4 chars/token would give."""
    text = "Trust in the LORD with all your heart, and don't lean on your own understanding."

    assert estimate_tokens(text) >= len(text) / 4
    assert estimate_tokens("") == 1
