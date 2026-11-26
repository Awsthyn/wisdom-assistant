from collections.abc import AsyncIterator

import pytest

from app.llm.errors import QuotaExceededError, TransientLLMError
from app.llm.retry import stream_with_retries, with_retries


async def _no_sleep(_: float) -> None:
    return None


async def test_transient_errors_are_retried_three_times_then_raised() -> None:
    """Protects the retry cap; rejects retrying forever or giving up before 3 retries."""
    attempts = 0

    async def always_fails() -> str:
        nonlocal attempts
        attempts += 1
        raise TransientLLMError("503")

    with pytest.raises(TransientLLMError):
        await with_retries(always_fails, sleep=_no_sleep)

    assert attempts == 4  # 1 call + 3 retries


async def test_transient_error_then_success_returns_result() -> None:
    """Protects recovery after a blip; rejects surfacing an error that a retry fixed."""
    attempts = 0

    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TransientLLMError("timeout")
        return "ok"

    assert await with_retries(flaky, sleep=_no_sleep) == "ok"
    assert attempts == 3


async def test_quota_errors_are_not_retried() -> None:
    """Protects the daily quota; rejects burning extra requests after a 429."""
    attempts = 0

    async def quota() -> str:
        nonlocal attempts
        attempts += 1
        raise QuotaExceededError("429")

    with pytest.raises(QuotaExceededError):
        await with_retries(quota, sleep=_no_sleep)

    assert attempts == 1


async def test_stream_is_retried_when_it_fails_before_any_output() -> None:
    """Protects stream start-up retries; rejects failing a stream that never emitted text."""
    attempts = 0

    async def open_stream() -> AsyncIterator[str]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TransientLLMError("connect")
        yield "a"
        yield "b"

    fragments = [f async for f in stream_with_retries(open_stream, sleep=_no_sleep)]

    assert fragments == ["a", "b"]
    assert attempts == 2


async def test_stream_is_not_restarted_after_output() -> None:
    """Protects against duplicated text; rejects replaying a stream that already emitted text."""
    attempts = 0

    async def open_stream() -> AsyncIterator[str]:
        nonlocal attempts
        attempts += 1
        yield "partial"
        raise TransientLLMError("dropped")

    received: list[str] = []

    async def consume() -> None:
        async for fragment in stream_with_retries(open_stream, sleep=_no_sleep):
            received.append(fragment)

    with pytest.raises(TransientLLMError):
        await consume()

    assert received == ["partial"]
    assert attempts == 1
