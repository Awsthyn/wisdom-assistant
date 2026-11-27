"""Demo-guard logic that needs no database: keys, client identity, quota reset, examples."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.examples import EXAMPLE_QUESTIONS
from app.main import create_app
from app.services.guard import (
    RateLimitDecision,
    answer_cache_key,
    client_key,
    normalize_question,
    query_embedding_key,
    quota_status_at,
    window_start,
)
from tests.conftest import make_settings

PACIFIC = ZoneInfo("America/Los_Angeles")


def test_case_and_final_punctuation_share_the_cache_key() -> None:
    """Protects cache reuse; rejects "What is love?" and "what is love" getting different keys."""
    assert answer_cache_key("chat", "What is love?", "gemini", "m") == answer_cache_key(
        "chat", "what is love", "gemini", "m"
    )
    assert query_embedding_key("e", "What is love?") == query_embedding_key("e", "what is love")


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("  What   is\tLOVE ?! ", "what is love"),
        ("Who wrote Psalm 23...", "who wrote psalm 23"),
        ("¿Qué es el amor?", "¿qué es el amor"),
        ("Is it 3:16?", "is it 3:16"),
    ],
)
def test_normalization_rules(raw: str, normalized: str) -> None:
    """Protects the normalization contract; rejects keeping case, repeated spaces or end marks."""
    assert normalize_question(raw) == normalized


def test_mode_provider_and_model_are_part_of_the_key() -> None:
    """Protects against cross-serving; rejects a chat answer replayed for agent or a new model."""
    base = answer_cache_key("chat", "What is love?", "gemini", "flash")
    assert answer_cache_key("agent", "What is love?", "gemini", "flash") != base
    assert answer_cache_key("chat", "What is love?", "anthropic", "flash") != base
    assert answer_cache_key("chat", "What is love?", "gemini", "pro") != base
    assert answer_cache_key("chat", "What is wisdom?", "gemini", "flash") != base


@pytest.mark.parametrize(
    ("forwarded_for", "peer", "hops", "expected"),
    [
        ("203.0.113.7", "10.0.0.1", 1, "203.0.113.7"),
        # The client can prepend anything; only the entry the proxy appended counts.
        ("1.1.1.1, 203.0.113.7", "10.0.0.1", 1, "203.0.113.7"),
        ("203.0.113.7, 198.51.100.2", "10.0.0.1", 2, "203.0.113.7"),
        ("not-an-ip", "10.0.0.1", 1, "10.0.0.1"),
        (None, "10.0.0.1", 1, "10.0.0.1"),
        ("203.0.113.7", "10.0.0.1", 0, "10.0.0.1"),
        ("203.0.113.7:51234", None, 1, "203.0.113.7"),
        ("2001:db8:1:2:aaaa::1", None, 1, "2001:db8:1:2::/64"),
        ("[2001:db8:1:2::9]:443", None, 1, "2001:db8:1:2::/64"),
        ("::ffff:203.0.113.7", None, 1, "203.0.113.7"),
        (None, None, 1, "unknown"),
    ],
)
def test_client_key_from_forwarded_for(
    forwarded_for: str | None, peer: str | None, hops: int, expected: str
) -> None:
    """Protects the per-IP limit; rejects trusting spoofable entries or splitting one IPv6 /64."""
    assert client_key(forwarded_for, peer, hops) == expected


def test_quota_is_exhausted_until_next_pacific_midnight() -> None:
    """Protects the daily-reset rule; rejects reporting exhausted after the reset or before it."""
    exhausted_at = datetime(2026, 11, 27, 20, 0, tzinfo=UTC)  # 12:00 in Los Angeles (PST)
    reset = datetime(2026, 11, 28, 8, 0, tzinfo=UTC)  # 00:00 PST

    during = quota_status_at(exhausted_at, datetime(2026, 11, 28, 7, 59, tzinfo=UTC), PACIFIC)
    after = quota_status_at(exhausted_at, reset, PACIFIC)

    assert (during.label, during.resets_at) == ("exhausted", reset)
    assert (after.label, after.resets_at) == ("available", None)
    assert quota_status_at(None, reset, PACIFIC).label == "available"


def test_quota_reset_follows_daylight_saving_time() -> None:
    """Protects the reset hour in summer; rejects a fixed UTC-8 offset (PDT is UTC-7)."""
    status = quota_status_at(
        datetime(2026, 7, 1, 18, 0, tzinfo=UTC), datetime(2026, 7, 1, 19, 0, tzinfo=UTC), PACIFIC
    )

    assert status.resets_at == datetime(2026, 7, 2, 7, 0, tzinfo=UTC)


def test_rate_limit_message_says_when_to_retry() -> None:
    """Protects the 429 message; rejects a refusal without the wait time and retry moment."""
    now = datetime(2026, 11, 27, 14, 23, 10, tzinfo=UTC)
    start = window_start(now)
    decision = RateLimitDecision(False, 10, start.replace(hour=15))

    assert start == datetime(2026, 11, 27, 14, 0, tzinfo=UTC)
    assert decision.retry_after_seconds(now) == 37 * 60 - 10
    assert decision.message(now) == (
        "Rate limit reached: 10 questions per hour. Try again in 37 minutes (at 15:00 UTC). "
        "Cached example questions still work."
    )


def test_unknown_quota_time_zone_fails_at_startup() -> None:
    """Protects configuration; rejects a typo in QUOTA_RESET_TIMEZONE surfacing only at runtime."""
    with pytest.raises(ValidationError, match="unknown time zone"):
        make_settings(quota_reset_timezone="America/Nowhere")


def test_examples_endpoint_lists_six_questions() -> None:
    """Protects the demo's example list; rejects a missing endpoint or a different count."""
    response = TestClient(create_app(make_settings())).get("/api/v1/examples")

    assert response.status_code == 200
    assert response.json() == {"questions": list(EXAMPLE_QUESTIONS)}
    assert len(EXAMPLE_QUESTIONS) == 6


def test_health_without_database_omits_quota() -> None:
    """Protects liveness without a database; rejects failing or inventing a quota status."""
    response = TestClient(create_app(make_settings(database_url=None))).get("/health")

    assert response.json() == {"status": "ok"}
