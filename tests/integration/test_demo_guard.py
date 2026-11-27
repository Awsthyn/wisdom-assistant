"""Demo protection end to end on real Postgres: rate limit, answer and embedding caches,
quota status. The LLM is the fake provider, which records every call it receives."""

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.books import BIBLE_BOOKS
from app.db.models import Passage
from app.dependencies import get_base_embedder, get_guard, get_llm, get_session
from app.llm import EmbedTask, LLMResponse, Message, QuotaExceededError, ToolCall
from app.llm.fake import FakeProvider, hash_embedding
from app.main import create_app
from app.repositories import books as books_repo
from app.services.chat import QUOTA_MESSAGE
from app.services.guard import DemoGuard
from tests.conftest import make_settings

QUESTION = "What do the wisdom books say about caring for the poor?"
ANSWER = "Pity on the poor is a loan to the LORD [Proverbs 19:17]."
_PROV = next(b for b in BIBLE_BOOKS if b.osis_code == "Prov")

Events = list[tuple[str, dict[str, Any]]]


@dataclass
class _Clock:
    now: datetime = field(default_factory=lambda: datetime(2026, 11, 27, 18, 10, tzinfo=UTC))

    def __call__(self) -> datetime:
        return self.now


class _StreamQuotaProvider(FakeProvider):
    async def stream(
        self, messages: Sequence[Message], system: str | None = None
    ) -> AsyncIterator[str]:
        raise QuotaExceededError("429")
        yield ""  # pragma: no cover - makes this an async generator


@pytest.fixture
async def demo_sessions(
    sessions: async_sessionmaker[AsyncSession],
) -> async_sessionmaker[AsyncSession]:
    """Empty demo tables plus one Proverbs passage that matches QUESTION exactly."""
    async with sessions.begin() as session:
        await session.execute(
            text(
                "TRUNCATE rate_limit_windows, answer_cache, query_embedding_cache, llm_quota_status"
            )
        )
        ids = await books_repo.upsert_books(
            session, [books_repo.BookRow("Prov", _PROV.name, "OT", 20, True)]
        )
        session.add(
            Passage(
                book_id=ids["Prov"],
                chapter=19,
                start_verse=15,
                end_verse=19,
                text="[17] He who has pity on the poor lends to the LORD.",
                reference="Proverbs 19:15-19",
                embedding=hash_embedding(QUESTION, EmbedTask.QUERY, 768),
            )
        )
    return sessions


class _Demo:
    """An app wired to the test database, a fake LLM and a controllable clock."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        provider: FakeProvider,
        *,
        rate_limit: int = 10,
        clock: _Clock | None = None,
    ) -> None:
        settings = make_settings(chat_min_score=0.6, rate_limit_per_hour=rate_limit)
        self.provider = provider
        self.clock = clock or _Clock()
        self.guard = DemoGuard.from_settings(sessions, settings, clock=self.clock)
        self.app = create_app(settings)

        async def session_override() -> AsyncIterator[AsyncSession]:
            async with sessions() as session:
                yield session

        self.app.dependency_overrides[get_session] = session_override
        self.app.dependency_overrides[get_base_embedder] = lambda: provider
        self.app.dependency_overrides[get_llm] = lambda: provider
        self.app.dependency_overrides[get_guard] = lambda: self.guard

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://t")


def _events(response: httpx.Response) -> Events:
    events: Events = []
    for block in response.text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields:
            events.append((fields["event"], json.loads(fields["data"])))
    return events


def _methods(provider: FakeProvider) -> list[str]:
    return [call.method for call in provider.calls]


async def test_rate_limit_blocks_the_eleventh_request(
    demo_sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Protects the quota from one visitor; rejects an 11th request in one hour getting through."""
    demo = _Demo(demo_sessions, FakeProvider())
    visitor = {"X-Forwarded-For": "203.0.113.7"}

    async with demo.client() as client:
        statuses = [
            (
                await client.post(
                    "/api/v1/chat", json={"question": f"off-topic question {n}"}, headers=visitor
                )
            ).status_code
            for n in range(1, 11)
        ]
        blocked = await client.post(
            "/api/v1/chat", json={"question": "off-topic question 11"}, headers=visitor
        )
        agent_blocked = await client.post(
            "/api/v1/agent", json={"question": "anything else"}, headers=visitor
        )
        other = await client.post(
            "/api/v1/chat",
            json={"question": "off-topic question 11"},
            # Render appends the real address last; whatever precedes it is ignored.
            headers={"X-Forwarded-For": "203.0.113.7, 198.51.100.9"},
        )

    assert statuses == [200] * 10
    assert blocked.status_code == 429
    assert blocked.headers["retry-after"] == str(50 * 60)
    assert blocked.json()["detail"] == (
        "Rate limit reached: 10 questions per hour. Try again in 50 minutes (at 19:00 UTC). "
        "Cached example questions still work."
    )
    assert agent_blocked.status_code == 429  # /chat and /agent share the budget
    assert other.status_code == 200  # a different client has its own budget
    assert len(_methods(demo.provider)) == 11  # nothing ran for the blocked requests


async def test_next_window_allows_requests_again(
    demo_sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Protects window expiry; rejects a client staying blocked after the hour ends."""
    clock = _Clock()
    demo = _Demo(demo_sessions, FakeProvider(), rate_limit=1, clock=clock)

    async with demo.client() as client:
        first = await client.post("/api/v1/chat", json={"question": "off-topic one"})
        second = await client.post("/api/v1/chat", json={"question": "off-topic two"})
        clock.now += timedelta(hours=1)
        third = await client.post("/api/v1/chat", json={"question": "off-topic three"})

    assert [first.status_code, second.status_code, third.status_code] == [200, 429, 200]
    async with demo_sessions() as session:
        windows = (await session.execute(text("SELECT count(*) FROM rate_limit_windows"))).scalar()
    assert windows == 1  # the expired window was deleted


async def test_cache_hit_does_not_call_the_provider(
    demo_sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Protects the quota on repeats; rejects a second LLM call or a replay without cached: true."""
    demo = _Demo(demo_sessions, FakeProvider(responses=[ANSWER]))

    async with demo.client() as client:
        live = _events(await client.post("/api/v1/chat", json={"question": QUESTION}))
        calls_after_first = len(demo.provider.calls)
        variant = "  WHAT do the wisdom books say about caring for the poor  "
        replay = _events(await client.post("/api/v1/chat", json={"question": variant}))

    assert _methods(demo.provider) == ["embed", "stream"]
    assert len(demo.provider.calls) == calls_after_first
    assert [name for name, _ in replay] == ["sources", "token", "citations", "done"]
    assert replay[1][1] == {"text": ANSWER}
    assert dict(replay)["sources"] == dict(live)["sources"]
    assert dict(replay)["citations"] == dict(live)["citations"]
    assert dict(live)["done"] == {"model": "fake-chat"}
    assert dict(replay)["done"] == {"model": "fake-chat", "cached": True}


async def test_cache_hits_do_not_count_against_the_rate_limit(
    demo_sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Protects cached examples during a limit; rejects 429s for answers that cost nothing."""
    demo = _Demo(demo_sessions, FakeProvider(responses=[ANSWER]), rate_limit=1)

    async with demo.client() as client:
        statuses = [
            (await client.post("/api/v1/chat", json={"question": QUESTION})).status_code
            for _ in range(5)
        ]
        fresh = await client.post("/api/v1/chat", json={"question": "a new question"})

    assert statuses == [200] * 5
    assert fresh.status_code == 429


async def test_answers_with_history_are_not_cached(
    demo_sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Protects follow-ups; rejects replaying a standalone answer for a question with history."""
    demo = _Demo(demo_sessions, FakeProvider(responses=[ANSWER, ANSWER]))
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]

    async with demo.client() as client:
        for _ in range(2):
            await client.post("/api/v1/chat", json={"question": QUESTION, "history": history})

    assert _methods(demo.provider) == ["embed", "stream", "stream"]  # embedding was cached


async def test_agent_answer_is_replayed_from_the_cache(
    demo_sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Protects the agent's quota; rejects re-running the tool loop for a repeated question."""
    provider = FakeProvider(
        tool_responses=[
            LLMResponse(text="", tool_calls=(ToolCall("a", "search_passages", {"query": "poor"}),)),
            LLMResponse(text="Pity on the poor [Proverbs 19:17].", model="fake-chat"),
        ]
    )
    demo = _Demo(demo_sessions, provider)

    async with demo.client() as client:
        live = _events(await client.post("/api/v1/agent", json={"question": "The poor?"}))
        replay = _events(await client.post("/api/v1/agent", json={"question": "the poor"}))

    assert _methods(provider).count("generate_with_tools") == 2
    assert [name for name, _ in replay] == [name for name, _ in live]
    assert dict(replay)["done"] == {**dict(live)["done"], "cached": True}


async def test_query_embeddings_are_cached_by_normalized_text(
    demo_sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Protects the embedding quota; rejects re-embedding "What is love?" as "what is love"."""
    demo = _Demo(demo_sessions, FakeProvider())

    async with demo.client() as client:
        first = await client.get("/api/v1/search", params={"q": "What is love?"})
        second = await client.get("/api/v1/search", params={"q": "what is love"})

    assert first.status_code == second.status_code == 200
    assert _methods(demo.provider) == ["embed"]
    assert first.json()["results"] == second.json()["results"]


async def test_exhausted_quota_is_reported_by_health(
    demo_sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Protects the UI's early warning; rejects /health saying available after a quota error."""
    demo = _Demo(demo_sessions, _StreamQuotaProvider())

    async with demo.client() as client:
        before = (await client.get("/health")).json()
        events = _events(await client.post("/api/v1/chat", json={"question": QUESTION}))
        after = (await client.get("/health")).json()

    assert before == {"status": "ok", "llm_quota": "available"}
    assert events[-1] == ("error", {"code": "quota_exceeded", "message": QUOTA_MESSAGE})
    assert QUOTA_MESSAGE == (
        "The free AI quota for today is used up. Cached example questions still work: "
        "try one of them."
    )
    # 18:10 UTC on Nov 27 is 10:10 in Los Angeles; the quota resets at 00:00 PST (08:00 UTC).
    assert after == {
        "status": "ok",
        "llm_quota": "exhausted",
        "llm_quota_resets_at": "2026-11-28T08:00:00Z",
    }

    # A restarted process (fresh guard, no memory) reads the same state from Postgres.
    restarted = _Demo(demo_sessions, FakeProvider(), clock=demo.clock)
    async with restarted.client() as client:
        assert (await client.get("/health")).json()["llm_quota"] == "exhausted"
        restarted.clock.now += timedelta(hours=14)  # past midnight Pacific
        assert (await client.get("/health")).json()["llm_quota"] == "available"


async def test_successful_answer_clears_the_exhausted_status(
    demo_sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Protects recovery from a short 429; rejects staying "exhausted" after the LLM answers."""
    demo = _Demo(demo_sessions, FakeProvider(responses=[ANSWER]))
    await demo.guard.record_quota_exhausted("fake")

    async with demo.client() as client:
        assert (await client.get("/health")).json()["llm_quota"] == "exhausted"
        await client.post("/api/v1/chat", json={"question": QUESTION})
        assert (await client.get("/health")).json()["llm_quota"] == "available"


async def test_embedding_quota_error_is_recorded(
    demo_sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Protects the status when embeddings run out; rejects a search 503 that /health ignores."""

    class _EmbedQuotaProvider(FakeProvider):
        async def embed(self, texts: Sequence[str], task: EmbedTask) -> list[list[float]]:
            raise QuotaExceededError("429")

    demo = _Demo(demo_sessions, _EmbedQuotaProvider())

    async with demo.client() as client:
        search = await client.get("/api/v1/search", params={"q": "What is love?"})
        health = (await client.get("/health")).json()

    assert search.status_code == 503
    assert health["llm_quota"] == "exhausted"
    async with demo_sessions() as session:
        source = (await session.execute(text("SELECT source FROM llm_quota_status"))).scalar()
    assert source == "embeddings"
