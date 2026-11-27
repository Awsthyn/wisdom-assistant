"""`POST /api/v1/chat` end to end: real pgvector retrieval, fake LLM, SSE over HTTP."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.books import BIBLE_BOOKS
from app.db.models import Passage
from app.dependencies import get_embedder, get_llm, get_session
from app.llm import EmbedTask
from app.llm.fake import FakeProvider, hash_embedding
from app.main import create_app
from app.repositories import books as books_repo
from tests.conftest import make_settings

QUESTION = "What do the wisdom books say about caring for the poor?"
_PROV = next(b for b in BIBLE_BOOKS if b.osis_code == "Prov")


async def _seed(sessions: async_sessionmaker[AsyncSession]) -> None:
    """One Proverbs passage whose embedding equals the question's (score 1.0)."""
    async with sessions.begin() as session:
        ids = await books_repo.upsert_books(
            session, [books_repo.BookRow("Prov", _PROV.name, "OT", 20, True)]
        )
        session.add_all(
            [
                Passage(
                    book_id=ids["Prov"],
                    chapter=19,
                    start_verse=15,
                    end_verse=19,
                    text="[17] He who has pity on the poor lends to the LORD.",
                    reference="Proverbs 19:15-19",
                    embedding=hash_embedding(QUESTION, EmbedTask.QUERY, 768),
                ),
                Passage(
                    book_id=ids["Prov"],
                    chapter=1,
                    start_verse=1,
                    end_verse=5,
                    text="[1] The proverbs of Solomon.",
                    reference="Proverbs 1:1-5",
                    embedding=hash_embedding("unrelated", EmbedTask.DOCUMENT, 768),
                ),
            ]
        )


async def _chat(
    sessions: async_sessionmaker[AsyncSession], provider: FakeProvider, question: str
) -> list[tuple[str, dict[str, Any]]]:
    app = create_app(make_settings(chat_min_score=0.6))

    async def session_override() -> AsyncIterator[AsyncSession]:
        async with sessions() as session:
            yield session

    app.dependency_overrides[get_session] = session_override
    app.dependency_overrides[get_embedder] = lambda: provider
    app.dependency_overrides[get_llm] = lambda: provider
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/chat", json={"question": question})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events: list[tuple[str, dict[str, Any]]] = []
    for block in response.text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields:
            events.append((fields["event"], json.loads(fields["data"])))
    return events


async def test_chat_streams_grounded_answer_from_the_database(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Protects the full RAG pipeline; rejects wrong retrieval, event order or citation check."""
    await _seed(sessions)
    provider = FakeProvider(responses=["Pity on the poor is a loan to the LORD [Proverbs 19:17]."])

    events = await _chat(sessions, provider, QUESTION)

    names = [name for name, _ in events]
    assert names[0] == "sources"
    assert names[-2:] == ["citations", "done"]
    assert events[0][1]["passages"][0]["reference"] == "Proverbs 19:15-19"
    citations = dict(events)["citations"]
    assert [c["reference"] for c in citations["valid"]] == ["Proverbs 19:17"]
    assert citations["discarded"] == []
    assert [c.method for c in provider.calls] == ["embed", "stream"]


async def test_off_topic_question_never_reaches_the_llm(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Protects quota on off-topic questions; rejects a `stream` call when similarity is low."""
    await _seed(sessions)
    provider = FakeProvider()

    events = await _chat(sessions, provider, "best pizza in Rosario")

    assert [(name, data["code"]) for name, data in events] == [("error", "no_results")]
    assert [c.method for c in provider.calls] == ["embed"]
