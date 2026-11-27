"""Cross-reference loading, the real tools and `/api/v1/agent` against Postgres + pgvector."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.books import BIBLE_BOOKS
from app.dependencies import get_embedder, get_llm, get_session
from app.llm import LLMResponse, ToolCall
from app.llm.fake import FakeProvider
from app.main import create_app
from app.repositories import books as books_repo
from app.repositories import verses as verses_repo
from app.services import ingest
from app.tools import CROSS_REFERENCES, LOOKUP_REFERENCE, ToolContext
from tests.conftest import make_settings

_BOOKS = {b.osis_code: b for b in BIBLE_BOOKS}

XREF_LINES = [
    "From Verse\tTo Verse\tVotes\t#www.openbible.info CC-BY\n",
    "Ps.23.1\tJohn.10.11\t212\n",
    "Ps.23.1\tEzek.34.11-Ezek.34.12\t90\n",
    "Ps.23.2\tJohn.10.11\t40\n",  # same target from verse 2: merged, best votes win
    "Ps.23.1\tPs.80.1\t3\n",  # below the votes threshold
    "Gen.1.1\tJohn.1.1\t300\n",  # Genesis is not loaded in this test database
    "John.10.11\tPs.23.1\t150\n",
]


async def _seed(sessions: async_sessionmaker[AsyncSession]) -> None:
    async with sessions.begin() as session:
        ids = await books_repo.upsert_books(
            session,
            [
                books_repo.BookRow(
                    code, _BOOKS[code].name, _BOOKS[code].testament, _BOOKS[code].position, corpus
                )
                for code, corpus in (("Ps", True), ("Ezek", False), ("John", False))
            ],
        )
        await verses_repo.upsert_verses(
            session,
            [
                *(verses_repo.VerseRow(ids["Ps"], 23, n, f"Ps 23:{n}.") for n in range(1, 7)),
                verses_repo.VerseRow(ids["Ezek"], 34, 11, "I myself will search for my sheep."),
                verses_repo.VerseRow(ids["Ezek"], 34, 12, "As a shepherd seeks out his flock."),
                verses_repo.VerseRow(ids["John"], 10, 11, "I am the good shepherd."),
            ],
        )


async def test_load_xrefs_filters_counts_and_is_idempotent(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Protects `ingest xrefs`; rejects low-vote rows, unknown books or duplicates on re-run."""
    await _seed(sessions)

    async with sessions() as session:
        first = await ingest.load_xrefs(session, XREF_LINES)
    async with sessions() as session:
        second = await ingest.load_xrefs(session, XREF_LINES)

    assert (first.loaded, first.from_corpus) == (4, 3)
    assert (first.skipped_low_votes, first.skipped_unknown_book) == (1, 1)
    assert (second.loaded, second.from_corpus) == (first.loaded, first.from_corpus)
    async with sessions() as session:
        assert (await ingest.ingest_status(session)).cross_references == 4


async def test_cross_references_tool_orders_by_votes_with_text(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Protects the cross_references tool; rejects wrong order, missing text or in_corpus."""
    await _seed(sessions)
    async with sessions() as session:
        await ingest.load_xrefs(session, XREF_LINES)

    async with sessions() as session:
        output = await CROSS_REFERENCES.run(
            ToolContext(session, FakeProvider()), {"reference": "Psalm 23:1-2"}
        )

    assert output.data["source"] == "Psalm 23:1-2"
    assert output.data["cross_references"] == [
        {
            "reference": "John 10:11",
            "votes": 212,
            "in_corpus": False,
            "text": "[11] I am the good shepherd.",
        },
        {
            "reference": "Ezekiel 34:11-12",
            "votes": 90,
            "in_corpus": False,
            "text": (
                "[11] I myself will search for my sheep. [12] As a shepherd seeks out his flock."
            ),
        },
    ]
    assert "2 cross reference(s) for Psalm 23:1-2 (2 outside the corpus)" in output.summary


async def test_lookup_reference_tool_reads_any_book(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Protects lookup_reference; rejects refusing non-corpus books or losing in_corpus: false."""
    await _seed(sessions)

    async with sessions() as session:
        output = await LOOKUP_REFERENCE.run(
            ToolContext(session, FakeProvider()), {"reference": "Jn 10:11"}
        )

    assert output.data == {
        "reference": "John 10:11",
        "in_corpus": False,
        "passages": [{"reference": "John 10:11", "text": "[11] I am the good shepherd."}],
    }


async def test_agent_endpoint_end_to_end_with_real_tools(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Protects /api/v1/agent over SSE; rejects wrong event order or unflagged cross references."""
    await _seed(sessions)
    async with sessions() as session:
        await ingest.load_xrefs(session, XREF_LINES)
    provider = FakeProvider(
        tool_responses=[
            LLMResponse(
                text="",
                tool_calls=(
                    ToolCall("a", "lookup_reference", {"reference": "Psalm 23:1"}),
                    ToolCall("b", "cross_references", {"reference": "Psalm 23:1", "limit": 5}),
                ),
            ),
            LLMResponse(
                text="The LORD is a shepherd [Psalm 23:1]; the image returns in [John 10:11].",
                model="fake-chat",
            ),
        ]
    )
    app = create_app(make_settings())

    async def session_override() -> AsyncIterator[AsyncSession]:
        async with sessions() as session:
            yield session

    app.dependency_overrides[get_session] = session_override
    app.dependency_overrides[get_embedder] = lambda: provider
    app.dependency_overrides[get_llm] = lambda: provider
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/agent", json={"question": "God as a shepherd?"})

    assert response.status_code == 200
    events: list[tuple[str, dict[str, Any]]] = []
    for block in response.text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields:
            events.append((fields["event"], json.loads(fields["data"])))
    assert [name for name, _ in events] == [
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "token",
        "citations",
        "done",
    ]
    assert events[2][1] == {
        "name": "cross_references",
        "arguments": {"reference": "Psalm 23:1", "limit": 5},
    }
    citations = dict(events)["citations"]
    assert [(c["reference"], c["outside_corpus"]) for c in citations["valid"]] == [
        ("Psalm 23:1", False),
        ("John 10:11", True),
    ]
    assert dict(events)["done"] == {"model": "fake-chat", "iterations": 2, "tool_calls": 2}
