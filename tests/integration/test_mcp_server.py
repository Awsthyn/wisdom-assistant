"""MCP tool calls through the SDK's in-process client against the test database."""

import json

from mcp import Client
from mcp_types import TextContent
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.books import BIBLE_BOOKS
from app.llm.fake import FakeProvider
from app.mcp_server import create_mcp_server
from app.repositories import books as books_repo
from app.repositories import verses as verses_repo

_BOOKS = {b.osis_code: b for b in BIBLE_BOOKS}


async def _seed(sessions: async_sessionmaker[AsyncSession]) -> None:
    async with sessions.begin() as session:
        ids = await books_repo.upsert_books(
            session,
            [
                books_repo.BookRow(
                    code, _BOOKS[code].name, _BOOKS[code].testament, _BOOKS[code].position, corpus
                )
                for code, corpus in (("Prov", True), ("Rom", False))
            ],
        )
        await verses_repo.upsert_verses(
            session,
            [
                verses_repo.VerseRow(ids["Prov"], 3, 5, "Trust in the LORD with all your heart."),
                verses_repo.VerseRow(ids["Prov"], 3, 6, "In all your ways acknowledge him."),
                verses_repo.VerseRow(ids["Rom"], 8, 28, "We know that all things work together."),
            ],
        )


async def _lookup(sessions: async_sessionmaker[AsyncSession], reference: str) -> TextContent:
    provider = FakeProvider()
    async with Client(create_mcp_server(sessions, provider)) as client:
        result = await client.call_tool("lookup_reference", {"reference": reference})
    assert provider.calls == []  # an exact lookup never embeds
    content = result.content[0]
    assert isinstance(content, TextContent)
    assert not result.is_error, content.text
    return content


async def test_lookup_reference_returns_the_verse_text(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Protects lookup_reference over MCP; rejects wrong text, reference or in_corpus."""
    await _seed(sessions)

    content = await _lookup(sessions, "Prov 3:5-6")

    assert json.loads(content.text) == {
        "reference": "Proverbs 3:5-6",
        "in_corpus": True,
        "passages": [
            {
                "reference": "Proverbs 3:5-6",
                "text": "[5] Trust in the LORD with all your heart. "
                "[6] In all your ways acknowledge him.",
            }
        ],
    }


async def test_lookup_outside_the_corpus_is_marked(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Protects in_corpus over MCP; rejects Romans 8:28 reported as a wisdom-book text."""
    await _seed(sessions)

    data = json.loads((await _lookup(sessions, "Romans 8:28")).text)

    assert (data["reference"], data["in_corpus"]) == ("Romans 8:28", False)


async def test_missing_verse_is_a_tool_error(sessions: async_sessionmaker[AsyncSession]) -> None:
    """Protects error reporting; rejects an empty success for a verse that does not exist."""
    await _seed(sessions)

    async with Client(create_mcp_server(sessions, FakeProvider())) as client:
        result = await client.call_tool("lookup_reference", {"reference": "Proverbs 3:99"})

    assert result.is_error
    content = result.content[0]
    assert isinstance(content, TextContent)
    assert "is not in the World English Bible" in content.text
