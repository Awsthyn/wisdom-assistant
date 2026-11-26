"""`/api/v1/search` end to end against a real Postgres + pgvector, with the fake provider."""

import math
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.books import BIBLE_BOOKS
from app.db.models import Passage
from app.dependencies import get_embedder, get_session
from app.llm import EmbedTask
from app.llm.fake import FakeProvider, hash_embedding
from app.main import create_app
from app.repositories import books as books_repo
from app.repositories import verses as verses_repo
from tests.conftest import make_settings

QUERY = "the tongue has power of life and death"
DIMENSIONS = 768
_BOOKS = {b.osis_code: b for b in BIBLE_BOOKS}


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector]


def _blend(a: list[float], b: list[float], weight: float) -> list[float]:
    """Unit vector between `a` and `b`; a larger `weight` is closer to `a`."""
    return _unit([weight * x + (1 - weight) * y for x, y in zip(a, b, strict=True)])


async def _seed(sessions: async_sessionmaker[AsyncSession]) -> None:
    """Psalms and Proverbs (corpus) and John (outside), with controlled passage embeddings."""
    query_vector = hash_embedding(QUERY, EmbedTask.QUERY, DIMENSIONS)

    def noise(label: str) -> list[float]:
        return hash_embedding(label, EmbedTask.DOCUMENT, DIMENSIONS)

    async with sessions.begin() as session:
        ids = await books_repo.upsert_books(
            session,
            [
                books_repo.BookRow(code, _BOOKS[code].name, _BOOKS[code].testament, pos, corpus)
                for code, pos, corpus in (("Ps", 19, True), ("Prov", 20, True), ("John", 43, False))
            ],
        )
        await verses_repo.upsert_verses(
            session,
            [
                *(
                    verses_repo.VerseRow(ids["Ps"], 23, n, f"Psalm 23 verse {n}.")
                    for n in range(1, 7)
                ),
                verses_repo.VerseRow(ids["John"], 3, 16, "For God so loved the world..."),
                verses_repo.VerseRow(ids["Prov"], 18, 21, "Death and life are in the tongue."),
            ],
        )
        passages: list[tuple[str, int, int, int, list[float] | None]] = [
            # (book, chapter, start, end, embedding)
            ("Prov", 18, 19, 23, query_vector),  # identical to the query: score 1.0
            ("Prov", 15, 1, 5, _blend(query_vector, noise("a"), 0.8)),
            ("Ps", 34, 11, 15, _blend(query_vector, noise("b"), 0.6)),
            ("Prov", 10, 18, 22, _blend(query_vector, noise("c"), 0.4)),
            ("Ps", 1, 1, 5, noise("d")),
            ("Ps", 23, 1, 5, noise("e")),
            ("Ps", 119, 1, 5, None),  # not embedded yet: must never be returned
        ]
        session.add_all(
            Passage(
                book_id=ids[code],
                chapter=chapter,
                start_verse=start,
                end_verse=end,
                text=f"{code} {chapter}:{start}-{end} text",
                reference=f"{_BOOKS[code].reference_name} {chapter}:{start}-{end}",
                embedding=embedding,
            )
            for code, chapter, start, end, embedding in passages
        )


@pytest.fixture
async def api(
    sessions: async_sessionmaker[AsyncSession],
) -> AsyncIterator[tuple[httpx.AsyncClient, FakeProvider]]:
    await _seed(sessions)
    fake = FakeProvider(dimensions=DIMENSIONS)
    app = create_app(make_settings())

    async def session_override() -> AsyncIterator[AsyncSession]:
        async with sessions() as session:
            yield session

    app.dependency_overrides[get_session] = session_override
    app.dependency_overrides[get_embedder] = lambda: fake
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, fake


async def _search(client: httpx.AsyncClient, **params: Any) -> httpx.Response:
    return await client.get("/api/v1/search", params=params)


async def test_semantic_results_are_ordered_by_score(
    api: tuple[httpx.AsyncClient, FakeProvider],
) -> None:
    """Protects ranking by cosine similarity; rejects unordered results or a wrong best match."""
    client, fake = api

    response = await _search(client, q=QUERY, k=4)

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "semantic"
    references = [r["reference"] for r in body["results"]]
    assert references == [
        "Proverbs 18:19-23",
        "Proverbs 15:1-5",
        "Psalm 34:11-15",
        "Proverbs 10:18-22",
    ]
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1.0, abs=1e-4)
    assert [c.method for c in fake.calls] == ["embed"]


async def test_semantic_result_shape(api: tuple[httpx.AsyncClient, FakeProvider]) -> None:
    """Protects the result contract; rejects missing fields, wrong book code or a broken URL."""
    client, _ = api

    [top] = (await _search(client, q=QUERY, k=1)).json()["results"]

    assert top == {
        "reference": "Proverbs 18:19-23",
        "book": "Prov",
        "book_name": "Proverbs",
        "chapter": 18,
        "start_verse": 19,
        "end_verse": 23,
        "text": "Prov 18:19-23 text",
        "score": pytest.approx(1.0, abs=1e-4),
        "in_corpus": True,
        "url": "https://ebible.org/engwebp/PRO18.htm",
    }


async def test_book_filter_returns_only_that_book(
    api: tuple[httpx.AsyncClient, FakeProvider],
) -> None:
    """Protects the book filter; rejects results from other books or dropping filtered matches."""
    client, _ = api

    response = await _search(client, q=QUERY, k=20, book="ps")

    results = response.json()["results"]
    assert response.status_code == 200
    assert {r["book"] for r in results} == {"Ps"}
    # All three embedded Psalms passages; the unembedded Psalm 119 one is never returned.
    assert results[0]["reference"] == "Psalm 34:11-15"
    assert len(results) == 3


async def test_exact_reference_does_not_embed(
    api: tuple[httpx.AsyncClient, FakeProvider],
) -> None:
    """Protects the quota-free reference path; rejects an embedding call for `Psalm 23`."""
    client, fake = api

    response = await _search(client, q="Psalm 23")

    body = response.json()
    assert response.status_code == 200
    assert body["mode"] == "reference"
    [chapter] = body["results"]
    assert chapter["reference"] == "Psalm 23:1-6"
    assert chapter["text"].startswith("[1] Psalm 23 verse 1. [2] Psalm 23 verse 2.")
    assert chapter["score"] is None
    assert chapter["in_corpus"] is True
    assert fake.calls == []


async def test_exact_reference_outside_corpus_is_marked(
    api: tuple[httpx.AsyncClient, FakeProvider],
) -> None:
    """Protects out-of-corpus labelling; rejects John 3:16 coming back as in_corpus: true."""
    client, fake = api

    response = await _search(client, q="Jn 3:16")

    [verse] = response.json()["results"]
    assert response.status_code == 200
    assert verse["reference"] == "John 3:16"
    assert verse["text"] == "[16] For God so loved the world..."
    assert verse["in_corpus"] is False
    assert verse["url"] == "https://ebible.org/engwebp/JHN03.htm"
    assert fake.calls == []


async def test_missing_verse_is_404(api: tuple[httpx.AsyncClient, FakeProvider]) -> None:
    """Protects well-formed but absent references; rejects an empty 200 or a semantic fallback."""
    client, fake = api

    response = await _search(client, q="John 3:99")

    assert response.status_code == 404
    assert fake.calls == []


async def test_verse_range_is_clamped_to_existing_verses(
    api: tuple[httpx.AsyncClient, FakeProvider],
) -> None:
    """Protects partial ranges; rejects a 404 or a reference claiming verses that do not exist."""
    client, _ = api

    [chapter] = (await _search(client, q="Psalm 23:5-40")).json()["results"]

    assert chapter["reference"] == "Psalm 23:5-6"
