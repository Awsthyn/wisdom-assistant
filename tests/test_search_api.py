"""Request validation and error mapping of `/api/v1/search`, without a database."""

from collections.abc import AsyncIterator, Sequence

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dependencies import get_embedder, get_session
from app.llm import EmbedTask, QuotaExceededError
from app.llm.fake import FakeProvider
from app.main import create_app
from tests.conftest import make_settings


class _QuotaEmbedder:
    embed_model = "quota"

    async def embed(self, texts: Sequence[str], task: EmbedTask) -> list[list[float]]:
        raise QuotaExceededError("429 RESOURCE_EXHAUSTED")


async def _no_session() -> AsyncIterator[object]:
    yield object()  # these requests must fail before touching the database


def _client(embedder: object | None = None) -> tuple[TestClient, FakeProvider]:
    fake = FakeProvider()
    app: FastAPI = create_app(make_settings())
    app.dependency_overrides[get_session] = _no_session
    app.dependency_overrides[get_embedder] = lambda: embedder or fake
    return TestClient(app), fake


def test_book_outside_corpus_is_422_listing_the_corpus_books() -> None:
    """Protects the corpus-only book filter; rejects book=John being accepted or a vague error."""
    client, fake = _client()

    response = client.get("/api/v1/search", params={"q": "wisdom and folly", "book": "John"})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "John" in detail
    for code in ("Job", "Ps", "Prov", "Eccl", "Song"):
        assert code in detail
    assert fake.calls == []


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"q": "ab"},
        {"q": "   "},
        {"q": "x" * 301},
        {"q": "wisdom", "k": 0},
        {"q": "wisdom", "k": 21},
    ],
)
def test_invalid_query_parameters_are_422(params: dict[str, str | int]) -> None:
    """Protects q (3-300 chars) and k (1-20) limits; rejects out-of-range values being searched."""
    client, fake = _client()

    response = client.get("/api/v1/search", params=params)

    assert response.status_code == 422
    assert fake.calls == []


def test_impossible_reference_is_422_without_embedding() -> None:
    """Protects `John 99:1` handling; rejects falling back to a (quota-spending) semantic search."""
    client, fake = _client()

    response = client.get("/api/v1/search", params={"q": "John 99:1"})

    assert response.status_code == 422
    assert "21 chapters" in response.json()["detail"]
    assert fake.calls == []


def test_exhausted_embedding_quota_is_503_with_a_clear_message() -> None:
    """Protects the free-tier quota path; rejects a 500 when Gemini answers 429."""
    client, _ = _client(_QuotaEmbedder())

    response = client.get("/api/v1/search", params={"q": "wisdom and folly"})

    assert response.status_code == 503
    assert "quota" in response.json()["detail"]


def test_search_without_database_configuration_is_503() -> None:
    """Protects deploys without DATABASE_URL; rejects a 500 or a crash at startup."""
    client = TestClient(create_app(make_settings(database_url=None)))

    response = client.get("/api/v1/search", params={"q": "Psalm 23"})

    assert response.status_code == 503
    assert "DATABASE_URL" in response.json()["detail"]
