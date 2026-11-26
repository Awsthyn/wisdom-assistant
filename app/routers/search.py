"""`GET /api/v1/search`: semantic search over the corpus, or an exact reference."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.corpus import CORPUS_BOOKS
from app.core.references import InvalidReferenceError
from app.dependencies import get_embedder, get_session
from app.llm import Embedder, LLMConfigError, QuotaExceededError, TransientLLMError
from app.services import search as search_service

router = APIRouter(prefix="/api/v1", tags=["search"])

MIN_QUERY_LENGTH = 3
MAX_QUERY_LENGTH = 300


class SearchResult(BaseModel):
    reference: str
    book: str  # OSIS code, e.g. "Prov"
    book_name: str
    chapter: int
    start_verse: int
    end_verse: int
    text: str
    score: float | None  # cosine similarity in semantic mode; null for exact references
    in_corpus: bool
    url: str


class SearchResponse(BaseModel):
    query: str
    mode: Literal["semantic", "reference"]
    results: list[SearchResult]


@router.get(
    "/search",
    summary="Semantic or exact-reference search",
    responses={
        404: {"description": "The reference does not exist in the WEB (e.g. `John 3:99`)."},
        422: {"description": "Invalid parameters, unknown corpus book or impossible reference."},
        503: {"description": "Database or embedding service unavailable (e.g. quota)."},
    },
)
async def search(
    session: Annotated[AsyncSession, Depends(get_session)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    q: Annotated[
        str,
        Query(
            min_length=MIN_QUERY_LENGTH,
            max_length=MAX_QUERY_LENGTH,
            description="Free text, or an exact reference such as `John 3:16` or `Psalm 23`.",
        ),
    ],
    k: Annotated[int, Query(ge=1, le=20, description="Number of semantic results.")] = 8,
    book: Annotated[
        str | None,
        Query(description=f"Semantic results from one corpus book: {', '.join(CORPUS_BOOKS)}."),
    ] = None,
) -> SearchResponse:
    """Exact references (any of the 66 books) are answered from the database without an
    embedding call and ignore `k` and `book`; out-of-corpus books come back with
    `in_corpus: false`. Anything else is a semantic search over the five wisdom books,
    ordered by cosine similarity."""
    if len(q.strip()) < MIN_QUERY_LENGTH:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"q must contain at least {MIN_QUERY_LENGTH} non-blank characters",
        )
    try:
        outcome = await search_service.search(session, embedder, q, k, book)
    except (search_service.UnknownCorpusBookError, InvalidReferenceError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except search_service.ReferenceNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except QuotaExceededError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Semantic search is temporarily unavailable: the embedding quota is exhausted. "
            "Exact references such as 'Psalm 23' still work.",
        ) from exc
    except TransientLLMError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The embedding service did not respond. Please try again.",
        ) from exc
    except LLMConfigError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Semantic search is not configured."
        ) from exc

    return SearchResponse(
        query=q,
        mode=outcome.mode,
        results=[
            SearchResult(
                reference=hit.reference,
                book=hit.book.osis_code,
                book_name=hit.book.name,
                chapter=hit.chapter,
                start_verse=hit.start_verse,
                end_verse=hit.end_verse,
                text=hit.text,
                score=hit.score,
                in_corpus=hit.in_corpus,
                url=hit.url,
            )
            for hit in outcome.hits
        ],
    )
