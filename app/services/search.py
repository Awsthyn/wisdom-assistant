"""Search: exact references first, semantic search over the corpus otherwise.

If the query parses as a reference (`John 3:16`, `Psalm 23`), the verses are read
straight from the database, from any of the 66 books, without an embedding call.
Anything else is embedded as a query and matched against corpus passages.
"""

from dataclasses import dataclass
from itertools import groupby
from typing import Final, Literal

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.books import BIBLE_BOOKS, BookInfo
from app.core.corpus import CORPUS_BOOKS
from app.core.references import Reference, parse_reference
from app.ingest.chunking import VerseText, format_passage_text, format_reference
from app.llm import Embedder, EmbedTask
from app.repositories import passages as passages_repo
from app.repositories import verses as verses_repo

SearchMode = Literal["semantic", "reference"]

_BOOKS_BY_OSIS: Final = {b.osis_code: b for b in BIBLE_BOOKS}
_CORPUS_BY_KEY: Final = {code.lower(): code for code in CORPUS_BOOKS}


class UnknownCorpusBookError(ValueError):
    """The `book` filter names a book outside the corpus."""

    def __init__(self, book: str) -> None:
        super().__init__(
            f"Unknown book '{book}'. Semantic search covers only the corpus books: "
            f"{', '.join(CORPUS_BOOKS)}."
        )


class ReferenceNotFoundError(LookupError):
    """The reference is well formed but none of its verses exist (e.g. `John 3:99`)."""


class NotAReferenceError(ValueError):
    """Text that was expected to be a reference does not parse as one."""


@dataclass(frozen=True, slots=True)
class SearchHit:
    book: BookInfo
    chapter: int
    start_verse: int
    end_verse: int
    reference: str
    text: str
    in_corpus: bool
    score: float | None  # cosine similarity; None for exact references

    @property
    def url(self) -> str:
        return self.book.chapter_url(self.chapter)


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    mode: SearchMode
    hits: list[SearchHit]


def resolve_corpus_book(book: str) -> str:
    """Map a `book` filter (case-insensitive OSIS code) to its code in `CORPUS_BOOKS`."""
    code = _CORPUS_BY_KEY.get(book.strip().lower())
    if code is None:
        raise UnknownCorpusBookError(book)
    return code


async def search(
    session: AsyncSession,
    embedder: Embedder,
    query: str,
    k: int,
    book: str | None = None,
) -> SearchOutcome:
    """Run a search. `book` filters semantic results; exact references ignore it.

    Raises `UnknownCorpusBookError`, `InvalidReferenceError` (from the parser),
    `ReferenceNotFoundError`, and the LLM layer's `QuotaExceededError` /
    `TransientLLMError` when the query embedding cannot be obtained.
    """
    corpus_book = resolve_corpus_book(book) if book is not None else None
    reference = parse_reference(query)
    if reference is not None:
        return SearchOutcome("reference", await reference_hits(session, reference, query))

    return SearchOutcome("semantic", await semantic_hits(session, embedder, query, k, corpus_book))


async def reference_hits(
    session: AsyncSession, reference: Reference, label: str | None = None
) -> list[SearchHit]:
    """The verses of `reference`, one hit per chapter. Raises `ReferenceNotFoundError`."""
    verses = await verses_repo.verses_in_range(
        session,
        reference.book.osis_code,
        (reference.start_chapter, reference.start_verse),
        (reference.end_chapter, reference.end_verse),
    )
    if not verses:
        shown = (label or "").strip() or reference.book.name
        raise ReferenceNotFoundError(f"'{shown}' is not in the World English Bible")
    return _group_by_chapter(reference.book, verses)


async def lookup_reference(session: AsyncSession, text: str) -> list[SearchHit]:
    """Exact text of a reference to any of the 66 books, one hit per chapter.

    Raises `NotAReferenceError`, `InvalidReferenceError` or `ReferenceNotFoundError`.
    """
    reference = parse_reference(text)
    if reference is None:
        raise NotAReferenceError(f"'{text.strip()}' is not a Bible reference (e.g. 'Psalm 23:1')")
    return await reference_hits(session, reference, text)


async def semantic_hits(
    session: AsyncSession,
    embedder: Embedder,
    query: str,
    k: int,
    corpus_book: str | None = None,
) -> list[SearchHit]:
    """The `k` corpus passages most similar to `query`, best first (one embedding call)."""
    [vector] = await embedder.embed([query.strip()], EmbedTask.QUERY)
    rows = await passages_repo.nearest_passages(session, vector, k, corpus_book)
    return [
        SearchHit(
            book=_BOOKS_BY_OSIS[row.osis_code],
            chapter=row.chapter,
            start_verse=row.start_verse,
            end_verse=row.end_verse,
            reference=row.reference,
            text=row.text,
            in_corpus=row.in_corpus,
            score=round(1.0 - row.distance, 4),
        )
        for row in rows
    ]


def _group_by_chapter(book: BookInfo, verses: list[verses_repo.StoredVerse]) -> list[SearchHit]:
    """One hit per chapter, so every hit has a single-chapter reference and URL."""
    hits: list[SearchHit] = []
    for chapter, group in groupby(verses, key=lambda v: v.chapter):
        chapter_verses = list(group)
        first, last = chapter_verses[0].verse, chapter_verses[-1].verse
        hits.append(
            SearchHit(
                book=book,
                chapter=chapter,
                start_verse=first,
                end_verse=last,
                reference=format_reference(book.reference_name, chapter, first, last),
                text=format_passage_text([VerseText(v.verse, v.text) for v in chapter_verses]),
                in_corpus=chapter_verses[0].in_corpus,
                score=None,
            )
        )
    return hits
