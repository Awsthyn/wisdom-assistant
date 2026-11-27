"""Cross references (OpenBible.info, CC-BY): where a verse's themes reappear in the Bible."""

from dataclasses import dataclass
from typing import Final

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.books import BIBLE_BOOKS, BookInfo
from app.core.references import InvalidReferenceError, Reference, parse_reference
from app.repositories import cross_references as xrefs_repo
from app.services.search import (
    NotAReferenceError,
    ReferenceNotFoundError,
    SearchHit,
    reference_hits,
)

_BOOKS_BY_OSIS: Final = {b.osis_code: b for b in BIBLE_BOOKS}
_LAST_VERSE: Final = 999  # "to the end of the chapter" for whole-chapter sources


@dataclass(frozen=True, slots=True)
class CrossReferenceHit:
    reference: str  # canonical label, e.g. "John 10:11" or "Romans 1:32-2:1"
    book: BookInfo
    votes: int
    in_corpus: bool
    passages: list[SearchHit]  # the target's text, one entry per chapter


@dataclass(frozen=True, slots=True)
class CrossReferenceResult:
    source: str
    targets: list[CrossReferenceHit]


async def cross_references(session: AsyncSession, text: str, limit: int) -> CrossReferenceResult:
    """Cross references of a verse or range within one chapter, most votes first.

    Each target carries its text (from any of the 66 books) and `in_corpus`.
    Raises `NotAReferenceError`, `InvalidReferenceError` or `ReferenceNotFoundError`.
    """
    source = parse_reference(text)
    if source is None:
        raise NotAReferenceError(f"'{text.strip()}' is not a Bible reference (e.g. 'Psalm 23:1')")
    if source.start_chapter != source.end_chapter:
        raise InvalidReferenceError(
            f"'{text.strip()}': cross references need verses from a single chapter"
        )
    rows = await xrefs_repo.targets_from(
        session,
        source.book.osis_code,
        source.start_chapter,
        source.start_verse,
        source.end_verse if source.end_verse is not None else _LAST_VERSE,
        limit,
    )
    targets: list[CrossReferenceHit] = []
    for row in rows:
        book = _BOOKS_BY_OSIS[row.osis_code]
        target = Reference(book, row.chapter, row.start_verse, row.end_chapter, row.end_verse)
        try:
            passages = await reference_hits(session, target)
        except ReferenceNotFoundError:
            continue  # versification gap: the target verse is not in the WEB
        targets.append(CrossReferenceHit(target.label, book, row.votes, row.in_corpus, passages))
    return CrossReferenceResult(source=source.label, targets=targets)
