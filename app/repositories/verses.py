from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Book, Verse

# 4 bind parameters per row; stays far below asyncpg's 32,767-parameter limit.
_UPSERT_BATCH = 2000


@dataclass(frozen=True, slots=True)
class VerseRow:
    book_id: int
    chapter: int
    verse: int
    text: str


@dataclass(frozen=True, slots=True)
class StoredVerse:
    osis_code: str
    chapter: int
    verse: int
    text: str
    in_corpus: bool


async def upsert_verses(session: AsyncSession, rows: Sequence[VerseRow]) -> None:
    """Insert verses, updating the text of any that already exist."""
    for start in range(0, len(rows), _UPSERT_BATCH):
        batch = rows[start : start + _UPSERT_BATCH]
        stmt = insert(Verse).values(
            [
                {"book_id": r.book_id, "chapter": r.chapter, "verse": r.verse, "text": r.text}
                for r in batch
            ]
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_verses_book_chapter_verse",
            set_={"text": stmt.excluded.text},
            where=Verse.text.is_distinct_from(stmt.excluded.text),
        )
        await session.execute(stmt)


async def count_verses(session: AsyncSession) -> tuple[int, int]:
    """Return `(total, in_corpus)`."""
    result = await session.execute(
        select(func.count(), func.count().filter(Book.in_corpus)).select_from(Verse).join(Book)
    )
    total, in_corpus = result.one()
    return total, in_corpus


async def get_verse(
    session: AsyncSession, osis_code: str, chapter: int, verse: int
) -> StoredVerse | None:
    result = await session.execute(
        select(Book.osis_code, Verse.chapter, Verse.verse, Verse.text, Book.in_corpus)
        .join(Book)
        .where(Book.osis_code == osis_code, Verse.chapter == chapter, Verse.verse == verse)
    )
    row = result.one_or_none()
    return StoredVerse(*row) if row else None


async def verses_for_book(session: AsyncSession, book_id: int) -> list[tuple[int, int, str]]:
    """`(chapter, verse, text)` for a book, in reading order."""
    result = await session.execute(
        select(Verse.chapter, Verse.verse, Verse.text)
        .where(Verse.book_id == book_id)
        .order_by(Verse.chapter, Verse.verse)
    )
    return [(chapter, verse, text) for chapter, verse, text in result.all()]
