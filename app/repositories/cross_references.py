from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import func, insert, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.db.models import Book, CrossReference

# 9 bind parameters per row; stays far below asyncpg's 32,767-parameter limit.
_INSERT_BATCH = 3000


@dataclass(frozen=True, slots=True)
class CrossReferenceRow:
    from_book_id: int
    from_chapter: int
    from_verse: int
    to_book_id: int
    to_chapter: int
    to_start_verse: int
    to_end_chapter: int
    to_end_verse: int
    votes: int


@dataclass(frozen=True, slots=True)
class CrossReferenceTarget:
    osis_code: str
    chapter: int
    start_verse: int
    end_chapter: int
    end_verse: int
    votes: int
    in_corpus: bool


async def replace_all(session: AsyncSession, rows: Sequence[CrossReferenceRow]) -> None:
    """Replace the whole table with `rows` (call inside one transaction).

    TRUNCATE rather than DELETE: it frees the old rows' space at once instead of
    leaving dead tuples for autovacuum, which matters on a 1 GB database.
    """
    await session.execute(text("TRUNCATE cross_references RESTART IDENTITY"))
    for start in range(0, len(rows), _INSERT_BATCH):
        batch = rows[start : start + _INSERT_BATCH]
        await session.execute(
            insert(CrossReference),
            [
                {
                    "from_book_id": r.from_book_id,
                    "from_chapter": r.from_chapter,
                    "from_verse": r.from_verse,
                    "to_book_id": r.to_book_id,
                    "to_chapter": r.to_chapter,
                    "to_start_verse": r.to_start_verse,
                    "to_end_chapter": r.to_end_chapter,
                    "to_end_verse": r.to_end_verse,
                    "votes": r.votes,
                }
                for r in batch
            ],
        )


async def count_cross_references(session: AsyncSession) -> tuple[int, int]:
    """Return `(total, from_corpus)`: all rows, and rows whose source is a corpus book."""
    result = await session.execute(
        select(func.count(), func.count().filter(Book.in_corpus))
        .select_from(CrossReference)
        .join(Book, Book.id == CrossReference.from_book_id)
    )
    total, from_corpus = result.one()
    return total, from_corpus


async def table_size_bytes(session: AsyncSession) -> int:
    """Table plus its indexes."""
    result = await session.execute(text("SELECT pg_total_relation_size('cross_references')"))
    return int(result.scalar_one())


async def targets_from(
    session: AsyncSession,
    osis_code: str,
    chapter: int,
    start_verse: int,
    end_verse: int,
    limit: int,
) -> list[CrossReferenceTarget]:
    """Targets of the source verses `chapter:start_verse-end_verse`, most votes first.

    A target reached from several source verses appears once, with its best vote count.
    """
    source = aliased(Book)
    target = aliased(Book)
    votes = func.max(CrossReference.votes).label("votes")
    result = await session.execute(
        select(
            target.osis_code,
            CrossReference.to_chapter,
            CrossReference.to_start_verse,
            CrossReference.to_end_chapter,
            CrossReference.to_end_verse,
            votes,
            target.in_corpus,
        )
        .join(source, source.id == CrossReference.from_book_id)
        .join(target, target.id == CrossReference.to_book_id)
        .where(
            source.osis_code == osis_code,
            CrossReference.from_chapter == chapter,
            CrossReference.from_verse.between(start_verse, end_verse),
        )
        .group_by(
            target.osis_code,
            target.position,
            target.in_corpus,
            CrossReference.to_chapter,
            CrossReference.to_start_verse,
            CrossReference.to_end_chapter,
            CrossReference.to_end_verse,
        )
        .order_by(votes.desc(), target.position, CrossReference.to_chapter)
        .limit(limit)
    )
    return [CrossReferenceTarget(*row) for row in result.all()]
