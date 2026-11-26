from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Book


@dataclass(frozen=True, slots=True)
class BookRow:
    osis_code: str
    name: str
    testament: str
    position: int
    in_corpus: bool


async def upsert_books(session: AsyncSession, rows: Sequence[BookRow]) -> dict[str, int]:
    """Insert or update books by OSIS code. Returns `{osis_code: id}`."""
    if not rows:
        return {}
    stmt = insert(Book).values(
        [
            {
                "osis_code": r.osis_code,
                "name": r.name,
                "testament": r.testament,
                "position": r.position,
                "in_corpus": r.in_corpus,
            }
            for r in rows
        ]
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[Book.osis_code],
        set_={
            "name": stmt.excluded.name,
            "testament": stmt.excluded.testament,
            "position": stmt.excluded.position,
            "in_corpus": stmt.excluded.in_corpus,
        },
    ).returning(Book.osis_code, Book.id)
    result = await session.execute(stmt)
    return {osis: book_id for osis, book_id in result.all()}


async def corpus_book_ids(session: AsyncSession) -> dict[int, str]:
    """`{book_id: osis_code}` for books with in_corpus = true."""
    result = await session.execute(select(Book.id, Book.osis_code).where(Book.in_corpus))
    return {book_id: osis for book_id, osis in result.all()}


async def count_books(session: AsyncSession) -> tuple[int, int]:
    """Return `(total, in_corpus)`."""
    result = await session.execute(
        select(func.count(), func.count().filter(Book.in_corpus)).select_from(Book)
    )
    total, in_corpus = result.one()
    return total, in_corpus
