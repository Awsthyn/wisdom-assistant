from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Passage


@dataclass(frozen=True, slots=True)
class PassageRow:
    chapter: int
    start_verse: int
    end_verse: int
    text: str
    reference: str


@dataclass(frozen=True, slots=True)
class SyncResult:
    inserted: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0


@dataclass(frozen=True, slots=True)
class PendingPassage:
    id: int
    text: str


async def sync_book_passages(
    session: AsyncSession, book_id: int, rows: Sequence[PassageRow]
) -> SyncResult:
    """Make the book's passages match `rows` exactly.

    New spans are inserted, spans whose text changed are updated and their
    embedding cleared (so they get re-embedded), spans no longer produced are
    deleted, and unchanged spans keep their embedding.
    """
    existing = {
        (chapter, start, end): (passage_id, text)
        for passage_id, chapter, start, end, text in (
            await session.execute(
                select(
                    Passage.id,
                    Passage.chapter,
                    Passage.start_verse,
                    Passage.end_verse,
                    Passage.text,
                ).where(Passage.book_id == book_id)
            )
        ).all()
    }
    wanted = {(r.chapter, r.start_verse, r.end_verse): r for r in rows}

    stale_ids = [pid for span, (pid, _) in existing.items() if span not in wanted]
    if stale_ids:
        await session.execute(delete(Passage).where(Passage.id.in_(stale_ids)))

    inserted = updated = unchanged = 0
    new_rows: list[Passage] = []
    for span, row in wanted.items():
        current = existing.get(span)
        if current is None:
            new_rows.append(
                Passage(
                    book_id=book_id,
                    chapter=row.chapter,
                    start_verse=row.start_verse,
                    end_verse=row.end_verse,
                    text=row.text,
                    reference=row.reference,
                )
            )
            inserted += 1
        elif current[1] != row.text:
            await session.execute(
                update(Passage)
                .where(Passage.id == current[0])
                .values(text=row.text, reference=row.reference, embedding=None, embedded_at=None)
            )
            updated += 1
        else:
            unchanged += 1
    session.add_all(new_rows)
    await session.flush()
    return SyncResult(inserted, updated, len(stale_ids), unchanged)


async def delete_passages_outside(session: AsyncSession, book_ids: Collection[int]) -> int:
    """Delete passages of books not in `book_ids` (e.g. a book removed from the corpus)."""
    result = await session.execute(
        delete(Passage).where(Passage.book_id.not_in(book_ids)).returning(Passage.id)
    )
    return len(result.all())


async def count_passages(session: AsyncSession) -> tuple[int, int]:
    """Return `(total, embedded)`."""
    result = await session.execute(
        select(func.count(), func.count(Passage.embedding)).select_from(Passage)
    )
    total, embedded = result.one()
    return total, embedded


async def next_unembedded(session: AsyncSession, limit: int) -> list[PendingPassage]:
    """Oldest passages that still have no embedding."""
    result = await session.execute(
        select(Passage.id, Passage.text)
        .where(Passage.embedding.is_(None))
        .order_by(Passage.id)
        .limit(limit)
    )
    return [PendingPassage(pid, text) for pid, text in result.all()]


async def save_embeddings(
    session: AsyncSession, embeddings: Sequence[tuple[int, list[float]]]
) -> None:
    now = datetime.now(UTC)
    for passage_id, vector in embeddings:
        await session.execute(
            update(Passage)
            .where(Passage.id == passage_id)
            .values(embedding=vector, embedded_at=now)
        )
