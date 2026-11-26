from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import delete, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Book, Passage


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


@dataclass(frozen=True, slots=True)
class PassageHit:
    osis_code: str
    chapter: int
    start_verse: int
    end_verse: int
    reference: str
    text: str
    in_corpus: bool
    distance: float


async def nearest_passages(
    session: AsyncSession, vector: Sequence[float], k: int, osis_code: str | None = None
) -> list[PassageHit]:
    """The `k` embedded passages closest to `vector` by cosine distance, nearest first.

    Uses the HNSW index. With a book filter, a plain HNSW scan visits only
    `hnsw.ef_search` candidates and filters afterwards, so a small book could come
    back with fewer than `k` results; pgvector's iterative scan (>= 0.8) keeps
    scanning until `k` rows pass the filter, in exact distance order.
    """
    await session.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
    distance = Passage.embedding.cosine_distance(list(vector)).label("distance")
    stmt = (
        select(
            Book.osis_code,
            Passage.chapter,
            Passage.start_verse,
            Passage.end_verse,
            Passage.reference,
            Passage.text,
            Book.in_corpus,
            distance,
        )
        .join(Book, Book.id == Passage.book_id)
        .where(Passage.embedding.is_not(None))
        .order_by(distance)
        .limit(k)
    )
    if osis_code is not None:
        stmt = stmt.where(Book.osis_code == osis_code)
    result = await session.execute(stmt)
    return [PassageHit(*row) for row in result.all()]
