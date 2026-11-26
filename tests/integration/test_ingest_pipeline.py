"""Load -> chunk -> embed against a real Postgres, using a tiny invented corpus."""

from collections.abc import Sequence

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.books import BookInfo
from app.db.models import Book, Passage
from app.ingest.pacing import MinuteRateLimiter
from app.ingest.vpl import ParsedVerse, parse_vpl
from app.llm import EmbedTask, QuotaExceededError
from app.llm.fake import FakeProvider
from app.repositories.verses import get_verse
from app.services import ingest

pytestmark = pytest.mark.usefixtures("migrated_database")

# Three invented books: two in the test corpus, one outside it.
CATALOG = (
    BookInfo(1, "WisA", "Wisdom A", "OT", "WSA"),
    BookInfo(2, "WisB", "Wisdom B", "OT", "WSB"),
    BookInfo(3, "Out", "Outside", "NT", "OUT"),
)
CORPUS = ("WisA", "WisB")

VPL = [
    *(f"WSA 1:{n} Saying {n} of chapter one." for n in range(1, 8)),  # 7 verses
    *(f"WSA 2:{n} Saying {n} of chapter two." for n in range(1, 4)),  # 3 verses
    *(f"WSB 1:{n} Line {n}." for n in range(1, 6)),  # 5 verses
    *(f"OUT 1:{n} Outside text {n}." for n in range(1, 5)),  # 4 verses
    "OUT 1:5 ",  # omitted verse: present in the file, no text
]

Sessions = async_sessionmaker[AsyncSession]


def _verses(lines: Sequence[str] = VPL) -> list[ParsedVerse]:
    return list(parse_vpl(lines))


def _limiter() -> MinuteRateLimiter:
    return MinuteRateLimiter(requests_per_minute=10_000, tokens_per_minute=10_000_000)


async def _load(sessions: Sessions, lines: Sequence[str] = VPL) -> ingest.LoadReport:
    async with sessions() as session:
        return await ingest.load_verses(session, _verses(lines), catalog=CATALOG, corpus=CORPUS)


async def _chunk(sessions: Sessions) -> ingest.ChunkReport:
    async with sessions() as session:
        return await ingest.chunk_corpus(session, catalog=CATALOG)


async def _passage_spans(sessions: Sessions) -> list[tuple[str, int, int, int]]:
    async with sessions() as session:
        result = await session.execute(
            select(Book.osis_code, Passage.chapter, Passage.start_verse, Passage.end_verse)
            .join(Book, Book.id == Passage.book_id)
            .order_by(Book.position, Passage.chapter, Passage.start_verse)
        )
        return [(osis, ch, start, end) for osis, ch, start, end in result.all()]


class QuotaAfter(FakeProvider):
    """Fake embedder that raises QuotaExceededError after `allowed` successful calls."""

    def __init__(self, allowed: int) -> None:
        super().__init__()
        self.allowed = allowed

    async def embed(self, texts: Sequence[str], task: EmbedTask) -> list[list[float]]:
        if self.allowed <= 0:
            raise QuotaExceededError("daily quota")
        self.allowed -= 1
        return await super().embed(texts, task)


async def test_load_counts_books_and_skips_empty_verses(sessions: Sessions) -> None:
    """Protects the load totals; rejects loading the empty OUT 1:5 or miscounting the corpus."""
    report = await _load(sessions)

    assert report == ingest.LoadReport(books=3, corpus_books=2, verses=19, corpus_verses=15)


async def test_load_marks_only_corpus_books_in_corpus(sessions: Sessions) -> None:
    """Protects in_corpus; rejects marking the outside book or missing a corpus book."""
    await _load(sessions)

    async with sessions() as session:
        rows = await session.execute(select(Book.osis_code, Book.in_corpus))
        flags = {osis: in_corpus for osis, in_corpus in rows.all()}
    assert flags == {"WisA": True, "WisB": True, "Out": False}


async def test_load_is_idempotent_and_updates_changed_text(sessions: Sessions) -> None:
    """Protects re-runs; rejects duplicate rows on a second load or stale text after a fix."""
    first = await _load(sessions)
    fixed = [line.replace("Line 3.", "Line three.") for line in VPL]
    second = await _load(sessions, fixed)

    assert second == first
    async with sessions() as session:
        verse = await get_verse(session, "WisB", 1, 3)
    assert verse is not None
    assert verse.text == "Line three."


async def test_chunking_skips_books_outside_the_corpus(sessions: Sessions) -> None:
    """Protects corpus scope; rejects generating passages (and embeddings) for other books."""
    await _load(sessions)
    report = await _chunk(sessions)

    spans = await _passage_spans(sessions)
    assert {osis for osis, *_ in spans} == {"WisA", "WisB"}
    assert report.per_book == {"WisA": 3, "WisB": 1}


async def test_chunks_do_not_cross_chapters_and_overlap_by_two(sessions: Sessions) -> None:
    """Protects the chunking rules in the DB; rejects cross-chapter or non-overlapping spans."""
    await _load(sessions)
    await _chunk(sessions)

    assert await _passage_spans(sessions) == [
        ("WisA", 1, 1, 5),
        ("WisA", 1, 4, 7),
        ("WisA", 2, 1, 3),
        ("WisB", 1, 1, 5),
    ]
    async with sessions() as session:
        refs = (await session.execute(select(Passage.reference).order_by(Passage.id))).scalars()
        assert list(refs) == [
            "Wisdom A 1:1-5",
            "Wisdom A 1:4-7",
            "Wisdom A 2:1-3",
            "Wisdom B 1:1-5",
        ]


async def test_chunking_is_idempotent(sessions: Sessions) -> None:
    """Protects re-runs; rejects duplicated passages or wiped embeddings on a second chunk."""
    await _load(sessions)
    await _chunk(sessions)
    second = await _chunk(sessions)

    assert (second.passages, second.inserted, second.unchanged) == (4, 0, 4)


async def test_changed_text_clears_only_that_passage_embedding(sessions: Sessions) -> None:
    """Protects re-embedding after a text fix; rejects stale vectors or re-embedding everything."""
    await _load(sessions)
    await _chunk(sessions)
    await ingest.embed_pending(sessions, FakeProvider(), _limiter())

    await _load(sessions, [line.replace("Line 3.", "Line three.") for line in VPL])
    report = await _chunk(sessions)

    assert report.updated == 1
    async with sessions() as session:
        pending = await session.execute(
            select(Passage.reference).where(Passage.embedding.is_(None))
        )
        assert list(pending.scalars()) == ["Wisdom B 1:1-5"]


async def test_embed_resumes_after_quota_error(sessions: Sessions) -> None:
    """Protects resumability; rejects losing finished batches or re-embedding them on resume."""
    await _load(sessions)
    await _chunk(sessions)

    first = await ingest.embed_pending(sessions, QuotaAfter(allowed=2), _limiter(), batch_size=1)
    assert (first.quota_reached, first.embedded_now, first.embedded_total, first.total) == (
        True,
        2,
        2,
        4,
    )
    async with sessions() as session:
        rows = await session.execute(
            select(Passage.id, Passage.embedded_at).where(Passage.embedding.is_not(None))
        )
        done_before = {passage_id: embedded_at for passage_id, embedded_at in rows.all()}

    resumed_provider = FakeProvider()
    second = await ingest.embed_pending(sessions, resumed_provider, _limiter(), batch_size=1)

    assert (second.quota_reached, second.embedded_now, second.embedded_total) == (False, 2, 4)
    assert len([c for c in resumed_provider.calls if c.method == "embed"]) == 2
    async with sessions() as session:
        for passage_id, embedded_at in done_before.items():
            passage = await session.get(Passage, passage_id)
            assert passage is not None
            assert passage.embedded_at == embedded_at


async def test_embed_limit_and_stored_vector_dimensions(sessions: Sessions) -> None:
    """Protects --limit and the vector column; rejects ignoring the limit or wrong dimensions."""
    await _load(sessions)
    await _chunk(sessions)

    report = await ingest.embed_pending(sessions, FakeProvider(), _limiter(), limit=3)

    assert (report.embedded_now, report.embedded_total) == (3, 3)
    async with sessions() as session:
        vector = (
            (await session.execute(select(Passage.embedding).where(Passage.embedding.is_not(None))))
            .scalars()
            .first()
        )
    assert vector is not None
    assert len(vector) == 768


async def test_status_reports_all_counts(sessions: Sessions) -> None:
    """Protects `ingest status`; rejects counts that mix corpus and non-corpus rows."""
    await _load(sessions)
    await _chunk(sessions)
    await ingest.embed_pending(sessions, FakeProvider(), _limiter(), limit=1)

    async with sessions() as session:
        status = await ingest.ingest_status(session)

    assert (
        status.books,
        status.corpus_books,
        status.verses,
        status.corpus_verses,
        status.passages,
        status.embedded_passages,
    ) == (3, 2, 19, 15, 4, 1)
    assert status.database_bytes > 0
