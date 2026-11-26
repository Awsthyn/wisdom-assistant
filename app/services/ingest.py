"""Ingestion pipeline: download -> load -> chunk -> embed, plus status.

Every step is idempotent and safe to re-run. `embed` only touches passages whose
embedding is NULL and commits after each batch, so it resumes where it stopped.
"""

import asyncio
import zipfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import httpx2
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.books import BIBLE_BOOKS, BookInfo
from app.core.corpus import CORPUS_BOOKS
from app.ingest.chunking import (
    DEFAULT_OVERLAP,
    DEFAULT_SIZE,
    VerseText,
    chunk_chapter,
    format_reference,
)
from app.ingest.pacing import MinuteRateLimiter, estimate_tokens
from app.ingest.vpl import ParsedVerse, parse_vpl
from app.llm import Embedder, EmbedTask, QuotaExceededError
from app.repositories import books as books_repo
from app.repositories import passages as passages_repo
from app.repositories import stats as stats_repo
from app.repositories import verses as verses_repo

SOURCE_URL = "https://ebible.org/Scriptures/engwebp_vpl.zip"
SOURCE_ZIP = "engwebp_vpl.zip"
SOURCE_MEMBER = "engwebp_vpl.txt"

# Gemini's batch embedding endpoint accepts at most 100 texts per request.
MAX_EMBED_BATCH = 100


class IngestError(RuntimeError):
    """The source or the database is not in the state a step requires."""


# --- download ----------------------------------------------------------------


async def download_source(raw_dir: Path, url: str = SOURCE_URL) -> Path:
    """Download the VPL zip into `raw_dir` and extract only the verse file.

    Returns the path of the extracted `.txt`. Raises `IngestError` if the file
    is missing or does not look like the expected VPL text.
    """
    async with httpx2.AsyncClient(follow_redirects=True, timeout=120) as client:
        response = await client.get(url)
        response.raise_for_status()
    return await asyncio.to_thread(_save_source, response.content, raw_dir, url)


def _save_source(content: bytes, raw_dir: Path, url: str) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / SOURCE_ZIP
    zip_path.write_bytes(content)
    txt_path = raw_dir / SOURCE_MEMBER
    with zipfile.ZipFile(zip_path) as archive:
        if SOURCE_MEMBER not in archive.namelist():
            raise IngestError(f"{SOURCE_MEMBER} not found in {url}")
        # Read the single member by name; never extract arbitrary archive paths.
        txt_path.write_bytes(archive.read(SOURCE_MEMBER))
    verify_source_file(txt_path)
    return txt_path


def verify_source_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise IngestError(f"{path} is missing or empty; run `ingest download` first")
    with path.open(encoding="utf-8-sig") as handle:
        first = handle.readline()
    if not first.startswith("GEN 1:1 "):
        raise IngestError(f"{path} does not start with Genesis 1:1; unexpected format")


def read_source(path: Path) -> list[ParsedVerse]:
    verify_source_file(path)
    with path.open(encoding="utf-8-sig") as handle:
        return list(parse_vpl(handle))


# --- load --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoadReport:
    books: int
    corpus_books: int
    verses: int
    corpus_verses: int


async def load_verses(
    session: AsyncSession,
    verses: Iterable[ParsedVerse],
    *,
    catalog: Sequence[BookInfo] = BIBLE_BOOKS,
    corpus: Sequence[str] = CORPUS_BOOKS,
) -> LoadReport:
    """Upsert every catalog book and every verse; mark corpus books `in_corpus`."""
    catalog_codes = {b.osis_code for b in catalog}
    missing = set(corpus) - catalog_codes
    if missing:
        raise IngestError(f"corpus books not in catalog: {sorted(missing)}")

    book_ids = await books_repo.upsert_books(
        session,
        [
            books_repo.BookRow(
                osis_code=b.osis_code,
                name=b.name,
                testament=b.testament,
                position=b.position,
                in_corpus=b.osis_code in corpus,
            )
            for b in catalog
        ],
    )
    by_vpl = {b.vpl_code: b for b in catalog}
    rows: list[verses_repo.VerseRow] = []
    for verse in verses:
        book = by_vpl.get(verse.book_code)
        if book is None:
            raise IngestError(f"unknown book code {verse.book_code!r}")
        rows.append(
            verses_repo.VerseRow(
                book_id=book_ids[book.osis_code],
                chapter=verse.chapter,
                verse=verse.verse,
                text=verse.text,
            )
        )
    await verses_repo.upsert_verses(session, rows)
    await session.commit()

    books_total, books_in_corpus = await books_repo.count_books(session)
    verses_total, verses_in_corpus = await verses_repo.count_verses(session)
    return LoadReport(books_total, books_in_corpus, verses_total, verses_in_corpus)


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str


async def verify_full_bible(session: AsyncSession, report: LoadReport) -> list[Check]:
    """Sanity checks for the real 66-book WEB load."""
    checks = [
        Check("66 books", report.books == 66, f"books={report.books}"),
        Check(
            "exactly 5 corpus books",
            report.corpus_books == 5,
            f"in_corpus={report.corpus_books}",
        ),
        Check(
            "31,000-31,200 verses",
            31_000 <= report.verses <= 31_200,
            f"verses={report.verses}",
        ),
        Check(
            "~4,785 corpus verses",
            4_700 <= report.corpus_verses <= 4_850,
            f"corpus_verses={report.corpus_verses}",
        ),
    ]
    expectations = [
        ("Ps", 23, 1, True, "A Psalm by David. The LORD is my shepherd"),
        ("Prov", 3, 5, True, "Trust in the LORD with all your heart"),
        ("John", 3, 16, False, "For God so loved the world"),
    ]
    for osis, chapter, number, in_corpus, prefix in expectations:
        verse = await verses_repo.get_verse(session, osis, chapter, number)
        label = f"{osis} {chapter}:{number}"
        if verse is None:
            checks.append(Check(label, False, "missing"))
            continue
        ok = verse.in_corpus == in_corpus and verse.text.startswith(prefix)
        checks.append(Check(label, ok, f"in_corpus={verse.in_corpus} text={verse.text[:60]!r}"))
    return checks


# --- chunk -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChunkReport:
    passages: int
    inserted: int
    updated: int
    deleted: int
    unchanged: int
    per_book: dict[str, int] = field(default_factory=dict[str, int])


async def chunk_corpus(
    session: AsyncSession,
    *,
    catalog: Sequence[BookInfo] = BIBLE_BOOKS,
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> ChunkReport:
    """Build passages for `in_corpus` books only; remove passages of other books."""
    by_osis = {b.osis_code: b for b in catalog}
    corpus_ids = await books_repo.corpus_book_ids(session)
    if not corpus_ids:
        raise IngestError("no corpus books in the database; run `ingest load` first")

    inserted = updated = unchanged = 0
    deleted = await passages_repo.delete_passages_outside(session, corpus_ids.keys())
    per_book: dict[str, int] = {}
    for book_id, osis in corpus_ids.items():
        chapters: dict[int, list[VerseText]] = defaultdict(list)
        for chapter, number, text in await verses_repo.verses_for_book(session, book_id):
            chapters[chapter].append(VerseText(number, text))
        rows = [
            passages_repo.PassageRow(
                chapter=chunk.chapter,
                start_verse=chunk.start_verse,
                end_verse=chunk.end_verse,
                text=chunk.text,
                reference=format_reference(
                    by_osis[osis].reference_name, chunk.chapter, chunk.start_verse, chunk.end_verse
                ),
            )
            for chapter, verses in sorted(chapters.items())
            for chunk in chunk_chapter(chapter, verses, size, overlap)
        ]
        result = await passages_repo.sync_book_passages(session, book_id, rows)
        inserted += result.inserted
        updated += result.updated
        deleted += result.deleted
        unchanged += result.unchanged
        per_book[osis] = len(rows)
    await session.commit()
    total, _ = await passages_repo.count_passages(session)
    return ChunkReport(total, inserted, updated, deleted, unchanged, per_book)


# --- embed -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EmbedReport:
    embedded_now: int
    embedded_total: int
    total: int
    quota_reached: bool


ProgressCallback = Callable[[int, int], None]


def _fit_batch(
    pending: Sequence[passages_repo.PendingPassage], max_tokens: int
) -> list[passages_repo.PendingPassage]:
    """Largest prefix of `pending` whose estimated tokens fit in `max_tokens` (at least 1)."""
    batch: list[passages_repo.PendingPassage] = []
    used = 0
    for passage in pending:
        tokens = estimate_tokens(passage.text)
        if batch and used + tokens > max_tokens:
            break
        batch.append(passage)
        used += tokens
    return batch


async def embed_pending(
    sessions: async_sessionmaker[AsyncSession],
    embedder: Embedder,
    limiter: MinuteRateLimiter,
    *,
    batch_size: int = MAX_EMBED_BATCH,
    limit: int | None = None,
    on_progress: ProgressCallback | None = None,
) -> EmbedReport:
    """Embed passages whose embedding is NULL, one committed batch at a time.

    Stops cleanly on `QuotaExceededError`: batches already committed are kept,
    so the next run resumes from the first NULL embedding.
    """
    batch_size = max(1, min(batch_size, MAX_EMBED_BATCH))
    embedded_now = 0
    quota_reached = False
    while limit is None or embedded_now < limit:
        want = batch_size if limit is None else min(batch_size, limit - embedded_now)
        async with sessions() as session:
            pending = await passages_repo.next_unembedded(session, want)
            if not pending:
                break
            batch = _fit_batch(pending, limiter.tokens_per_minute)
            await limiter.acquire(sum(estimate_tokens(p.text) for p in batch))
            try:
                vectors = await embedder.embed([p.text for p in batch], EmbedTask.DOCUMENT)
            except QuotaExceededError:
                quota_reached = True
                break
            await passages_repo.save_embeddings(
                session, [(p.id, v) for p, v in zip(batch, vectors, strict=True)]
            )
            await session.commit()
            embedded_now += len(batch)
            if on_progress is not None:
                total, embedded = await passages_repo.count_passages(session)
                on_progress(embedded, total)

    async with sessions() as session:
        total, embedded_total = await passages_repo.count_passages(session)
    return EmbedReport(embedded_now, embedded_total, total, quota_reached)


# --- status ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StatusReport:
    books: int
    corpus_books: int
    verses: int
    corpus_verses: int
    passages: int
    embedded_passages: int
    database_bytes: int


async def ingest_status(session: AsyncSession) -> StatusReport:
    books_total, books_in_corpus = await books_repo.count_books(session)
    verses_total, verses_in_corpus = await verses_repo.count_verses(session)
    passages_total, embedded = await passages_repo.count_passages(session)
    size = await stats_repo.database_size_bytes(session)
    return StatusReport(
        books_total, books_in_corpus, verses_total, verses_in_corpus, passages_total, embedded, size
    )
