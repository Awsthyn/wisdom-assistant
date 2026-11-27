"""Corpus ingestion CLI.

    uv run python -m app.scripts.ingest download
    uv run python -m app.scripts.ingest load
    uv run python -m app.scripts.ingest chunk
    uv run python -m app.scripts.ingest embed [--limit N]
    uv run python -m app.scripts.ingest xrefs [--refresh]
    uv run python -m app.scripts.ingest status

Uses DATABASE_URL_DIRECT (falls back to DATABASE_URL). Run
`uv run alembic upgrade head` once before the first `load`.
"""

import argparse
import asyncio
import io
import sys
from collections.abc import Awaitable, Callable, Coroutine
from pathlib import Path
from typing import Any, cast

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.db.session import create_engine, session_factory
from app.ingest.pacing import MinuteRateLimiter
from app.llm import LLMError, build_llm_provider
from app.services import ingest

RAW_DIR = Path("data/raw")


def _database_url(settings: Settings) -> str:
    secret = settings.database_url_direct or settings.database_url
    if secret is None or not secret.get_secret_value():
        raise SystemExit("DATABASE_URL_DIRECT (or DATABASE_URL) is not set; see .env.example")
    return secret.get_secret_value()


async def _with_sessions(
    settings: Settings, work: Callable[[async_sessionmaker[AsyncSession]], Awaitable[int]]
) -> int:
    engine = create_engine(_database_url(settings))
    try:
        return await work(session_factory(engine))
    finally:
        await engine.dispose()


async def cmd_download(_: Settings, __: argparse.Namespace) -> int:
    path = await ingest.download_source(RAW_DIR)
    print(f"Downloaded {ingest.SOURCE_URL}")
    print(f"Verified {path} ({path.stat().st_size:,} bytes)")
    return 0


async def cmd_load(settings: Settings, _: argparse.Namespace) -> int:
    verses = ingest.read_source(RAW_DIR / ingest.SOURCE_MEMBER)

    async def work(sessions: async_sessionmaker[AsyncSession]) -> int:
        async with sessions() as session:
            report = await ingest.load_verses(session, verses)
            checks = await ingest.verify_full_bible(session, report)
        print(f"Books:  {report.books} ({report.corpus_books} in corpus)")
        print(f"Verses: {report.verses:,} ({report.corpus_verses:,} in corpus)")
        for check in checks:
            print(f"  [{'OK' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
        return 0 if all(c.ok for c in checks) else 1

    return await _with_sessions(settings, work)


async def cmd_chunk(settings: Settings, _: argparse.Namespace) -> int:
    async def work(sessions: async_sessionmaker[AsyncSession]) -> int:
        async with sessions() as session:
            report = await ingest.chunk_corpus(session)
        print(f"Passages: {report.passages:,}")
        print(
            f"  inserted={report.inserted} updated={report.updated} "
            f"deleted={report.deleted} unchanged={report.unchanged}"
        )
        for osis, count in report.per_book.items():
            print(f"  {osis}: {count}")
        return 0

    return await _with_sessions(settings, work)


def _print_progress(embedded: int, total: int) -> None:
    percent = 100 * embedded / total if total else 100.0
    print(f"  {embedded}/{total} ({percent:.1f}%)", flush=True)


async def cmd_embed(settings: Settings, args: argparse.Namespace) -> int:
    try:
        provider = build_llm_provider(settings)
    except LLMError as exc:
        print(f"Cannot build the embeddings provider: {exc}")
        return 1
    limiter = MinuteRateLimiter(
        requests_per_minute=settings.embed_rpm, tokens_per_minute=settings.embed_tpm
    )

    async def work(sessions: async_sessionmaker[AsyncSession]) -> int:
        print(
            f"Embedding with {provider.embed_model} "
            f"(EMBED_RPM={settings.embed_rpm}, EMBED_TPM={settings.embed_tpm})"
        )
        report = await ingest.embed_pending(
            sessions, provider, limiter, limit=args.limit, on_progress=_print_progress
        )
        if report.quota_reached:
            print(
                f"Daily quota reached: {report.embedded_total}/{report.total} passages embedded. "
                "Run again tomorrow."
            )
            return 0
        print(
            f"Done: {report.embedded_now} embedded now, "
            f"{report.embedded_total}/{report.total} total."
        )
        return 0

    return await _with_sessions(settings, work)


async def cmd_xrefs(settings: Settings, args: argparse.Namespace) -> int:
    path = RAW_DIR / ingest.XREFS_MEMBER
    if args.refresh or not path.is_file():
        path = await ingest.download_xrefs(RAW_DIR)
        print(f"Downloaded {ingest.XREFS_URL}")
    lines = ingest.read_xrefs_lines(path)

    async def work(sessions: async_sessionmaker[AsyncSession]) -> int:
        async with sessions() as session:
            r = await ingest.load_xrefs(session, lines)
        print(f"Source rows:            {r.source_rows:,}")
        print(f"Skipped (votes < {ingest.XREFS_MIN_VOTES}):   {r.skipped_low_votes:,}")
        print(f"Skipped (cross-book):   {r.skipped_cross_book:,}")
        print(f"Skipped (unknown book): {r.skipped_unknown_book:,}")
        print(f"Duplicates merged:      {r.duplicates:,}")
        print(f"Loaded:                 {r.loaded:,}")
        print(f"From a corpus book:     {r.from_corpus:,}")
        print(f"Table size:             {r.table_bytes / 1_048_576:.1f} MB (with indexes)")
        return 0

    return await _with_sessions(settings, work)


async def cmd_status(settings: Settings, _: argparse.Namespace) -> int:
    async def work(sessions: async_sessionmaker[AsyncSession]) -> int:
        async with sessions() as session:
            s = await ingest.ingest_status(session)
        percent = 100 * s.embedded_passages / s.passages if s.passages else 0.0
        print(f"Books:     {s.books} total, {s.corpus_books} in corpus")
        print(f"Verses:    {s.verses:,} total, {s.corpus_verses:,} in corpus")
        print(f"Passages:  {s.passages:,}")
        print(f"Embedded:  {s.embedded_passages:,}/{s.passages:,} ({percent:.1f}%)")
        print(f"Xrefs:     {s.cross_references:,}")
        print(f"DB size:   {s.database_bytes / 1_048_576:.1f} MB")
        return 0

    return await _with_sessions(settings, work)


COMMANDS: dict[str, Callable[[Settings, argparse.Namespace], Coroutine[Any, Any, int]]] = {
    "download": cmd_download,
    "load": cmd_load,
    "chunk": cmd_chunk,
    "embed": cmd_embed,
    "xrefs": cmd_xrefs,
    "status": cmd_status,
}


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy code page; verse text uses curly quotes.
    stdout = sys.stdout
    if isinstance(stdout, io.TextIOWrapper):
        cast("io.TextIOWrapper[io.BufferedWriter]", stdout).reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="ingest", description="Corpus ingestion CLI.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        command = sub.add_parser(name)
        if name == "embed":
            command.add_argument("--limit", type=int, default=None, help="embed at most N passages")
        if name == "xrefs":
            command.add_argument("--refresh", action="store_true", help="download the file again")
    args = parser.parse_args(argv)
    try:
        return asyncio.run(COMMANDS[args.command](get_settings(), args))
    except ingest.IngestError as exc:
        print(f"Error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Completed batches are saved; run the same command to resume.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
