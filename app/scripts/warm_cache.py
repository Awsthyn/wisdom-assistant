"""Pre-answer the curated example questions with the real LLM and store them in the cache.

    uv run python -m app.scripts.warm_cache            # chat answers, skipping cached ones
    uv run python -m app.scripts.warm_cache --agent    # agent answers too
    uv run python -m app.scripts.warm_cache --refresh  # answer again even if cached

Uses DATABASE_URL, the configured LLM_PROVIDER and GEMINI_API_KEY (embeddings).
The answers go through exactly the same pipeline and cache keys as the API, so the
examples keep working after the daily quota runs out. Stops at the first quota error.
"""

import argparse
import asyncio
import sys
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.core.examples import EXAMPLE_QUESTIONS
from app.db.session import create_engine, session_factory
from app.llm import Embedder, LLMError, LLMProvider, build_embedder, build_llm_provider
from app.services import agent as agent_service
from app.services import chat as chat_service
from app.services.guard import AnswerEntry, CachedQueryEmbedder, DemoGuard, Mode, StreamEvent
from app.tools import BIBLE_TOOLS, ToolContext


def _live_events(
    mode: Mode,
    question: str,
    session: AsyncSession,
    embedder: Embedder,
    llm: LLMProvider,
    settings: Settings,
) -> AsyncIterator[StreamEvent]:
    if mode == "chat":
        return (
            chat_service.to_stream_event(event)
            async for event in chat_service.chat_events(
                session,
                embedder,
                llm,
                question,
                top_k=settings.chat_top_k,
                min_score=settings.chat_min_score,
            )
        )
    context = ToolContext(session=session, embedder=embedder)
    return (
        agent_service.to_stream_event(event)
        async for event in agent_service.agent_events(llm, BIBLE_TOOLS, context, question)
    )


async def warm(
    sessions: async_sessionmaker[AsyncSession],
    settings: Settings,
    llm: LLMProvider,
    embedder: Embedder,
    modes: tuple[Mode, ...],
    *,
    refresh: bool,
) -> int:
    """Answer and cache every example; returns how many could not be cached."""
    guard = DemoGuard.from_settings(sessions, settings)
    cached_embedder = CachedQueryEmbedder(embedder, guard)
    failures = 0
    for mode in modes:
        for question in EXAMPLE_QUESTIONS:
            entry = AnswerEntry.build(mode, question, llm.name, llm.chat_model)
            if not refresh and await guard.cached_answer(entry.key) is not None:
                print(f"[{mode}] already cached: {question}")
                continue
            last: StreamEvent | None = None
            valid = 0
            async with sessions() as session:
                live = _live_events(mode, question, session, cached_embedder, llm, settings)
                async for event in guard.track(live, source=llm.name, entry=entry):
                    last = event
                    if event.name == "citations":
                        valid = len(event.data["valid"])
            if last is not None and last.name == "done":
                print(f"[{mode}] cached ({valid} valid citations): {question}")
                continue
            failures += 1
            code = last.data.get("code") if last is not None else "no events"
            print(f"[{mode}] NOT cached ({code}): {question}", file=sys.stderr)
            if code == "quota_exceeded":
                print("Quota exhausted; run again after the daily reset.", file=sys.stderr)
                return failures
    return failures


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    if settings.database_url is None or not settings.database_url.get_secret_value():
        print("DATABASE_URL is not set; see .env.example", file=sys.stderr)
        return 1
    try:
        llm = build_llm_provider(settings)
        embedder = build_embedder(settings)
    except LLMError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    modes: tuple[Mode, ...] = ("chat", "agent") if args.agent else ("chat",)
    engine = create_engine(settings.database_url.get_secret_value())
    try:
        failures = await warm(
            session_factory(engine), settings, llm, embedder, modes, refresh=args.refresh
        )
    finally:
        await engine.dispose()
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--agent", action="store_true", help="also cache agent answers")
    parser.add_argument("--refresh", action="store_true", help="answer cached questions again")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
