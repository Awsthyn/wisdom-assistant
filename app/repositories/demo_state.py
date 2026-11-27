"""Public-demo state: rate-limit windows, cached answers and query embeddings, quota status.

Callers own the transaction (`async with sessions.begin()`); nothing here commits.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import CachedAnswer, CachedQueryEmbedding, LLMQuotaStatus, RateLimitWindow

_QUOTA_ROW_ID = 1


async def consume_request(
    session: AsyncSession, client_key: str, window_start: datetime, limit: int
) -> int | None:
    """Count one request for `client_key` in the window, atomically.

    Returns the request count after this one, or None when the window already
    holds `limit` requests (the counter is then left unchanged).
    """
    statement = (
        insert(RateLimitWindow)
        .values(client_key=client_key, window_start=window_start, requests=1)
        .on_conflict_do_update(
            index_elements=[RateLimitWindow.client_key, RateLimitWindow.window_start],
            set_={"requests": RateLimitWindow.requests + 1},
            where=RateLimitWindow.requests < limit,
        )
        .returning(RateLimitWindow.requests)
    )
    return (await session.execute(statement)).scalar_one_or_none()


async def delete_windows_before(session: AsyncSession, cutoff: datetime) -> None:
    await session.execute(delete(RateLimitWindow).where(RateLimitWindow.window_start < cutoff))


async def get_answer(
    session: AsyncSession, key: str, not_before: datetime
) -> list[dict[str, Any]] | None:
    """The cached events for `key`, if stored at or after `not_before`."""
    statement = select(CachedAnswer.events).where(
        CachedAnswer.key == key, CachedAnswer.created_at >= not_before
    )
    return (await session.execute(statement)).scalar_one_or_none()


async def put_answer(
    session: AsyncSession,
    *,
    key: str,
    mode: str,
    question: str,
    provider: str,
    model: str,
    events: list[dict[str, Any]],
    created_at: datetime,
) -> None:
    values = {
        "mode": mode,
        "question": question,
        "provider": provider,
        "model": model,
        "events": events,
        "created_at": created_at,
    }
    statement = (
        insert(CachedAnswer)
        .values(key=key, **values)
        .on_conflict_do_update(index_elements=[CachedAnswer.key], set_=values)
    )
    await session.execute(statement)


async def delete_answers_before(session: AsyncSession, cutoff: datetime) -> None:
    await session.execute(delete(CachedAnswer).where(CachedAnswer.created_at < cutoff))


async def get_query_embedding(
    session: AsyncSession, key: str, not_before: datetime
) -> list[float] | None:
    statement = select(CachedQueryEmbedding.embedding).where(
        CachedQueryEmbedding.key == key, CachedQueryEmbedding.created_at >= not_before
    )
    vector = (await session.execute(statement)).scalar_one_or_none()
    return None if vector is None else [float(x) for x in vector]


async def put_query_embedding(
    session: AsyncSession, key: str, model: str, embedding: list[float], created_at: datetime
) -> None:
    values = {"model": model, "embedding": embedding, "created_at": created_at}
    statement = (
        insert(CachedQueryEmbedding)
        .values(key=key, **values)
        .on_conflict_do_update(index_elements=[CachedQueryEmbedding.key], set_=values)
    )
    await session.execute(statement)


async def delete_query_embeddings_before(session: AsyncSession, cutoff: datetime) -> None:
    await session.execute(
        delete(CachedQueryEmbedding).where(CachedQueryEmbedding.created_at < cutoff)
    )


async def quota_exhausted_at(session: AsyncSession) -> datetime | None:
    statement = select(LLMQuotaStatus.exhausted_at).where(LLMQuotaStatus.id == _QUOTA_ROW_ID)
    return (await session.execute(statement)).scalar_one_or_none()


async def mark_quota_exhausted(session: AsyncSession, source: str, at: datetime) -> None:
    values = {"source": source, "exhausted_at": at}
    statement = (
        insert(LLMQuotaStatus)
        .values(id=_QUOTA_ROW_ID, **values)
        .on_conflict_do_update(index_elements=[LLMQuotaStatus.id], set_=values)
    )
    await session.execute(statement)


async def clear_quota_exhausted(session: AsyncSession) -> None:
    await session.execute(
        update(LLMQuotaStatus)
        .where(LLMQuotaStatus.id == _QUOTA_ROW_ID, LLMQuotaStatus.exhausted_at.is_not(None))
        .values(exhausted_at=None)
    )
