"""Async engine and session factories, with Neon-friendly URL handling."""

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# libpq-only query parameters that asyncpg rejects; SSL is passed via connect_args.
_LIBPQ_ONLY_PARAMS = ("sslmode", "channel_binding")
_SSL_MODES = {"require", "verify-ca", "verify-full"}


@dataclass(frozen=True, slots=True)
class EngineConfig:
    url: URL
    connect_args: dict[str, Any] = field(default_factory=dict[str, Any])


def engine_config(raw_url: str) -> EngineConfig:
    """Turn a libpq-style URL (as Neon shows it) into an asyncpg URL plus connect args.

    - `postgres://` / `postgresql://` become `postgresql+asyncpg://`.
    - `sslmode=require` becomes asyncpg's `ssl="require"`; `channel_binding` is dropped.
    - Pooled Neon hosts (`-pooler`) run PgBouncer in transaction mode, which breaks
      prepared-statement caches, so both asyncpg's and SQLAlchemy's are disabled.
    """
    url = make_url(raw_url.strip().strip('"'))
    if url.drivername in ("postgres", "postgresql", "postgresql+psycopg", "postgresql+psycopg2"):
        url = url.set(drivername="postgresql+asyncpg")

    query = dict(url.query)
    sslmode = query.get("sslmode")
    url = url.difference_update_query(_LIBPQ_ONLY_PARAMS)

    connect_args: dict[str, Any] = {}
    if isinstance(sslmode, str) and sslmode in _SSL_MODES:
        connect_args["ssl"] = "require"
    if url.host and "-pooler" in url.host:
        connect_args["statement_cache_size"] = 0
        url = url.update_query_dict({"prepared_statement_cache_size": "0"})
    return EngineConfig(url=url, connect_args=connect_args)


def create_engine(raw_url: str) -> AsyncEngine:
    config = engine_config(raw_url)
    return create_async_engine(
        config.url, connect_args=config.connect_args, pool_pre_ping=True, pool_size=5
    )


def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)
