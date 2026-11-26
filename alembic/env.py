"""Alembic environment (async, asyncpg).

The database URL is taken, in order, from:
1. `config.attributes["database_url"]` (set programmatically, e.g. by tests),
2. DATABASE_URL_DIRECT, then DATABASE_URL (via app settings).
Migrations should use the direct (non-pooled) Neon URL.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection

from app.core.config import get_settings
from app.db.models import Base
from app.db.session import create_engine

config = context.config

if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    explicit = config.attributes.get("database_url")
    if isinstance(explicit, str) and explicit:
        return explicit
    settings = get_settings()
    secret = settings.database_url_direct or settings.database_url
    if secret is None or not secret.get_secret_value():
        raise RuntimeError("Set DATABASE_URL_DIRECT (or DATABASE_URL) to run migrations")
    return secret.get_secret_value()


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    engine = create_engine(_database_url())
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
