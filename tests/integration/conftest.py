"""Integration fixtures: a real Postgres + pgvector, migrated with Alembic.

Set TEST_DATABASE_URL to a disposable database (CI uses a pgvector service
container). Without it these tests are skipped locally and fail in CI. The schema is dropped and
re-created, so never point it at Neon or any database you care about.
"""

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.session import create_engine, session_factory

ROOT = Path(__file__).resolve().parents[2]
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")


async def _reset_schema(url: str) -> None:
    engine = create_engine(url)
    async with engine.begin() as connection:
        await connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        await connection.execute(text("CREATE SCHEMA public"))
    await engine.dispose()


@pytest.fixture(scope="session")
def migrated_database() -> str:
    """Fresh schema built by running every Alembic migration once per test session."""
    if not TEST_DATABASE_URL:
        if os.environ.get("CI"):
            pytest.fail("TEST_DATABASE_URL must be set in CI; integration tests may not skip")
        pytest.skip("TEST_DATABASE_URL is not set")
    asyncio.run(_reset_schema(TEST_DATABASE_URL))
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "alembic"))
    config.attributes["database_url"] = TEST_DATABASE_URL
    config.attributes["configure_logger"] = False
    command.upgrade(config, "head")
    return TEST_DATABASE_URL


@pytest.fixture
async def sessions(migrated_database: str) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Session factory over empty tables (truncated before each test)."""
    engine = create_engine(migrated_database)
    async with engine.begin() as connection:
        await connection.execute(text("TRUNCATE passages, verses, books RESTART IDENTITY CASCADE"))
    yield session_factory(engine)
    await engine.dispose()
