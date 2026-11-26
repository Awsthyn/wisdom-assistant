from app.db.session import engine_config

NEON_DIRECT = (
    "postgresql://user:pw@ep-x-123.us-east-2.aws.neon.tech/neondb"
    "?sslmode=require&channel_binding=require"
)
NEON_POOLED = NEON_DIRECT.replace("ep-x-123", "ep-x-123-pooler")


def test_neon_url_is_converted_for_asyncpg() -> None:
    """Protects Neon connectivity; rejects passing libpq-only params that asyncpg refuses."""
    config = engine_config(NEON_DIRECT)

    assert config.url.drivername == "postgresql+asyncpg"
    assert "sslmode" not in config.url.query
    assert "channel_binding" not in config.url.query
    assert config.connect_args == {"ssl": "require"}


def test_pooled_url_disables_prepared_statement_caches() -> None:
    """Protects PgBouncer transaction mode; rejects statement caches that break pooled hosts."""
    config = engine_config(NEON_POOLED)

    assert config.connect_args["statement_cache_size"] == 0
    assert config.url.query["prepared_statement_cache_size"] == "0"


def test_quoted_local_url_keeps_plain_connection() -> None:
    """Protects .env values in quotes and local DBs; rejects forcing SSL where none was asked."""
    config = engine_config('"postgresql+asyncpg://postgres:postgres@localhost:5432/test"')

    assert config.url.host == "localhost"
    assert config.connect_args == {}
