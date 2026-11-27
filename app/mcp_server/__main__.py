"""Run the MCP server over stdio: `uv run python -m app.mcp_server`.

Reads DATABASE_URL (and GEMINI_API_KEY, used only to embed search_passages
queries) from the environment or `.env`. stdout carries the MCP protocol, so
nothing else may be printed to it; errors go to stderr.
"""

import sys

from app.core.config import get_settings
from app.db.session import create_engine, session_factory
from app.llm import build_embedder
from app.mcp_server import LazyEmbedder, create_mcp_server


def main() -> int:
    settings = get_settings()
    if settings.database_url is None or not settings.database_url.get_secret_value():
        print("DATABASE_URL is not set; see .env.example", file=sys.stderr)
        return 1
    engine = create_engine(settings.database_url.get_secret_value())
    server = create_mcp_server(
        session_factory(engine), LazyEmbedder(lambda: build_embedder(settings))
    )
    server.run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
