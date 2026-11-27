"""FastAPI application entry point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import Settings, get_settings
from app.dependencies import AppResources
from app.routers import agent, chat, health, search

DESCRIPTION = """
Study assistant for the five wisdom books of the World English Bible
(Job, Psalms, Proverbs, Ecclesiastes, Song of Songs).

Semantic search, RAG chat with verified verse citations, a tool-calling agent
and an MCP server.

Scripture text: World English Bible (ebible.org), public domain.
Cross references: OpenBible.info, CC-BY.
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app. Accepts explicit settings so tests can inject their own."""
    settings = settings or get_settings()
    resources = AppResources(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        yield
        await resources.aclose()

    app = FastAPI(
        title="wisdom-assistant",
        summary="RAG study assistant for the wisdom books of the World English Bible.",
        description=DESCRIPTION,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.resources = resources
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
    app.include_router(health.router)
    app.include_router(search.router)
    app.include_router(chat.router)
    app.include_router(agent.router)
    return app


app = create_app()
