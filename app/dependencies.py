"""Process-wide resources (DB engine, LLM provider) and their FastAPI dependencies.

Resources are created lazily on first use, so the app starts (and `/health`
answers) even when the database or the LLM are not configured, and an exact
reference search never needs an LLM key. Tests replace the dependencies with
`app.dependency_overrides`.

The demo guard (rate limit, caches, quota status) shares the API's database; it
is disabled when DATABASE_URL is not set, which also makes every database-backed
endpoint answer 503, so no LLM call can happen unguarded.
"""

from collections.abc import AsyncIterator, Sequence
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.session import create_engine, session_factory
from app.llm import Embedder, EmbedTask, LLMConfigError, LLMProvider, build_llm_provider
from app.services.guard import CachedQueryEmbedder, DemoGuard


class NotConfiguredError(RuntimeError):
    """A resource the request needs is missing from the environment."""


class AppResources:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._engine: AsyncEngine | None = None
        self._sessions: async_sessionmaker[AsyncSession] | None = None
        self._llm: LLMProvider | None = None
        self._guard: DemoGuard | None = None

    def sessions(self) -> async_sessionmaker[AsyncSession]:
        if self._sessions is None:
            if self.settings.database_url is None:
                raise NotConfiguredError("DATABASE_URL is not set")
            self._engine = create_engine(self.settings.database_url.get_secret_value())
            self._sessions = session_factory(self._engine)
        return self._sessions

    def guard(self) -> DemoGuard:
        if self._guard is None:
            try:
                sessions = self.sessions()
            except NotConfiguredError:
                sessions = None
            self._guard = DemoGuard.from_settings(sessions, self.settings)
        return self._guard

    def llm(self) -> LLMProvider:
        if self._llm is None:
            self._llm = build_llm_provider(self.settings)
        return self._llm

    async def aclose(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()


class LazyEmbedder:
    """`Embedder` that builds the LLM provider only when an embedding is requested."""

    def __init__(self, resources: AppResources) -> None:
        self._resources = resources

    @property
    def embed_model(self) -> str:
        return self._resources.llm().embed_model

    async def embed(self, texts: Sequence[str], task: EmbedTask) -> list[list[float]]:
        return await self._resources.llm().embed(texts, task)


def _resources(request: Request) -> AppResources:
    resources: AppResources = request.app.state.resources
    return resources


async def get_session(
    resources: Annotated[AppResources, Depends(_resources)],
) -> AsyncIterator[AsyncSession]:
    try:
        sessions = resources.sessions()
    except NotConfiguredError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    async with sessions() as session:
        yield session


def get_guard(resources: Annotated[AppResources, Depends(_resources)]) -> DemoGuard:
    return resources.guard()


def get_base_embedder(resources: Annotated[AppResources, Depends(_resources)]) -> Embedder:
    """The provider's embedder, without the query-embedding cache."""
    return LazyEmbedder(resources)


def get_embedder(
    base: Annotated[Embedder, Depends(get_base_embedder)],
    guard: Annotated[DemoGuard, Depends(get_guard)],
) -> Embedder:
    """Query embeddings go through the cache (same normalized text, same vector)."""
    return CachedQueryEmbedder(base, guard)


def get_llm(resources: Annotated[AppResources, Depends(_resources)]) -> LLMProvider:
    try:
        return resources.llm()
    except LLMConfigError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


def get_settings_dep(resources: Annotated[AppResources, Depends(_resources)]) -> Settings:
    """The settings this app was created with (not the process-wide cached ones)."""
    return resources.settings
