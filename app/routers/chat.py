"""`POST /api/v1/chat`: grounded answers streamed as Server-Sent Events."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.dependencies import get_embedder, get_guard, get_llm, get_session, get_settings_dep
from app.llm import Embedder, LLMProvider
from app.routers.protection import RATE_LIMIT_RESPONSES, enforce_rate_limit, to_sse
from app.services import chat as chat_service
from app.services.guard import AnswerEntry, DemoGuard, StreamEvent

router = APIRouter(prefix="/api/v1", tags=["chat"])


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    history: list[HistoryMessage] = Field(
        default_factory=list[HistoryMessage],
        max_length=chat_service.MAX_HISTORY,
        description="Previous turns, oldest first (at most the last 6 messages).",
    )


@dataclass(frozen=True, slots=True)
class ChatPlan:
    request: ChatRequest
    cached: list[StreamEvent] | None  # a complete earlier answer to replay
    entry: AnswerEntry | None  # where to cache a fresh answer (None with history)


async def plan_chat(
    request: ChatRequest,
    http_request: Request,
    llm: Annotated[LLMProvider, Depends(get_llm)],
    guard: Annotated[DemoGuard, Depends(get_guard)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> ChatPlan:
    """Cache first (a hit costs nothing and is not rate limited), then the rate limit.

    Runs as a dependency so a 429 is sent before the event stream starts.
    """
    entry = None
    if not request.history:  # with history the answer depends on the conversation
        entry = AnswerEntry.build("chat", request.question, llm.name, llm.chat_model)
        cached = await guard.cached_answer(entry.key)
        if cached is not None:
            return ChatPlan(request, cached, entry)
    await enforce_rate_limit(http_request, guard, settings)
    return ChatPlan(request, None, entry)


@router.post(
    "/chat",
    response_class=EventSourceResponse,
    summary="Ask a question; the grounded answer streams as Server-Sent Events",
    responses=RATE_LIMIT_RESPONSES,
)
async def chat(
    plan: Annotated[ChatPlan, Depends(plan_chat)],
    session: Annotated[AsyncSession, Depends(get_session)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    llm: Annotated[LLMProvider, Depends(get_llm)],
    guard: Annotated[DemoGuard, Depends(get_guard)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> AsyncIterator[ServerSentEvent]:
    """Events, in order: `sources` (retrieved passages), `token` (answer fragments),
    `citations` (`valid` and `discarded`, checked in code against the sources) and
    `done`. Off-topic questions get a single `error` event with code `no_results`
    without calling the LLM; failures end the stream with `error`
    (`quota_exceeded` or `internal`).

    A question asked before (same normalized text, no history, same model) is
    replayed from the cache without calling the LLM or counting against the rate
    limit; its `done` event carries `cached: true`."""
    if plan.cached is not None:
        for event in plan.cached:
            yield to_sse(event)
        return

    history = [chat_service.HistoryTurn(m.role, m.content) for m in plan.request.history]
    live = (
        chat_service.to_stream_event(event)
        async for event in chat_service.chat_events(
            session,
            embedder,
            llm,
            plan.request.question,
            history,
            top_k=settings.chat_top_k,
            min_score=settings.chat_min_score,
        )
    )
    async for event in guard.track(live, source=llm.name, entry=plan.entry):
        yield to_sse(event)
