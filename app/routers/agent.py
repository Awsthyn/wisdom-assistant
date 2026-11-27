"""`POST /api/v1/agent`: tool-calling agent, streamed as Server-Sent Events."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.dependencies import get_embedder, get_guard, get_llm, get_session, get_settings_dep
from app.llm import Embedder, LLMProvider
from app.routers.protection import RATE_LIMIT_RESPONSES, enforce_rate_limit, to_sse
from app.services import agent as agent_service
from app.services.guard import AnswerEntry, DemoGuard, StreamEvent
from app.tools import BIBLE_TOOLS, ToolContext

router = APIRouter(prefix="/api/v1", tags=["agent"])


class AgentRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@dataclass(frozen=True, slots=True)
class AgentPlan:
    request: AgentRequest
    cached: list[StreamEvent] | None
    entry: AnswerEntry


async def plan_agent(
    request: AgentRequest,
    http_request: Request,
    llm: Annotated[LLMProvider, Depends(get_llm)],
    guard: Annotated[DemoGuard, Depends(get_guard)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> AgentPlan:
    """Cache first (not rate limited), then the rate limit, before the stream starts."""
    entry = AnswerEntry.build("agent", request.question, llm.name, llm.chat_model)
    cached = await guard.cached_answer(entry.key)
    if cached is None:
        await enforce_rate_limit(http_request, guard, settings)
    return AgentPlan(request, cached, entry)


@router.post(
    "/agent",
    response_class=EventSourceResponse,
    summary="Ask the tool-calling agent; its steps and answer stream as Server-Sent Events",
    responses=RATE_LIMIT_RESPONSES,
)
async def agent(
    plan: Annotated[AgentPlan, Depends(plan_agent)],
    session: Annotated[AsyncSession, Depends(get_session)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    llm: Annotated[LLMProvider, Depends(get_llm)],
    guard: Annotated[DemoGuard, Depends(get_guard)],
) -> AsyncIterator[ServerSentEvent]:
    """Events: `tool_call` (name, arguments) and `tool_result` (short summary) for each
    step, then `token` (the answer), `citations` (`valid` / `discarded`; texts outside
    the wisdom books are valid only if a tool returned them and carry
    `outside_corpus: true`) and `done`. Failures end with `error` (`quota_exceeded`,
    `internal` or `iteration_limit`). At most 5 model calls and 8 tool calls.

    Repeated questions are replayed from the cache (`done.cached: true`) without
    calling the LLM or counting against the rate limit."""
    if plan.cached is not None:
        for event in plan.cached:
            yield to_sse(event)
        return

    context = ToolContext(session=session, embedder=embedder)
    live = (
        agent_service.to_stream_event(event)
        async for event in agent_service.agent_events(
            llm, BIBLE_TOOLS, context, plan.request.question
        )
    )
    async for event in guard.track(live, source=llm.name, entry=plan.entry):
        yield to_sse(event)
