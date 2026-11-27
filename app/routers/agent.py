"""`POST /api/v1/agent`: tool-calling agent, streamed as Server-Sent Events."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_embedder, get_llm, get_session
from app.llm import Embedder, LLMProvider
from app.services import agent as agent_service
from app.services.citations import Citation
from app.tools import BIBLE_TOOLS, ToolContext

router = APIRouter(prefix="/api/v1", tags=["agent"])


class AgentRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


def _citation(c: Citation) -> dict[str, object]:
    return {
        "cited": c.cited,
        "reference": c.reference,
        "book": c.book,
        "url": c.url,
        "outside_corpus": c.outside_corpus,
        "reason": c.reason,
    }


def _to_sse(event: agent_service.AgentEvent) -> ServerSentEvent:
    match event:
        case agent_service.ToolCallEvent(name=name, arguments=arguments):
            return ServerSentEvent(
                event="tool_call", data={"name": name, "arguments": dict(arguments)}
            )
        case agent_service.ToolResultEvent(name=name, summary=summary, is_error=is_error):
            return ServerSentEvent(
                event="tool_result", data={"name": name, "summary": summary, "is_error": is_error}
            )
        case agent_service.TokenEvent(text=text):
            return ServerSentEvent(event="token", data={"text": text})
        case agent_service.CitationsEvent(report=report):
            return ServerSentEvent(
                event="citations",
                data={
                    "valid": [_citation(c) for c in report.valid],
                    "discarded": [_citation(c) for c in report.discarded],
                },
            )
        case agent_service.DoneEvent(model=model, iterations=iterations, tool_calls=tool_calls):
            return ServerSentEvent(
                event="done",
                data={"model": model, "iterations": iterations, "tool_calls": tool_calls},
            )
        case agent_service.ErrorEvent(code=code, message=message):
            return ServerSentEvent(event="error", data={"code": code, "message": message})


@router.post(
    "/agent",
    response_class=EventSourceResponse,
    summary="Ask the tool-calling agent; its steps and answer stream as Server-Sent Events",
)
async def agent(
    request: AgentRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    llm: Annotated[LLMProvider, Depends(get_llm)],
) -> AsyncIterator[ServerSentEvent]:
    """Events: `tool_call` (name, arguments) and `tool_result` (short summary) for each
    step, then `token` (the answer), `citations` (`valid` / `discarded`; texts outside
    the wisdom books are valid only if a tool returned them and carry
    `outside_corpus: true`) and `done`. Failures end with `error` (`quota_exceeded`,
    `internal` or `iteration_limit`). At most 5 model calls and 8 tool calls."""
    context = ToolContext(session=session, embedder=embedder)
    async for event in agent_service.agent_events(llm, BIBLE_TOOLS, context, request.question):
        yield _to_sse(event)
