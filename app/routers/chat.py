"""`POST /api/v1/chat`: grounded answers streamed as Server-Sent Events."""

from collections.abc import AsyncIterator
from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.dependencies import get_embedder, get_llm, get_session, get_settings_dep
from app.llm import Embedder, LLMProvider
from app.services import chat as chat_service
from app.services.citations import Citation

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


class SourcePassage(BaseModel):
    id: int  # the number the passage has in the prompt
    reference: str
    book: str
    chapter: int
    start_verse: int
    end_verse: int
    text: str
    url: str
    score: float | None


class CitationOut(BaseModel):
    cited: str
    reference: str
    book: str | None
    url: str | None
    reason: str | None = None


def _citation(c: Citation) -> CitationOut:
    return CitationOut(
        cited=c.cited, reference=c.reference, book=c.book, url=c.url, reason=c.reason
    )


def _to_sse(event: chat_service.ChatEvent) -> ServerSentEvent:
    match event:
        case chat_service.SourcesEvent(passages=passages):
            data = {
                "passages": [
                    SourcePassage(
                        id=n,
                        reference=p.reference,
                        book=p.book.osis_code,
                        chapter=p.chapter,
                        start_verse=p.start_verse,
                        end_verse=p.end_verse,
                        text=p.text,
                        url=p.url,
                        score=p.score,
                    ).model_dump()
                    for n, p in enumerate(passages, start=1)
                ]
            }
            return ServerSentEvent(event="sources", data=data)
        case chat_service.TokenEvent(text=text):
            return ServerSentEvent(event="token", data={"text": text})
        case chat_service.CitationsEvent(report=report):
            data = {
                "valid": [_citation(c).model_dump() for c in report.valid],
                "discarded": [_citation(c).model_dump() for c in report.discarded],
            }
            return ServerSentEvent(event="citations", data=data)
        case chat_service.DoneEvent(model=model):
            return ServerSentEvent(event="done", data={"model": model})
        case chat_service.ErrorEvent(code=code, message=message):
            return ServerSentEvent(event="error", data={"code": code, "message": message})


@router.post(
    "/chat",
    response_class=EventSourceResponse,
    summary="Ask a question; the grounded answer streams as Server-Sent Events",
)
async def chat(
    request: ChatRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    llm: Annotated[LLMProvider, Depends(get_llm)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> AsyncIterator[ServerSentEvent]:
    """Events, in order: `sources` (retrieved passages), `token` (answer fragments),
    `citations` (`valid` and `discarded`, checked in code against the sources) and
    `done`. Off-topic questions get a single `error` event with code `no_results`
    without calling the LLM; failures end the stream with `error`
    (`quota_exceeded` or `internal`)."""
    history = [chat_service.HistoryTurn(m.role, m.content) for m in request.history]
    async for event in chat_service.chat_events(
        session,
        embedder,
        llm,
        request.question,
        history,
        top_k=settings.chat_top_k,
        min_score=settings.chat_min_score,
    ):
        yield _to_sse(event)
