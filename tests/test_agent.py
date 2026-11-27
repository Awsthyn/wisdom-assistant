"""Agent loop with the fake provider, scripted tool calls and stub tools (no database)."""

from collections.abc import Sequence
from typing import Any, cast

from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.books import BIBLE_BOOKS
from app.llm import (
    LLMResponse,
    Message,
    QuotaExceededError,
    ToolCall,
    ToolResultMessage,
    ToolSpec,
    UserMessage,
)
from app.llm.fake import FakeProvider
from app.services import agent as agent_service
from app.services.agent import AgentEvent, agent_events
from app.services.search import SearchHit
from app.tools import Tool, ToolContext, ToolError, ToolOutput

_BOOKS = {b.osis_code: b for b in BIBLE_BOOKS}


def _hit(code: str, chapter: int, start: int, end: int, *, in_corpus: bool) -> SearchHit:
    name = _BOOKS[code].reference_name
    label = f"{name} {chapter}:{start}" + (f"-{end}" if end != start else "")
    return SearchHit(_BOOKS[code], chapter, start, end, label, "...", in_corpus, None)


PSALM_23 = _hit("Ps", 23, 1, 4, in_corpus=True)
JOHN_10 = _hit("John", 10, 11, 11, in_corpus=False)


class _RefArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reference: str


class _Stubs:
    """Stub tools named like the real ones, recording every call in order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def tool(self, name: str, passages: tuple[SearchHit, ...]) -> Tool[_RefArgs]:
        async def handler(_: ToolContext, args: _RefArgs) -> ToolOutput:
            self.calls.append((name, args.reference))
            if args.reference == "boom":
                raise ToolError("Unknown reference 'boom'")
            return ToolOutput({"reference": args.reference}, f"{name} ok", passages)

        return Tool(name, f"stub {name}", _RefArgs, handler)

    @property
    def tools(self) -> list[Tool[Any]]:
        return [
            self.tool("search_passages", (PSALM_23,)),
            self.tool("lookup_reference", (PSALM_23,)),
            self.tool("cross_references", (JOHN_10,)),
        ]


CONTEXT = ToolContext(session=cast(AsyncSession, None), embedder=FakeProvider())


def _calls(*calls: tuple[str, dict[str, Any]]) -> LLMResponse:
    return LLMResponse(
        text="",
        tool_calls=tuple(ToolCall(f"c{i}", name, args) for i, (name, args) in enumerate(calls)),
    )


async def _run(provider: FakeProvider, stubs: _Stubs, **limits: int) -> list[AgentEvent]:
    question = "God as a shepherd?"
    return [e async for e in agent_events(provider, stubs.tools, CONTEXT, question, **limits)]


def _names(events: list[AgentEvent]) -> list[str]:
    return [type(e).__name__ for e in events]


def _tool_results(provider: FakeProvider, call_index: int) -> ToolResultMessage:
    last = provider.calls[call_index].messages[-1]
    assert isinstance(last, ToolResultMessage)
    return last


async def test_tools_run_in_the_order_the_model_asked() -> None:
    """Protects the loop; rejects reordered, skipped or duplicated tool executions."""
    stubs = _Stubs()
    provider = FakeProvider(
        tool_responses=[
            _calls(
                ("search_passages", {"reference": "shepherd"}),
                ("lookup_reference", {"reference": "Psalm 23:1"}),
            ),
            _calls(("cross_references", {"reference": "Psalm 23:1"})),
            LLMResponse(text="The LORD is a shepherd [Psalm 23:1]."),
        ]
    )

    events = await _run(provider, stubs)

    assert stubs.calls == [
        ("search_passages", "shepherd"),
        ("lookup_reference", "Psalm 23:1"),
        ("cross_references", "Psalm 23:1"),
    ]
    assert _names(events) == [
        "ToolCallEvent",
        "ToolResultEvent",
        "ToolCallEvent",
        "ToolResultEvent",
        "ToolCallEvent",
        "ToolResultEvent",
        "TokenEvent",
        "CitationsEvent",
        "DoneEvent",
    ]
    done = events[-1]
    assert isinstance(done, agent_service.DoneEvent)
    assert (done.iterations, done.tool_calls) == (3, 3)
    # Results go back to the model in call order, matched by call id.
    assert [r.call_id for r in _tool_results(provider, 1).results] == ["c0", "c1"]


async def test_iteration_limit_stops_a_model_that_never_answers() -> None:
    """Protects against endless loops; rejects more than 5 model calls."""
    stubs = _Stubs()
    provider = FakeProvider(
        tool_responses=[_calls(("search_passages", {"reference": f"q{i}"})) for i in range(10)]
    )

    events = await _run(provider, stubs)

    assert len(provider.calls) == agent_service.MAX_ITERATIONS
    assert len(stubs.calls) == agent_service.MAX_ITERATIONS - 1
    last = events[-1]
    assert isinstance(last, agent_service.ErrorEvent)
    assert last.code == "iteration_limit"
    # The model was told on its last turn that no more tools are possible.
    assert "no more tool calls" in _tool_results(provider, -1).results[0].content


async def test_tool_call_limit_caps_executions() -> None:
    """Protects the 8-call budget; rejects executing every call of a greedy model."""
    stubs = _Stubs()
    provider = FakeProvider(
        tool_responses=[
            _calls(*(("search_passages", {"reference": f"q{i}"}) for i in range(10))),
            LLMResponse(text="Done."),
        ]
    )

    events = await _run(provider, stubs)

    assert len(stubs.calls) == agent_service.MAX_TOOL_CALLS
    results = _tool_results(provider, 1).results
    assert len(results) == 10  # every call gets an answer, executed or not
    assert [r.is_error for r in results[8:]] == [True, True]
    assert "limit of tool calls" in results[9].content
    assert isinstance(events[-1], agent_service.DoneEvent)


async def test_invalid_arguments_go_back_to_the_model() -> None:
    """Protects robustness; rejects crashing on bad arguments instead of reporting them."""
    stubs = _Stubs()
    provider = FakeProvider(
        tool_responses=[
            _calls(
                ("lookup_reference", {"ref": "Psalm 23"}),
                ("lookup_reference", {"reference": "boom"}),
                ("no_such_tool", {}),
            ),
            LLMResponse(text="I could not find it."),
        ]
    )

    events = await _run(provider, stubs)

    results = _tool_results(provider, 1).results
    assert [r.is_error for r in results] == [True, True, True]
    assert "Invalid arguments for lookup_reference" in results[0].content
    assert "Unknown reference 'boom'" in results[1].content
    assert "Unknown tool 'no_such_tool'" in results[2].content
    summaries = [e for e in events if isinstance(e, agent_service.ToolResultEvent)]
    assert [s.is_error for s in summaries] == [True, True, True]
    assert isinstance(events[-1], agent_service.DoneEvent)


async def test_citations_are_checked_against_tool_texts() -> None:
    """Protects post-generation verification; rejects a cited verse no tool returned."""
    stubs = _Stubs()
    provider = FakeProvider(
        tool_responses=[
            _calls(("lookup_reference", {"reference": "Psalm 23:1-4"})),
            LLMResponse(text="Shepherd [Psalm 23:1]; also [Psalm 80:1] and [Proverbs 3:5]."),
        ]
    )

    events = await _run(provider, stubs)

    [report] = [e.report for e in events if isinstance(e, agent_service.CitationsEvent)]
    assert [c.reference for c in report.valid] == ["Psalm 23:1"]
    assert [c.reference for c in report.discarded] == ["Psalm 80:1", "Proverbs 3:5"]


async def test_outside_corpus_text_from_cross_references_is_valid_and_flagged() -> None:
    """Protects agent cross references; rejects discarding John 10:11 or hiding it is non-corpus."""
    stubs = _Stubs()
    provider = FakeProvider(
        tool_responses=[
            _calls(("cross_references", {"reference": "Psalm 23:1"})),
            LLMResponse(text="It reappears in John [John 10:11]. Not [John 3:16]."),
        ]
    )

    events = await _run(provider, stubs)

    [report] = [e.report for e in events if isinstance(e, agent_service.CitationsEvent)]
    assert [(c.reference, c.outside_corpus) for c in report.valid] == [("John 10:11", True)]
    assert [(c.reference, c.reason) for c in report.discarded] == [
        ("John 3:16", "not in the retrieved passages")
    ]


async def test_system_prompt_and_tools_are_sent_and_question_is_delimited() -> None:
    """Protects the agent principles; rejects a missing system prompt, tools or delimiters."""
    stubs = _Stubs()
    provider = FakeProvider(tool_responses=[LLMResponse(text="No tools needed.")])

    await _run(provider, stubs)

    [call] = provider.calls
    assert call.system == agent_service.AGENT_SYSTEM_PROMPT
    assert [t.name for t in call.tools] == [t.name for t in stubs.tools]
    first = call.messages[0]
    assert isinstance(first, UserMessage)
    assert first.content.startswith("<question>")


class _QuotaProvider(FakeProvider):
    async def generate_with_tools(
        self, messages: Sequence[Message], system: str | None, tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        raise QuotaExceededError("429")


async def test_quota_becomes_an_error_event() -> None:
    """Protects the free-tier path; rejects an exception escaping the agent stream."""
    events = await _run(_QuotaProvider(), _Stubs())

    assert len(events) == 1
    assert isinstance(events[0], agent_service.ErrorEvent)
    assert events[0].code == "quota_exceeded"
