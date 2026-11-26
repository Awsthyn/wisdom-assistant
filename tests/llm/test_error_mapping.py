"""SDK exceptions -> LLM-layer exceptions, at the mapping and at the provider level."""

from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types import Message as AnthropicMessage
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.llm import retry
from app.llm.anthropic_provider import AnthropicProvider, map_anthropic_error
from app.llm.base import (
    AssistantMessage,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    ToolSpec,
    UserMessage,
)
from app.llm.errors import (
    LLMError,
    LLMResponseError,
    QuotaExceededError,
    TransientLLMError,
)
from app.llm.fake import FakeProvider
from app.llm.gemini import GeminiProvider, map_gemini_error

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


@pytest.fixture(autouse=True)
def _instant_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_delay(attempt: int) -> float:
        return 0.0

    monkeypatch.setattr(retry, "backoff_delay", no_delay)


def _anthropic_status_error(status: int) -> anthropic.APIStatusError:
    request = httpx2.Request("POST", ANTHROPIC_URL)
    response = httpx2.Response(status, request=request)
    cls: type[anthropic.APIStatusError] = {
        429: anthropic.RateLimitError,
        500: anthropic.InternalServerError,
        400: anthropic.BadRequestError,
    }.get(status, anthropic.APIStatusError)
    return cls(f"status {status}", response=response, body=None)


def _gemini_error(code: int, status: str) -> genai_errors.APIError:
    body = {"error": {"code": code, "message": f"{status} from test", "status": status}}
    cls = genai_errors.ClientError if code < 500 else genai_errors.ServerError
    return cls(code, body)


def _gemini_provider(client: genai.Client) -> GeminiProvider:
    return GeminiProvider(
        api_key="fake", chat_model="gemini-test", embed_model="emb", client=client
    )


def _anthropic_provider(client: anthropic.AsyncAnthropic) -> AnthropicProvider:
    return AnthropicProvider(
        api_key="fake", model="claude-test", embedder=FakeProvider(), client=client
    )


# --- Anthropic mapping -------------------------------------------------------


def test_anthropic_429_maps_to_quota_exceeded() -> None:
    """Protects quota detection; rejects treating an Anthropic 429 as retryable or generic."""
    assert isinstance(map_anthropic_error(_anthropic_status_error(429)), QuotaExceededError)


@pytest.mark.parametrize("status", [500, 503, 529])
def test_anthropic_5xx_maps_to_transient(status: int) -> None:
    """Protects retries on server errors; rejects a 5xx/529 that would not be retried."""
    assert isinstance(map_anthropic_error(_anthropic_status_error(status)), TransientLLMError)


def test_anthropic_timeout_maps_to_transient() -> None:
    """Protects retries on timeouts; rejects an APITimeoutError that escapes unmapped."""
    error = anthropic.APITimeoutError(request=httpx2.Request("POST", ANTHROPIC_URL))
    assert isinstance(map_anthropic_error(error), TransientLLMError)


def test_anthropic_400_maps_to_non_retryable_error() -> None:
    """Protects against retrying bad requests; rejects a 400 mapped as transient or quota."""
    mapped = map_anthropic_error(_anthropic_status_error(400))
    assert type(mapped) is LLMError


# --- Gemini mapping ----------------------------------------------------------


def test_gemini_resource_exhausted_maps_to_quota_exceeded() -> None:
    """Protects quota detection; rejects treating Gemini RESOURCE_EXHAUSTED as retryable."""
    error = _gemini_error(429, "RESOURCE_EXHAUSTED")
    assert isinstance(map_gemini_error(error), QuotaExceededError)


def test_gemini_503_maps_to_transient() -> None:
    """Protects retries on Gemini overload; rejects a 503 that would not be retried."""
    assert isinstance(map_gemini_error(_gemini_error(503, "UNAVAILABLE")), TransientLLMError)


def test_gemini_transport_timeout_maps_to_transient() -> None:
    """Protects retries on network timeouts; rejects an httpx2 timeout escaping unmapped."""
    assert isinstance(map_gemini_error(httpx2.ReadTimeout("slow")), TransientLLMError)


def test_gemini_400_maps_to_non_retryable_error() -> None:
    """Protects against retrying bad requests; rejects a 400 mapped as transient or quota."""
    assert type(map_gemini_error(_gemini_error(400, "INVALID_ARGUMENT"))) is LLMError


def test_unrelated_exceptions_are_not_mapped() -> None:
    """Protects real bugs from being masked; rejects swallowing e.g. a ValueError as an LLMError."""
    assert map_gemini_error(ValueError("bug")) is None
    assert map_anthropic_error(ValueError("bug")) is None


# --- Provider level ----------------------------------------------------------


async def test_gemini_provider_raises_quota_without_retrying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protects the free-tier quota; rejects retrying a 429 or leaking the SDK exception."""
    client = genai.Client(api_key="fake")
    calls = 0

    async def generate_content(**_: Any) -> types.GenerateContentResponse:
        nonlocal calls
        calls += 1
        raise _gemini_error(429, "RESOURCE_EXHAUSTED")

    monkeypatch.setattr(client.aio.models, "generate_content", generate_content)

    with pytest.raises(QuotaExceededError):
        await _gemini_provider(client).generate([UserMessage("hi")])
    assert calls == 1


async def test_gemini_provider_retries_5xx_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Protects resilience to blips; rejects failing on a 503 that a retry would have fixed."""
    client = genai.Client(api_key="fake")
    calls = 0

    async def generate_content(**_: Any) -> types.GenerateContentResponse:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise _gemini_error(503, "UNAVAILABLE")
        return types.GenerateContentResponse(
            candidates=[
                types.Candidate(content=types.Content(role="model", parts=[types.Part(text="ok")]))
            ]
        )

    monkeypatch.setattr(client.aio.models, "generate_content", generate_content)

    assert await _gemini_provider(client).generate([UserMessage("hi")]) == "ok"
    assert calls == 3


async def test_anthropic_provider_gives_up_after_three_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protects the retry cap; rejects more than 1 call + 3 retries on persistent 5xx."""
    client = anthropic.AsyncAnthropic(api_key="fake", max_retries=0)
    calls = 0

    async def create(**_: Any) -> AnthropicMessage:
        nonlocal calls
        calls += 1
        raise _anthropic_status_error(500)

    monkeypatch.setattr(client.messages, "create", create)

    with pytest.raises(TransientLLMError):
        await _anthropic_provider(client).generate([UserMessage("hi")])
    assert calls == 4


async def test_anthropic_tool_call_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Protects the tool-call contract; rejects losing call ids/args or the replayed native turn."""
    client = anthropic.AsyncAnthropic(api_key="fake", max_retries=0)
    sent: list[dict[str, Any]] = []
    tool_turn = AnthropicMessage.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": [
                {"type": "text", "text": "Looking it up."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "lookup_reference",
                    "input": {"reference": "Prov 3:5"},
                },
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    final_turn = AnthropicMessage.model_validate(
        {
            **tool_turn.model_dump(),
            "content": [{"type": "text", "text": "Done."}],
            "stop_reason": "end_turn",
        }
    )
    replies = [tool_turn, final_turn]

    async def create(**kwargs: Any) -> AnthropicMessage:
        sent.append(kwargs)
        return replies.pop(0)

    monkeypatch.setattr(client.messages, "create", create)
    provider = _anthropic_provider(client)
    tool = ToolSpec("lookup_reference", "Look up a verse", {"type": "object"})

    first = await provider.generate_with_tools([UserMessage("Prov 3:5?")], "sys", [tool])
    [call] = first.tool_calls
    assert (call.id, call.name, call.arguments) == (
        "toolu_1",
        "lookup_reference",
        {"reference": "Prov 3:5"},
    )

    history = [
        UserMessage("Prov 3:5?"),
        first.to_message(),
        ToolResultMessage((ToolResult(call.id, call.name, "Trust in the LORD..."),)),
    ]
    second = await provider.generate_with_tools(history, "sys", [tool])

    assert second.text == "Done."
    replayed = sent[1]["messages"]
    assert replayed[1] == {"role": "assistant", "content": tool_turn.content}
    assert replayed[2]["content"][0]["tool_use_id"] == "toolu_1"


async def test_anthropic_refusal_raises_response_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Protects refusal handling; rejects returning an empty answer as if it were valid."""
    client = anthropic.AsyncAnthropic(api_key="fake", max_retries=0)
    refusal = AnthropicMessage.model_validate(
        {
            "id": "msg_2",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": [],
            "stop_reason": "refusal",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 0},
        }
    )

    async def create(**_: Any) -> AnthropicMessage:
        return refusal

    monkeypatch.setattr(client.messages, "create", create)

    with pytest.raises(LLMResponseError):
        await _anthropic_provider(client).generate([UserMessage("hi")])


async def test_gemini_rebuilds_foreign_assistant_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Protects provider switching; rejects dropping tool calls from a non-Gemini turn."""
    client = genai.Client(api_key="fake")
    sent: list[Any] = []

    async def generate_content(**kwargs: Any) -> types.GenerateContentResponse:
        sent.append(kwargs["contents"])
        return types.GenerateContentResponse(
            candidates=[
                types.Candidate(content=types.Content(role="model", parts=[types.Part(text="ok")]))
            ]
        )

    monkeypatch.setattr(client.aio.models, "generate_content", generate_content)
    history = [
        UserMessage("q"),
        AssistantMessage(tool_calls=(ToolCall("gemini-local-abc", "lookup", {"r": "Job 1:1"}),)),
        ToolResultMessage((ToolResult("gemini-local-abc", "lookup", "There was a man..."),)),
    ]
    await _gemini_provider(client).generate(history)

    model_turn, tool_turn = sent[0][1], sent[0][2]
    fc = model_turn.parts[0].function_call
    assert (fc.name, fc.args, fc.id) == ("lookup", {"r": "Job 1:1"}, None)
    fr = tool_turn.parts[0].function_response
    assert (fr.name, fr.response, fr.id) == ("lookup", {"output": "There was a man..."}, None)
