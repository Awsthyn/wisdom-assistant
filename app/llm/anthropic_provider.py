"""Anthropic provider (anthropic SDK): chat, streaming and tool calling.

Anthropic has no embeddings API, so `embed` is delegated to an injected
`Embedder` (Gemini in practice). Embeddings therefore always come from Gemini,
even when chat runs on Anthropic, which also keeps stored vectors compatible.
"""

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

import anthropic
from anthropic.types import ContentBlock, MessageParam, ToolParam
from anthropic.types import Message as AnthropicMessage

from app.llm.base import (
    AssistantMessage,
    Embedder,
    EmbedTask,
    LLMResponse,
    Message,
    ToolCall,
    ToolResultMessage,
    ToolSpec,
    UserMessage,
)
from app.llm.errors import LLMError, LLMResponseError, QuotaExceededError, TransientLLMError
from app.llm.retry import stream_with_retries, with_retries


@dataclass(frozen=True, slots=True)
class _AnthropicTurn:
    """Native content of an assistant turn, replayed verbatim (keeps thinking blocks)."""

    blocks: list[ContentBlock]


def map_anthropic_error(exc: BaseException) -> LLMError | None:
    """Translate an anthropic SDK exception into an LLM-layer error.

    Returns None for exceptions this layer does not own (they propagate unchanged).
    Order matters: the most specific classes come first.
    """
    if isinstance(exc, anthropic.RateLimitError):
        return QuotaExceededError(f"Anthropic rate limit or quota exceeded: {exc.message}")
    if isinstance(exc, anthropic.APIConnectionError):  # includes APITimeoutError
        return TransientLLMError(f"Anthropic connection error: {type(exc).__name__}")
    if isinstance(exc, anthropic.APIStatusError):
        if exc.status_code == 408 or exc.status_code >= 500:
            return TransientLLMError(f"Anthropic server error {exc.status_code}: {exc.message}")
        return LLMError(f"Anthropic API error {exc.status_code}: {exc.message}")
    if isinstance(exc, anthropic.AnthropicError):
        return LLMError(f"Anthropic error: {exc}")
    return None


async def _translate_errors[T](call: Callable[[], Awaitable[T]]) -> T:
    try:
        return await call()
    except Exception as exc:
        mapped = map_anthropic_error(exc)
        if mapped is None:
            raise
        raise mapped from exc


def _to_params(messages: Sequence[Message]) -> list[MessageParam]:
    params: list[MessageParam] = []
    for message in messages:
        match message:
            case UserMessage(content=text):
                params.append({"role": "user", "content": text})
            case AssistantMessage(provider_state=_AnthropicTurn(blocks=blocks)):
                params.append({"role": "assistant", "content": blocks})
            case AssistantMessage(content=text, tool_calls=calls):
                content: list[Any] = []
                if text:
                    content.append({"type": "text", "text": text})
                content.extend(
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                    for call in calls
                )
                params.append({"role": "assistant", "content": content})
            case ToolResultMessage(results=results):
                params.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": result.call_id,
                                "content": result.content,
                                "is_error": result.is_error,
                            }
                            for result in results
                        ],
                    }
                )
    return params


def _to_tools(tools: Sequence[ToolSpec]) -> list[ToolParam]:
    return [
        {"name": tool.name, "description": tool.description, "input_schema": tool.parameters}
        for tool in tools
    ]


def _check_stop_reason(message: AnthropicMessage) -> None:
    if message.stop_reason == "refusal":
        category = message.stop_details.category if message.stop_details else None
        raise LLMResponseError(f"Anthropic declined the request (category={category})")


def _text(message: AnthropicMessage) -> str:
    return "".join(block.text for block in message.content if block.type == "text")


class AnthropicProvider:
    """`LLMProvider` backed by the Anthropic Messages API, with delegated embeddings."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        embedder: Embedder,
        max_tokens: int = 16000,
        timeout_seconds: float = 60.0,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        # SDK retries are disabled so `app.llm.retry` is the single retry policy.
        self._client = client or anthropic.AsyncAnthropic(
            api_key=api_key, max_retries=0, timeout=timeout_seconds
        )
        self._model = model
        self._embedder = embedder
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def chat_model(self) -> str:
        return self._model

    @property
    def embed_model(self) -> str:
        return self._embedder.embed_model

    async def _create(
        self,
        messages: Sequence[Message],
        system: str | None,
        tools: Sequence[ToolSpec] = (),
    ) -> AnthropicMessage:
        params = _to_params(messages)

        async def call() -> AnthropicMessage:
            return await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=params,
                system=system or anthropic.omit,
                tools=_to_tools(tools) if tools else anthropic.omit,
            )

        message = await with_retries(lambda: _translate_errors(call))
        _check_stop_reason(message)
        return message

    async def generate(self, messages: Sequence[Message], system: str | None = None) -> str:
        return _text(await self._create(messages, system))

    async def stream(
        self, messages: Sequence[Message], system: str | None = None
    ) -> AsyncIterator[str]:
        params = _to_params(messages)

        async def open_stream() -> AsyncIterator[str]:
            try:
                async with self._client.messages.stream(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    messages=params,
                    system=system or anthropic.omit,
                ) as stream:
                    async for text in stream.text_stream:
                        yield text
                    _check_stop_reason(await stream.get_final_message())
            except Exception as exc:
                mapped = map_anthropic_error(exc)
                if mapped is None:
                    raise
                raise mapped from exc

        async for fragment in stream_with_retries(open_stream):
            yield fragment

    async def generate_with_tools(
        self,
        messages: Sequence[Message],
        system: str | None,
        tools: Sequence[ToolSpec],
    ) -> LLMResponse:
        message = await self._create(messages, system, tools)
        calls = tuple(
            ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
            for block in message.content
            if block.type == "tool_use"
        )
        return LLMResponse(
            text=_text(message),
            tool_calls=calls,
            model=message.model,
            provider_state=_AnthropicTurn(blocks=list(message.content)),
        )

    async def embed(self, texts: Sequence[str], task: EmbedTask) -> list[list[float]]:
        return await self._embedder.embed(texts, task)
