"""Provider-neutral types and the `LLMProvider` protocol.

Everything outside `app/llm/` talks to language models only through these types.
"""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class EmbedTask(StrEnum):
    """What an embedding is for. Gemini embeds documents and queries differently."""

    DOCUMENT = "document"
    QUERY = "query"


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool the model may call. `parameters` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The output of running a tool, sent back to the model."""

    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class UserMessage:
    content: str


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    """A previous model turn.

    `provider_state` is an opaque copy of the provider's native response content
    (e.g. Anthropic thinking blocks, Gemini thought signatures). A provider replays
    it verbatim when it recognizes it and otherwise rebuilds the turn from
    `content` and `tool_calls`. Callers must never inspect it.
    """

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    provider_state: object | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    """Results for every tool call of the preceding assistant turn."""

    results: tuple[ToolResult, ...]


Message = UserMessage | AssistantMessage | ToolResultMessage


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Result of `generate_with_tools`: text, tool calls, or both."""

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    model: str = ""
    provider_state: object | None = field(default=None, compare=False, repr=False)

    def to_message(self) -> AssistantMessage:
        """Turn this response into a history entry for the next request."""
        return AssistantMessage(
            content=self.text,
            tool_calls=self.tool_calls,
            provider_state=self.provider_state,
        )


class Embedder(Protocol):
    """The embedding half of a provider. Lets Anthropic chat reuse Gemini embeddings."""

    @property
    def embed_model(self) -> str: ...

    async def embed(self, texts: Sequence[str], task: EmbedTask) -> list[list[float]]: ...


@runtime_checkable
class LLMProvider(Protocol):
    """Common interface implemented by Gemini, Anthropic and the fake provider."""

    @property
    def name(self) -> str: ...

    @property
    def chat_model(self) -> str: ...

    @property
    def embed_model(self) -> str: ...

    async def generate(self, messages: Sequence[Message], system: str | None = None) -> str:
        """Return the full text answer."""
        ...

    def stream(self, messages: Sequence[Message], system: str | None = None) -> AsyncIterator[str]:
        """Yield the answer as text fragments, in order."""
        ...

    async def generate_with_tools(
        self,
        messages: Sequence[Message],
        system: str | None,
        tools: Sequence[ToolSpec],
    ) -> LLMResponse:
        """Return a response that may contain tool calls instead of (or besides) text."""
        ...

    async def embed(self, texts: Sequence[str], task: EmbedTask) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...
