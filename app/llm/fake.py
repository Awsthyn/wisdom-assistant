"""Deterministic in-memory provider for tests. Never touches the network."""

import hashlib
import math
import struct
from collections import deque
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from app.llm.base import EmbedTask, LLMResponse, Message, ToolSpec, UserMessage


@dataclass(frozen=True, slots=True)
class RecordedCall:
    """One call made to the fake, for assertions in tests."""

    method: str
    messages: tuple[Message, ...]
    system: str | None = None
    tools: tuple[ToolSpec, ...] = ()


def hash_embedding(text: str, task: EmbedTask, dimensions: int) -> list[float]:
    """Unit-length vector derived from SHA-256 of `task:text`; stable across runs and machines."""
    values: list[float] = []
    counter = 0
    seed = f"{task.value}:{text}".encode()
    while len(values) < dimensions:
        digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        # 8 unsigned 32-bit ints per digest, mapped to [-1, 1].
        values.extend(x / 0xFFFFFFFF * 2 - 1 for x in struct.unpack(">8I", digest))
        counter += 1
    vector = values[:dimensions]
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector]


class FakeProvider:
    """`LLMProvider` with scripted answers and hash-based embeddings.

    - `responses`: texts returned in order by `generate` and `stream`; once used up,
      a default answer that echoes the last user message is returned.
    - `tool_responses`: `LLMResponse`s returned in order by `generate_with_tools`
      (use them to script tool calls); once used up, a plain text response.
    """

    def __init__(
        self,
        *,
        responses: Sequence[str] = (),
        tool_responses: Sequence[LLMResponse] = (),
        dimensions: int = 768,
    ) -> None:
        self._responses = deque(responses)
        self._tool_responses = deque(tool_responses)
        self._dimensions = dimensions
        self.calls: list[RecordedCall] = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def chat_model(self) -> str:
        return "fake-chat"

    @property
    def embed_model(self) -> str:
        return "fake-embed"

    def _next_text(self, messages: Sequence[Message]) -> str:
        if self._responses:
            return self._responses.popleft()
        last_user = next((m.content for m in reversed(messages) if isinstance(m, UserMessage)), "")
        return f"Fake answer to: {last_user}"

    async def generate(self, messages: Sequence[Message], system: str | None = None) -> str:
        self.calls.append(RecordedCall("generate", tuple(messages), system))
        return self._next_text(messages)

    async def stream(
        self, messages: Sequence[Message], system: str | None = None
    ) -> AsyncIterator[str]:
        self.calls.append(RecordedCall("stream", tuple(messages), system))
        text = self._next_text(messages)
        words = text.split(" ")
        for index, word in enumerate(words):
            yield word if index == len(words) - 1 else f"{word} "

    async def generate_with_tools(
        self,
        messages: Sequence[Message],
        system: str | None,
        tools: Sequence[ToolSpec],
    ) -> LLMResponse:
        self.calls.append(
            RecordedCall("generate_with_tools", tuple(messages), system, tuple(tools))
        )
        if self._tool_responses:
            return self._tool_responses.popleft()
        return LLMResponse(text=self._next_text(messages), model=self.chat_model)

    async def embed(self, texts: Sequence[str], task: EmbedTask) -> list[list[float]]:
        self.calls.append(RecordedCall("embed", ()))
        return [hash_embedding(text, task, self._dimensions) for text in texts]
