"""Gemini provider (google-genai SDK): chat, streaming, tool calling and embeddings."""

import math
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

import httpx2
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.llm.base import (
    AssistantMessage,
    EmbedTask,
    LLMResponse,
    Message,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    ToolSpec,
    UserMessage,
)
from app.llm.errors import LLMError, LLMResponseError, QuotaExceededError, TransientLLMError
from app.llm.retry import stream_with_retries, with_retries

# Gemini's batch embedding endpoint accepts at most 100 texts per request.
EMBED_BATCH_SIZE = 100

# Gemini may omit function call ids; we invent local ones so callers always have
# an id, and strip them again when sending the function response back.
_LOCAL_CALL_ID_PREFIX = "gemini-local-"

_TASK_TYPES: dict[EmbedTask, str] = {
    EmbedTask.DOCUMENT: "RETRIEVAL_DOCUMENT",
    EmbedTask.QUERY: "RETRIEVAL_QUERY",
}


def map_gemini_error(exc: BaseException) -> LLMError | None:
    """Translate a google-genai / transport exception into an LLM-layer error.

    Returns None for exceptions this layer does not own (they propagate unchanged).
    """
    if isinstance(exc, genai_errors.APIError):
        if exc.code == 429 or exc.status == "RESOURCE_EXHAUSTED":
            return QuotaExceededError(f"Gemini quota exhausted: {exc.message}")
        if exc.code == 408 or exc.code >= 500:
            return TransientLLMError(f"Gemini server error {exc.code}: {exc.message}")
        return LLMError(f"Gemini API error {exc.code}: {exc.message}")
    if isinstance(exc, httpx2.TransportError):  # timeouts and connection failures
        return TransientLLMError(f"Gemini transport error: {type(exc).__name__}")
    return None


async def _translate_errors[T](call: Callable[[], Awaitable[T]]) -> T:
    try:
        return await call()
    except Exception as exc:
        mapped = map_gemini_error(exc)
        if mapped is None:
            raise
        raise mapped from exc


def _l2_normalize(vector: list[float]) -> list[float]:
    # gemini-embedding-001 only normalizes its full 3072-dim output; truncated
    # outputs (e.g. 768) must be normalized so cosine and dot product agree.
    norm = math.sqrt(sum(x * x for x in vector))
    return vector if norm == 0 else [x / norm for x in vector]


def _to_contents(messages: Sequence[Message]) -> list[types.Content]:
    contents: list[types.Content] = []
    for message in messages:
        match message:
            case UserMessage(content=text):
                contents.append(types.Content(role="user", parts=[types.Part(text=text)]))
            case AssistantMessage(provider_state=types.Content() as native):
                # Replay verbatim: keeps thought signatures required for tool calling.
                contents.append(native)
            case AssistantMessage(content=text, tool_calls=calls):
                parts: list[types.Part] = []
                if text:
                    parts.append(types.Part(text=text))
                parts.extend(
                    types.Part(
                        function_call=types.FunctionCall(
                            id=_remote_call_id(call.id), name=call.name, args=call.arguments
                        )
                    )
                    for call in calls
                )
                contents.append(types.Content(role="model", parts=parts))
            case ToolResultMessage(results=results):
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                function_response=types.FunctionResponse(
                                    id=_remote_call_id(result.call_id),
                                    name=result.name,
                                    response=_function_response(result),
                                )
                            )
                            for result in results
                        ],
                    )
                )
    return contents


def _function_response(result: ToolResult) -> dict[str, Any]:
    return {"error" if result.is_error else "output": result.content}


def _remote_call_id(call_id: str) -> str | None:
    return None if call_id.startswith(_LOCAL_CALL_ID_PREFIX) else call_id


def _first_candidate_content(response: types.GenerateContentResponse) -> types.Content:
    candidates = response.candidates or []
    if not candidates or candidates[0].content is None:
        feedback = response.prompt_feedback
        reason = feedback.block_reason if feedback else None
        raise LLMResponseError(f"Gemini returned no content (block_reason={reason})")
    return candidates[0].content


def _visible_text(content: types.Content) -> str:
    return "".join(part.text for part in content.parts or [] if part.text and not part.thought)


def _tool_calls(content: types.Content) -> tuple[ToolCall, ...]:
    calls: list[ToolCall] = []
    for part in content.parts or []:
        fc = part.function_call
        if fc is None or not fc.name:
            continue
        call_id = fc.id or f"{_LOCAL_CALL_ID_PREFIX}{uuid.uuid4().hex[:12]}"
        args: dict[str, Any] = dict(fc.args or {})
        calls.append(ToolCall(id=call_id, name=fc.name, arguments=args))
    return tuple(calls)


class GeminiProvider:
    """`LLMProvider` backed by the Gemini API."""

    def __init__(
        self,
        *,
        api_key: str,
        chat_model: str,
        embed_model: str,
        embedding_dimensions: int = 768,
        timeout_seconds: float = 60.0,
        client: genai.Client | None = None,
    ) -> None:
        self._client = client or genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
        )
        self._chat_model = chat_model
        self._embed_model = embed_model
        self._dimensions = embedding_dimensions

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def chat_model(self) -> str:
        return self._chat_model

    @property
    def embed_model(self) -> str:
        return self._embed_model

    def _config(
        self, system: str | None, tools: Sequence[ToolSpec] = ()
    ) -> types.GenerateContentConfig:
        # Typed as the SDK's own union: with the `mcp` package installed it also admits
        # MCP tools, and `list` is invariant, so `list[types.Tool]` would not match.
        gemini_tools: list[types.ToolUnion] | None = None
        if tools:
            gemini_tools = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool.name,
                            description=tool.description,
                            parameters_json_schema=tool.parameters,
                        )
                        for tool in tools
                    ]
                )
            ]
        return types.GenerateContentConfig(
            system_instruction=system,
            tools=gemini_tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    async def _generate_content(
        self, messages: Sequence[Message], config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse:
        contents = _to_contents(messages)

        async def call() -> types.GenerateContentResponse:
            return await self._client.aio.models.generate_content(  # pyright: ignore[reportUnknownMemberType]  # SDK union includes PIL.Image (not installed)
                model=self._chat_model, contents=contents, config=config
            )

        return await with_retries(lambda: _translate_errors(call))

    async def generate(self, messages: Sequence[Message], system: str | None = None) -> str:
        response = await self._generate_content(messages, self._config(system))
        return _visible_text(_first_candidate_content(response))

    async def stream(
        self, messages: Sequence[Message], system: str | None = None
    ) -> AsyncIterator[str]:
        contents = _to_contents(messages)
        config = self._config(system)

        async def open_stream() -> AsyncIterator[str]:
            try:
                chunks = await self._client.aio.models.generate_content_stream(  # pyright: ignore[reportUnknownMemberType]  # SDK union includes PIL.Image (not installed)
                    model=self._chat_model, contents=contents, config=config
                )
                async for chunk in chunks:
                    for candidate in chunk.candidates or []:
                        if candidate.content is not None:
                            text = _visible_text(candidate.content)
                            if text:
                                yield text
            except Exception as exc:
                mapped = map_gemini_error(exc)
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
        response = await self._generate_content(messages, self._config(system, tools))
        content = _first_candidate_content(response)
        return LLMResponse(
            text=_visible_text(content),
            tool_calls=_tool_calls(content),
            model=response.model_version or self._chat_model,
            provider_state=content,
        )

    async def embed(self, texts: Sequence[str], task: EmbedTask) -> list[list[float]]:
        vectors: list[list[float]] = []
        config = types.EmbedContentConfig(
            task_type=_TASK_TYPES[task], output_dimensionality=self._dimensions
        )
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = list(texts[start : start + EMBED_BATCH_SIZE])

            async def call(batch: list[str] = batch) -> types.EmbedContentResponse:
                return await self._client.aio.models.embed_content(  # pyright: ignore[reportUnknownMemberType]  # SDK union includes PIL.Image (not installed)
                    model=self._embed_model, contents=batch, config=config
                )

            response = await with_retries(lambda call=call: _translate_errors(call))
            embeddings = response.embeddings or []
            if len(embeddings) != len(batch):
                raise LLMResponseError(
                    f"Gemini returned {len(embeddings)} embeddings for {len(batch)} texts"
                )
            for embedding in embeddings:
                values = list(embedding.values or [])
                if len(values) != self._dimensions:
                    raise LLMResponseError(
                        f"Expected {self._dimensions}-dim embedding, got {len(values)}"
                    )
                vectors.append(_l2_normalize(values))
        return vectors
