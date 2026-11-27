"""Tool definitions shared by the agent (B5) and the MCP server (B6).

A tool is defined once: a Pydantic model for its arguments (used both to
validate calls and to generate the JSON Schema the LLM sees), a description,
and an async handler that calls the service layer.
"""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm import Embedder, ToolSpec
from app.services.search import SearchHit


@dataclass(frozen=True, slots=True)
class ToolContext:
    """What a tool may use. The caller owns the session's lifetime."""

    session: AsyncSession
    embedder: Embedder


class ToolError(Exception):
    """A problem the model can fix (bad arguments, unknown reference...).

    The message is returned to the model as the tool result; it never aborts the agent.
    """


@dataclass(frozen=True, slots=True)
class ToolOutput:
    data: dict[str, Any]  # sent to the model as JSON
    summary: str  # one short line for UIs (the agent's `tool_result` event)
    passages: tuple[SearchHit, ...] = field(default=())  # texts returned, for citation checks

    def to_json(self) -> str:
        return json.dumps(self.data, ensure_ascii=False)


def json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """A provider-friendly JSON Schema for `model`.

    Drops Pydantic's `title` keys and `additionalProperties`, and turns
    `X | None` (`anyOf: [X, null]`) into plain optional `X`, which both Gemini
    function declarations and Anthropic tool schemas accept.
    """

    def clean(node: Any) -> Any:
        if isinstance(node, dict):
            source = cast("dict[str, Any]", node)
            items: dict[str, Any] = {
                k: clean(v) for k, v in source.items() if k not in ("title", "additionalProperties")
            }
            any_of = items.get("anyOf")
            if isinstance(any_of, list):
                options = [o for o in cast("list[Any]", any_of) if o != {"type": "null"}]
                if len(options) == 1:
                    del items["anyOf"]
                    items = {**options[0], **items}
            if items.get("default", ...) is None:
                del items["default"]
            return items
        if isinstance(node, list):
            return [clean(v) for v in cast("list[Any]", node)]
        return node

    schema: dict[str, Any] = clean(model.model_json_schema())
    return schema


@dataclass(frozen=True, slots=True)
class Tool[A: BaseModel]:
    name: str
    description: str
    args_model: type[A]
    handler: Callable[[ToolContext, A], Awaitable[ToolOutput]]

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, json_schema(self.args_model))

    async def run(self, context: ToolContext, arguments: dict[str, Any]) -> ToolOutput:
        """Validate `arguments` and run the handler. Raises `ToolError` on bad input."""
        try:
            args = self.args_model.model_validate(arguments)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}"
                for e in exc.errors()
            )
            raise ToolError(f"Invalid arguments for {self.name}: {problems}") from exc
        return await self.handler(context, args)
