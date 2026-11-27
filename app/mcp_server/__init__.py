"""MCP server exposing the Bible tools of `app/tools/` (stdio transport).

The tools are not redefined here: each SDK tool is a thin adapter whose input
schema is exactly `app.tools`' JSON Schema and whose body calls `Tool.run`, so
validation, descriptions and behaviour are shared with the agent.
"""

import inspect
from collections.abc import Callable
from typing import Any, Final

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError as McpToolError
from mcp.server.mcpserver.tools import Tool as McpTool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.llm import Embedder, EmbedTask, LLMConfigError, QuotaExceededError, TransientLLMError
from app.tools import BIBLE_TOOLS, Tool, ToolContext, ToolError

SERVER_NAME: Final = "wisdom-assistant"
ATTRIBUTION_URI: Final = "wisdom://attribution"

INSTRUCTIONS: Final = """\
Study tools for the five wisdom books of the World English Bible (Job, Psalms, \
Proverbs, Ecclesiastes, Song of Songs).

- search_passages: find passages in the wisdom books by meaning ("trusting God in \
suffering"). Use it first for thematic questions.
- lookup_reference: exact text of any reference in the 66 books ("Proverbs 3:5-6").
- cross_references: where a verse's words or themes reappear elsewhere in the Bible.

Results outside the wisdom books carry in_corpus: false. Cite verses as \
[Book Chapter:Verse]. Sources and licenses: read the resource wisdom://attribution."""

ATTRIBUTION: Final = """\
# Sources and licenses

- **World English Bible (WEB)**: scripture text, from https://ebible.org/. Public domain.
- **OpenBible.info cross references**: https://www.openbible.info/labs/cross-references/.
  Licensed under Creative Commons Attribution (CC-BY 4.0,
  https://creativecommons.org/licenses/by/4.0/). Only cross references with at least
  5 votes are included.

wisdom-assistant: https://github.com/Awsthyn/wisdom-assistant
"""


class LazyEmbedder:
    """Builds the real embedder on first use, so lookups work without an API key."""

    def __init__(self, factory: Callable[[], Embedder]) -> None:
        self._factory = factory
        self._embedder: Embedder | None = None

    def _get(self) -> Embedder:
        if self._embedder is None:
            self._embedder = self._factory()
        return self._embedder

    @property
    def embed_model(self) -> str:
        return self._get().embed_model

    async def embed(self, texts: Any, task: EmbedTask) -> list[list[float]]:
        return await self._get().embed(texts, task)


def _signature(tool: Tool[Any]) -> inspect.Signature:
    """Keyword-only parameters named like the tool's fields.

    Types are `Any` on purpose: the SDK only checks presence; `Tool.run`
    validates against the tool's own Pydantic model.
    """
    parameters = [
        inspect.Parameter(
            name,
            inspect.Parameter.KEYWORD_ONLY,
            annotation=Any,
            default=inspect.Parameter.empty if field.is_required() else field.default,
        )
        for name, field in tool.args_model.model_fields.items()
    ]
    return inspect.Signature(parameters, return_annotation=str)


def _adapter(
    tool: Tool[Any], sessions: async_sessionmaker[AsyncSession], embedder: Embedder
) -> McpTool:
    async def call(**arguments: Any) -> str:
        async with sessions() as session:
            try:
                output = await tool.run(ToolContext(session, embedder), arguments)
            except ToolError as exc:
                raise McpToolError(str(exc)) from exc
            except QuotaExceededError as exc:
                raise McpToolError("The embedding quota is exhausted; try again later.") from exc
            except TransientLLMError as exc:
                raise McpToolError("The embedding service did not respond; try again.") from exc
            except LLMConfigError as exc:
                raise McpToolError(f"Semantic search is not configured: {exc}") from exc
        return output.to_json()

    call.__name__ = tool.name
    call.__signature__ = _signature(tool)  # type: ignore[attr-defined]
    mcp_tool = McpTool.from_function(
        call, name=tool.name, description=tool.description, structured_output=False
    )
    mcp_tool.parameters = tool.spec.parameters  # publish exactly app.tools' schema
    return mcp_tool


def create_mcp_server(
    sessions: async_sessionmaker[AsyncSession],
    embedder: Embedder,
    tools: tuple[Tool[Any], ...] = BIBLE_TOOLS,
) -> MCPServer:
    server = MCPServer(
        name=SERVER_NAME,
        title="Wisdom Assistant",
        instructions=INSTRUCTIONS,
        version="0.1.0",
        tools=[_adapter(tool, sessions, embedder) for tool in tools],
    )

    @server.resource(
        ATTRIBUTION_URI,
        name="attribution",
        title="Sources and licenses",
        description="Licenses of the texts this server returns (WEB, OpenBible.info).",
        mime_type="text/markdown",
    )
    def attribution() -> str:  # pyright: ignore[reportUnusedFunction]
        return ATTRIBUTION

    return server
