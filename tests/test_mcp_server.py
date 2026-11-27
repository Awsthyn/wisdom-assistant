"""The MCP server through the SDK's in-process client (no database needed here)."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, cast

from mcp import Client
from mcp_types import TextContent, TextResourceContents
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.llm.fake import FakeProvider
from app.mcp_server import ATTRIBUTION_URI, create_mcp_server
from app.tools import BIBLE_TOOLS


@asynccontextmanager
async def _unused_session() -> AsyncGenerator[object]:
    yield object()  # these calls fail validation before any query runs


SESSIONS = cast(async_sessionmaker[AsyncSession], _unused_session)


async def test_lists_exactly_the_three_tools_with_their_schemas() -> None:
    """Protects the MCP surface; rejects extra, missing or re-derived tool schemas."""
    async with Client(create_mcp_server(SESSIONS, FakeProvider())) as client:
        listed = (await client.list_tools()).tools

    assert [t.name for t in listed] == ["lookup_reference", "search_passages", "cross_references"]
    for tool, mcp_tool in zip(BIBLE_TOOLS, listed, strict=True):
        assert mcp_tool.input_schema == tool.spec.parameters
        assert mcp_tool.description == tool.description


async def test_invalid_arguments_return_a_tool_error_message() -> None:
    """Protects clients from crashes; rejects a generic failure without the validation reason."""
    async with Client(create_mcp_server(SESSIONS, FakeProvider())) as client:
        result = await client.call_tool("search_passages", {"query": "wisdom", "k": 99})

    assert result.is_error
    content = result.content[0]
    assert isinstance(content, TextContent)
    assert "Invalid arguments for search_passages" in content.text
    assert "k: Input should be less than or equal to 10" in content.text


async def test_attribution_resource_lists_both_licenses() -> None:
    """Protects the attribution requirement; rejects a resource missing WEB or OpenBible CC-BY."""
    async with Client(create_mcp_server(SESSIONS, FakeProvider())) as client:
        resources = (await client.list_resources()).resources
        read = await client.read_resource(ATTRIBUTION_URI)

    assert [(str(r.uri), r.mime_type) for r in resources] == [(ATTRIBUTION_URI, "text/markdown")]
    [content] = read.contents
    assert isinstance(content, TextResourceContents)
    text: Any = content.text
    assert "World English Bible" in text
    assert "Public domain" in text
    assert "OpenBible.info" in text
    assert "CC-BY" in text
