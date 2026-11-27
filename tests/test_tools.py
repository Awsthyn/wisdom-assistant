"""Tool definitions: schemas the providers accept, and argument validation."""

from typing import Any, cast

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.fake import FakeProvider
from app.tools import (
    BIBLE_TOOLS,
    CROSS_REFERENCES,
    LOOKUP_REFERENCE,
    SEARCH_PASSAGES,
    ToolContext,
    ToolError,
)

# Validation fails before any handler runs, so no database is needed here.
CONTEXT = ToolContext(session=cast(AsyncSession, None), embedder=FakeProvider())


def _walk(node: Any) -> list[dict[str, Any]]:
    if isinstance(node, dict):
        mapping = cast(dict[str, Any], node)
        return [mapping, *(d for v in mapping.values() for d in _walk(v))]
    if isinstance(node, list):
        return [d for v in cast(list[Any], node) for d in _walk(v)]
    return []


def test_the_three_tools_are_defined_once_with_unique_names() -> None:
    """Protects the shared tool registry (agent and MCP); rejects missing or duplicate tools."""
    assert [t.name for t in BIBLE_TOOLS] == [
        "lookup_reference",
        "search_passages",
        "cross_references",
    ]


def test_schemas_are_plain_json_schema_objects() -> None:
    """Protects provider compatibility; rejects Pydantic titles, anyOf-null or extra keys."""
    for tool in BIBLE_TOOLS:
        schema = tool.spec.parameters
        assert schema["type"] == "object"
        for node in _walk(schema):
            assert "title" not in node
            assert "anyOf" not in node
            assert "additionalProperties" not in node


def test_required_and_optional_parameters() -> None:
    """Protects the documented signatures; rejects `k`, `book` or `limit` becoming required."""
    search = SEARCH_PASSAGES.spec.parameters
    assert search["required"] == ["query"]
    assert search["properties"]["k"]["default"] == 5
    assert search["properties"]["book"]["type"] == "string"
    assert CROSS_REFERENCES.spec.parameters["required"] == ["reference"]
    assert CROSS_REFERENCES.spec.parameters["properties"]["limit"]["default"] == 10
    assert LOOKUP_REFERENCE.spec.parameters["required"] == ["reference"]


@pytest.mark.parametrize(
    ("tool_name", "arguments", "message"),
    [
        ("lookup_reference", {}, "reference: Field required"),
        ("search_passages", {"query": "wisdom", "k": 50}, "k: Input should be less than"),
        ("search_passages", {"query": "hi"}, "query: String should have at least 3"),
        ("cross_references", {"reference": "Ps 23:1", "votes": 3}, "votes: Extra inputs"),
    ],
)
async def test_invalid_arguments_raise_tool_error(
    tool_name: str, arguments: dict[str, Any], message: str
) -> None:
    """Protects the agent from bad model output; rejects running a tool with invalid arguments."""
    tool = next(t for t in BIBLE_TOOLS if t.name == tool_name)

    with pytest.raises(ToolError, match=message):
        await tool.run(CONTEXT, arguments)
