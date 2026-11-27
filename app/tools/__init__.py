"""Tools exposed to the LLM agent and (in B6) to MCP clients. Defined once, here."""

from typing import Any, Final

from app.tools.base import Tool, ToolContext, ToolError, ToolOutput, json_schema
from app.tools.bible import CROSS_REFERENCES, LOOKUP_REFERENCE, SEARCH_PASSAGES

BIBLE_TOOLS: Final[tuple[Tool[Any], ...]] = (LOOKUP_REFERENCE, SEARCH_PASSAGES, CROSS_REFERENCES)

__all__ = [
    "BIBLE_TOOLS",
    "CROSS_REFERENCES",
    "LOOKUP_REFERENCE",
    "SEARCH_PASSAGES",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolOutput",
    "json_schema",
]
