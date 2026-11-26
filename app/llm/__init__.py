"""LLM provider layer.

The only package allowed to import the Gemini (`google.genai`) and Anthropic
(`anthropic`) SDKs. Everything else depends on `LLMProvider` and the neutral
types re-exported here.
"""

from app.llm.base import (
    AssistantMessage,
    Embedder,
    EmbedTask,
    LLMProvider,
    LLMResponse,
    Message,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    ToolSpec,
    UserMessage,
)
from app.llm.errors import (
    LLMConfigError,
    LLMError,
    LLMResponseError,
    QuotaExceededError,
    TransientLLMError,
)
from app.llm.factory import build_llm_provider

__all__ = [
    "AssistantMessage",
    "EmbedTask",
    "Embedder",
    "LLMConfigError",
    "LLMError",
    "LLMProvider",
    "LLMResponse",
    "LLMResponseError",
    "Message",
    "QuotaExceededError",
    "ToolCall",
    "ToolResult",
    "ToolResultMessage",
    "ToolSpec",
    "TransientLLMError",
    "UserMessage",
    "build_llm_provider",
]
