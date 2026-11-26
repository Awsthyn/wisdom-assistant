"""Builds the configured `LLMProvider` from settings."""

from pydantic import SecretStr

from app.core.config import Settings
from app.llm.anthropic_provider import AnthropicProvider
from app.llm.base import LLMProvider
from app.llm.errors import LLMConfigError
from app.llm.gemini import GeminiProvider


def _secret(value: SecretStr | None) -> str | None:
    """Return the secret's value, treating an empty string as unset."""
    if value is None:
        return None
    return value.get_secret_value() or None


def _gemini(settings: Settings, purpose: str) -> GeminiProvider:
    api_key = _secret(settings.gemini_api_key)
    if api_key is None:
        raise LLMConfigError(
            f"GEMINI_API_KEY is not set. It is required {purpose}. "
            "Add it to your environment or .env (see .env.example)."
        )
    return GeminiProvider(
        api_key=api_key,
        chat_model=settings.gemini_chat_model,
        embed_model=settings.gemini_embed_model,
        embedding_dimensions=settings.embedding_dimensions,
        timeout_seconds=settings.llm_timeout_seconds,
    )


def build_llm_provider(settings: Settings) -> LLMProvider:
    """Return the provider selected by `LLM_PROVIDER`.

    Raises `LLMConfigError` with an actionable message when a required key is missing.
    With `LLM_PROVIDER=anthropic`, `GEMINI_API_KEY` is still required because
    embeddings always come from Gemini.
    """
    match settings.llm_provider:
        case "gemini":
            return _gemini(settings, "when LLM_PROVIDER=gemini")
        case "anthropic":
            anthropic_key = _secret(settings.anthropic_api_key)
            if anthropic_key is None:
                raise LLMConfigError(
                    "ANTHROPIC_API_KEY is not set. It is required when LLM_PROVIDER=anthropic. "
                    "Set it, or switch to LLM_PROVIDER=gemini."
                )
            embedder = _gemini(
                settings, "for embeddings even when LLM_PROVIDER=anthropic (Anthropic has none)"
            )
            return AnthropicProvider(
                api_key=anthropic_key,
                model=settings.anthropic_model,
                embedder=embedder,
                max_tokens=settings.anthropic_max_tokens,
                timeout_seconds=settings.llm_timeout_seconds,
            )
