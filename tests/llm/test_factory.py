import pytest

from app.llm.anthropic_provider import AnthropicProvider
from app.llm.errors import LLMConfigError
from app.llm.factory import build_llm_provider
from app.llm.gemini import GeminiProvider
from tests.conftest import make_settings


def test_gemini_is_selected_when_configured() -> None:
    """Protects LLM_PROVIDER=gemini; rejects building any other provider or ignoring the models."""
    provider = build_llm_provider(
        make_settings(llm_provider="gemini", gemini_api_key="k", gemini_chat_model="chat-x")
    )

    assert isinstance(provider, GeminiProvider)
    assert provider.chat_model == "chat-x"
    assert provider.embed_model == "gemini-embedding-001"


def test_gemini_without_key_fails_with_clear_message() -> None:
    """Protects the missing-key error; rejects a silent default or a message without the var."""
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        build_llm_provider(make_settings(llm_provider="gemini"))


def test_empty_gemini_key_counts_as_missing() -> None:
    """Protects against `GEMINI_API_KEY=` in .env; rejects building a provider with an empty key."""
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        build_llm_provider(make_settings(llm_provider="gemini", gemini_api_key=""))


def test_anthropic_is_selected_and_embeds_with_gemini() -> None:
    """Protects LLM_PROVIDER=anthropic; rejects embeddings that don't come from Gemini."""
    provider = build_llm_provider(
        make_settings(
            llm_provider="anthropic",
            anthropic_api_key="a",
            gemini_api_key="g",
            anthropic_model="claude-haiku-5-5",
        )
    )

    assert isinstance(provider, AnthropicProvider)
    assert provider.chat_model == "claude-haiku-5-5"
    assert provider.embed_model == "gemini-embedding-001"


def test_anthropic_without_its_key_fails_with_clear_message() -> None:
    """Protects the Anthropic missing-key error; rejects falling back to Gemini silently."""
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
        build_llm_provider(make_settings(llm_provider="anthropic", gemini_api_key="g"))


def test_anthropic_without_gemini_key_fails_because_of_embeddings() -> None:
    """Protects the embeddings dependency; rejects an Anthropic provider that cannot embed."""
    with pytest.raises(LLMConfigError, match=r"GEMINI_API_KEY.*embeddings"):
        build_llm_provider(make_settings(llm_provider="anthropic", anthropic_api_key="a"))
