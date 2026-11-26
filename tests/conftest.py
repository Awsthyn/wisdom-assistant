import pytest

from app.core.config import Settings

_ENV_VARS = (
    "LLM_PROVIDER",
    "GEMINI_API_KEY",
    "GEMINI_CHAT_MODEL",
    "GEMINI_EMBED_MODEL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "CORS_ORIGINS",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's real keys and settings out of every test."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def make_settings(**overrides: object) -> Settings:
    """Settings built only from explicit values, never from a local .env file."""
    return Settings(_env_file=None, **overrides)  # pyright: ignore[reportCallIssue, reportArgumentType]
