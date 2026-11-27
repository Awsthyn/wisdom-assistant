"""Application settings, loaded only from environment variables (and a local .env)."""

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

LLMProviderName = Literal["gemini", "anthropic"]


class Settings(BaseSettings):
    """Typed view of the process environment.

    Secrets are `SecretStr` so they never show up in reprs or logs.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM provider selection
    llm_provider: LLMProviderName = "gemini"
    llm_timeout_seconds: float = Field(default=60.0, gt=0)

    # Gemini (also used for embeddings when the chat provider is Anthropic)
    gemini_api_key: SecretStr | None = None
    gemini_chat_model: str = "gemini-3.5-flash-lite"
    gemini_embed_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768

    # Anthropic (optional)
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-haiku-5-5"
    anthropic_max_tokens: int = Field(default=16000, gt=0)

    # Embedding ingestion pacing. Defaults sit ~20% under Gemini's free-tier limits
    # for gemini-embedding-001 (100 RPM, 30,000 TPM) to leave headroom.
    embed_rpm: int = Field(default=80, gt=0)
    embed_tpm: int = Field(default=24_000, gt=0)

    # RAG chat: passages retrieved per question, and the minimum cosine similarity of
    # the best passage below which the question is answered `no_results` without
    # calling the LLM (saves quota on off-topic questions). 0.57 was calibrated on
    # gemini-embedding-001 at 768 dims: on-topic questions scored 0.596-0.729 and
    # off-topic ones 0.490-0.590, so it errs on the side of answering.
    chat_top_k: int = Field(default=8, ge=1, le=20)
    chat_min_score: float = Field(default=0.57, ge=0, le=1)

    # Database. DATABASE_URL is the pooled URL used by the API; DATABASE_URL_DIRECT
    # (non-pooled) is used by migrations and ingestion. Both accept Neon's
    # `postgresql://...?sslmode=require` form.
    database_url: SecretStr | None = None
    database_url_direct: SecretStr | None = None

    # HTTP
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list[str])

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, value: object) -> object:
        """Accept a comma-separated string such as `http://a.com,http://b.com`."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
