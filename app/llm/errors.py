"""Provider-neutral LLM exceptions. SDK exceptions never escape `app/llm/`."""


class LLMError(Exception):
    """Base class for every error raised by the LLM layer."""


class LLMConfigError(LLMError):
    """The provider cannot be built from the current settings (e.g. a missing API key)."""


class QuotaExceededError(LLMError):
    """The provider rejected the call for quota or rate limits (HTTP 429). Not retried."""


class TransientLLMError(LLMError):
    """A timeout, connection failure or 5xx. Retried with exponential backoff."""


class LLMResponseError(LLMError):
    """The provider answered, but not with something usable (refusal, empty or malformed)."""
