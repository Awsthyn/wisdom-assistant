"""Manual smoke test: one real chat call and one real embedding call.

Usage: `uv run python -m app.scripts.llm_smoke`

Uses the configured provider and real API quota. Never run in CI. Prints only
model names, the vector length and pass/fail; never prompts, answers or keys.
"""

import asyncio
import sys

from app.core.config import get_settings
from app.llm import EmbedTask, LLMError, UserMessage, build_llm_provider


async def main() -> int:
    settings = get_settings()
    try:
        provider = build_llm_provider(settings)
    except LLMError as exc:
        print(f"config: FAILED ({exc})")
        return 1

    ok = True
    print(f"provider: {provider.name}")

    try:
        answer = await provider.generate([UserMessage("Reply with the single word: ok")])
        print(f"chat model: {provider.chat_model} -> {'OK' if answer.strip() else 'EMPTY'}")
        ok = ok and bool(answer.strip())
    except LLMError as exc:
        print(f"chat model: {provider.chat_model} -> FAILED ({type(exc).__name__}: {exc})")
        ok = False

    try:
        [vector] = await provider.embed(["The fear of the LORD"], EmbedTask.QUERY)
        print(f"embed model: {provider.embed_model} -> OK, dimensions={len(vector)}")
    except LLMError as exc:
        print(f"embed model: {provider.embed_model} -> FAILED ({type(exc).__name__}: {exc})")
        ok = False

    print("result:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
