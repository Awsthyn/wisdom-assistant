# wisdom-assistant

A study assistant for the five wisdom books of the World English Bible — Job, Psalms,
Proverbs, Ecclesiastes and Song of Songs — with semantic search, RAG chat that cites
verses, a tool-calling agent and an MCP server.

> Status: early setup. No application code yet.

## Stack

FastAPI (Python 3.12+) · Postgres + pgvector (Neon) · Gemini or Anthropic · Next.js (Vercel)

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```sh
uv sync
cp .env.example .env              # then fill in GEMINI_API_KEY
uv run uvicorn app.main:app --reload   # http://localhost:8000/docs

uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest
```

Unit tests never call a real LLM; they use a deterministic fake provider.
To check your real keys and models (spends a tiny bit of quota):

```sh
uv run python -m app.scripts.llm_smoke
```

### LLM providers

`LLM_PROVIDER=gemini` (default) or `anthropic`. Embeddings always use Gemini
(`gemini-embedding-001`, 768 dimensions), because Anthropic has no embeddings API.

## Attributions

- **World English Bible** — [ebible.org](https://ebible.org/), public domain.
- **Cross references** — [OpenBible.info](https://www.openbible.info/labs/cross-references/),
  licensed under [CC-BY](https://creativecommons.org/licenses/by/4.0/).
