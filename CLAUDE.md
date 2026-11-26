# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## Project

**wisdom-assistant** is an AI Engineering portfolio project: a study assistant for the five
wisdom books of the World English Bible (WEB, public domain) — Job, Psalms, Proverbs,
Ecclesiastes and Song of Songs. Features: semantic search, RAG chat that cites verses, a
tool-calling agent, and an MCP server.

Deployment: API on Render Free, Postgres + pgvector on Neon Free, frontend on Vercel Hobby.
LLM: Gemini (free tier) by default, switchable to Anthropic via an environment variable.

**Language:** all code, comments, commit messages and documentation are written in English.

## Stack

- **Python 3.12+**, managed with **uv**
- **FastAPI**, **Pydantic v2**, **pydantic-settings**
- **SQLAlchemy 2.0 (async)** + **asyncpg**
- **pgvector**: the Postgres extension and the `pgvector` Python library
- **Alembic** for migrations
- **pytest** + **pytest-asyncio** + **httpx** for tests
- **Ruff** (lint + format), **pyright** in strict mode
- LLM SDKs: **google-genai** (Gemini) and **anthropic**
- MCP: the official Python SDK (`mcp`)
- Frontend: **Next.js (App Router) + TypeScript** in `/web`

## Architecture

- Layering is strict: **routers -> services -> repositories**.
  - Routers handle HTTP only (validation, status codes, serialization) and call services.
  - Services hold business logic and call repositories and the LLM layer.
  - Repositories are the only layer that talks to the database.
  - Never skip a layer (e.g. a router must not call a repository directly).
- `app/llm/` defines a common provider interface with Gemini and Anthropic implementations.
  **No module outside `app/llm/` may import `google.genai` or `anthropic` directly.**
  The active provider is selected by an environment variable.

## Assistant principles

These are product requirements, not suggestions:

1. Answers are grounded **only** in retrieved passages.
2. Every claim carries a citation in the form `Book Chapter:Verse` (e.g. `Proverbs 3:5`).
3. Tone is neutral: no doctrinal or denominational positions.
4. A post-generation verification step **removes any citation that is not in the retrieved
   passages**.
5. If no relevant passages are retrieved, the assistant says so instead of answering.

## Corpus

- The corpus is the five wisdom books, OSIS codes `Job`, `Ps`, `Prov`, `Eccl`, `Song`.
- They are defined **exactly once**, in the constant `CORPUS_BOOKS` in `app/core/corpus.py`.
  Everything else imports that constant; never hard-code the list elsewhere.
- Verses from all 66 books are loaded into the database, but **only corpus books produce
  passages and embeddings**.
- The **chat** answers only from the corpus.
- The **agent** may cite texts from other books that reach it through cross references;
  those citations are explicitly **marked as outside the corpus**.

## Testing

- **No unit test calls a real LLM API.** Unit tests use a deterministic fake provider that
  implements the `app/llm/` interface.
- Real LLM calls happen only in **evals**, which are marked (pytest marker) and run manually,
  never in the default test run or CI.

## Cost constraints

Everything runs on free tiers; design for them:

- **Gemini free tier**: low daily quota; keep calls minimal and cache where sensible.
  On the free tier prompts may be used for training, so **only public data** ever goes
  into prompts.
- **Neon Free**: 1 GB storage. Embeddings are **768 dimensions**.
- **Render Free**: 512 MB RAM, ephemeral disk, the service sleeps when idle, and there is
  **no pre-deploy command** (migrations and data loading must run another way).

## Configuration and secrets

- Configuration comes **only from environment variables** (via pydantic-settings).
- `.env.example` lists every variable with **fake** values.
- `.env` is in `.gitignore`. **Never commit secrets.**

## Definition of done

Before declaring any task finished, run all of the following and show the commands with
their **actual** output. All must pass:

```sh
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

## Attributions

The README must always include these attributions:

- **World English Bible** — [ebible.org](https://ebible.org/), public domain.
- **OpenBible.info cross references** — [openbible.info](https://www.openbible.info/labs/cross-references/), licensed CC-BY.
