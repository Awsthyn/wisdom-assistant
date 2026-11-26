# wisdom-assistant

A study assistant for the five wisdom books of the World English Bible — Job, Psalms,
Proverbs, Ecclesiastes and Song of Songs — with semantic search, RAG chat that cites
verses, a tool-calling agent and an MCP server.

> Status: in development. Corpus ingestion is done; search and chat come next.

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
Integration tests need a disposable Postgres with pgvector (its schema is dropped and
recreated; never point this at Neon). Without `TEST_DATABASE_URL` they are skipped
locally; CI always runs them against a `pgvector/pgvector:pg17` service container.

```sh
docker run -d --name wisdom-pg -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=wisdom_test   -p 54329:5432 pgvector/pgvector:pg17
TEST_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:54329/wisdom_test uv run pytest
```

To check your real keys and models (spends a tiny bit of quota):

```sh
uv run python -m app.scripts.llm_smoke
```

### LLM providers

`LLM_PROVIDER=gemini` (default) or `anthropic`. Embeddings always use Gemini
(`gemini-embedding-001`, 768 dimensions), because Anthropic has no embeddings API.

## Corpus

**Scope.** The semantic corpus is the five wisdom books: Job, Psalms, Proverbs,
Ecclesiastes and Song of Songs (`CORPUS_BOOKS` in `app/core/corpus.py`). A focused
corpus keeps retrieval precise, the embedding job inside one day of Gemini's free
quota, and the evaluation set small enough to curate by hand.

All 66 books of the WEB (Protestant canon, `engwebp`) are still loaded as plain verses
(31,098 verses; 4,785 of them in the corpus). The rest of the Bible is reference text
only: it is reachable through exact-reference lookups and the agent's cross references,
always marked as outside the corpus, and it is never embedded or used by the chat.

The source is ebible.org's verse-per-line file. Five verse numbers there have no text
(e.g. Luke 17:36, Acts 8:37), because the WEB omits them on text-critical grounds;
they are skipped. Psalm superscriptions are part of verse 1, as in the WEB.

**Chunking.** Each chapter is cut into windows of 5 verses that overlap by 2
(verses 1-5, 4-8, 7-11, ...), and a window never crosses a chapter boundary; the last
window of a chapter may be shorter. Five verses give the embedding enough context to
capture a thought, without diluting it across several topics; the overlap keeps a
thought that straddles two windows retrievable from either one. In Proverbs a window
can mix unrelated sayings; phase B8's evals decide whether a different strategy pays
off there. Each passage stores its text with verse markers (`[5] Trust in the LORD...`)
and a citation such as `Proverbs 3:4-8`.

This produces **1,512 passages** (Job 343, Psalms 769, Proverbs 296, Ecclesiastes 69,
Song of Songs 35). That is fewer than "verses ÷ 3" suggests because many chapters are
short (most psalms) and a final window that would only repeat verses already covered
is not emitted.

**Size on disk.** Measured on Neon after loading all verses and passages with 50
embeddings: 16.3 MB. Each 768-dim vector takes ~3 KB, so the full 1,512 embeddings
plus the HNSW index add roughly 10 MB: **about 25-30 MB in total (estimate), under 3%
of Neon Free's 1 GB.**

### Ingestion

Uses `DATABASE_URL_DIRECT` (falls back to `DATABASE_URL`). Every step is idempotent.

```sh
uv run alembic upgrade head                      # tables + pgvector extension + HNSW index
uv run python -m app.scripts.ingest download     # -> data/raw/ (git-ignored)
uv run python -m app.scripts.ingest load         # 66 books, verses, sanity checks
uv run python -m app.scripts.ingest chunk        # passages for the 5 corpus books
uv run python -m app.scripts.ingest embed        # resumable; --limit N to try a few
uv run python -m app.scripts.ingest status
```

`embed` only fills embeddings that are still NULL and commits after each batch, so it
can be stopped (Ctrl+C, error, quota) and resumed. It paces itself under `EMBED_RPM`
requests and `EMBED_TPM` estimated tokens per minute. When the daily quota runs out it
stops cleanly and asks you to run it again the next day.

## Attributions

- **World English Bible** — [ebible.org](https://ebible.org/), public domain.
- **Cross references** — [OpenBible.info](https://www.openbible.info/labs/cross-references/),
  licensed under [CC-BY](https://creativecommons.org/licenses/by/4.0/).
