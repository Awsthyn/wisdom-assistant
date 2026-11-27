# wisdom-assistant

A study assistant for the five wisdom books of the World English Bible — Job, Psalms,
Proverbs, Ecclesiastes and Song of Songs — with semantic search, RAG chat that cites
verses, a tool-calling agent and an MCP server.

> Status: in development. Ingestion, search, RAG chat, the tool-calling agent, the MCP server and the public-demo protection (rate limit, caches, quota status) are done; the web UI comes next.

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
uv run python -m app.scripts.ingest xrefs        # OpenBible.info cross references; --refresh
uv run python -m app.scripts.ingest status
```

`embed` only fills embeddings that are still NULL and commits after each batch, so it
can be stopped (Ctrl+C, error, quota) and resumed. It paces itself under `EMBED_RPM`
requests and `EMBED_TPM` estimated tokens per minute. When the daily quota runs out it
stops cleanly and asks you to run it again the next day.

`xrefs` loads [OpenBible.info cross references](https://www.openbible.info/labs/cross-references/)
(CC-BY) with **at least 5 votes**, for all 66 books, because targets outside the wisdom
books exist in `verses`. It rebuilds the table in one transaction, so re-running it is
safe. Current numbers:

- 344,799 source rows, of which 118,592 are loaded. 226,202 have fewer than 5 votes
  and 5 are ranges that span two books.
- 23,507 cross references start in a corpus book.
- The table takes 16.9 MB with its indexes, and the whole database 45 MB.
- Target ranges may cross chapters (`Romans 1:32-2:1`), so the table also stores
  `to_end_chapter`.

## Search

`GET /api/v1/search?q=...&k=8&book=Prov`

- **Exact references** (`John 3:16`, `Jn 3:16-18`, `Psalm 23`, `1 Cor 13:4`) are
  read straight from the database, from any of the 66 books, **without an embedding
  call**. Books outside the corpus come back with `in_corpus: false`. Common English
  abbreviations and ordinals (`1 Jn`, `First John`, `II Kings`) are understood;
  impossible references such as `John 99:1` return 422, and missing verses return 404.
  `k` and `book` do not apply to them.
- **Anything else** is embedded as a query and matched by cosine similarity against
  the corpus passages (HNSW index). `book` restricts results to one corpus book
  (`Job`, `Ps`, `Prov`, `Eccl`, `Song`); any other value returns 422. With a filter,
  pgvector's iterative index scan keeps searching until `k` matches are found.
- Each result has `reference`, `book`, `book_name`, `chapter`, `start_verse`,
  `end_verse`, `text`, `score` (null for exact references), `in_corpus` and `url`
  (the chapter on ebible.org).
- `q` is 3–300 characters and `k` is 1–20. If the embedding quota is exhausted, semantic
  search returns 503; exact references keep working.

## Chat

`POST /api/v1/chat` with `{"question": "...", "history": [...]}` (`question` 3–500
characters; `history` at most the last 6 `user`/`assistant` messages) answers as
Server-Sent Events:

| Event | Data |
|---|---|
| `sources` | the retrieved passages: `id`, `reference`, `text`, `url`, `score` |
| `token` | `{"text": ...}` answer fragments, as the LLM produces them |
| `citations` | `valid` and `discarded` citations (each discarded one with a `reason`) |
| `done` | `{"model": ...}`, plus `"cached": true` when replayed from the cache |
| `error` | `{"code", "message"}` with code `quota_exceeded`, `no_results` or `internal` |

```sh
curl -N -X POST http://localhost:8000/api/v1/chat -H 'Content-Type: application/json' \
  -d '{"question": "What do the wisdom books say about caring for the poor?"}'
```

How the answer stays grounded:

- The top `CHAT_TOP_K` (8) corpus passages are retrieved. If the best one scores below
  `CHAT_MIN_SCORE`, the reply is `error` / `no_results` and **the LLM is not called**.
  The default 0.57 was calibrated on real questions with `gemini-embedding-001`
  (on-topic 0.60–0.73, off-topic 0.49–0.59); recalibrate if the embedding model changes.
- The prompt gives the passages numbered and delimited (`<passages>`, `<question>`),
  and the system prompt carries the assistant principles. Prompt delimiters are
  stripped from user text, and the system prompt says that user text never changes
  the rules.
- After generation, **code** (not the model) extracts every `[...]` citation,
  including grouped ones like `[Proverbs 14:31, 22:9]`, normalizes it with the
  reference parser and keeps it only if its book is in the corpus and every verse
  it names is in the retrieved passages. Anything else is reported as discarded.

## Agent

`POST /api/v1/agent` with `{"question": "..."}` runs a tool-calling loop. The model
decides which tools to call, may call them several times, and then answers. It works
with both providers through `generate_with_tools`: Gemini function calling and
Anthropic tool use are translated inside `app/llm/`.

The tools are defined once in [app/tools/](app/tools/), each with its JSON Schema
generated from a Pydantic model. The MCP server reuses them.

| Tool | What it returns |
|---|---|
| `lookup_reference(reference)` | Exact text of a reference from any of the 66 books, with `in_corpus` |
| `search_passages(query, k=5, book=None)` | Semantic search over the five wisdom books |
| `cross_references(reference, limit=10)` | Cross references by votes, each with its text and `in_corpus` |

The agent answers as Server-Sent Events:

| Event | Data |
|---|---|
| `tool_call` | `{"name", "arguments"}` |
| `tool_result` | `{"name", "summary", "is_error"}`, a short line, not the full result |
| `token` | `{"text"}`, the final answer |
| `citations` | `valid` / `discarded`, each with `outside_corpus` |
| `done` | `{"model", "iterations", "tool_calls"}`, plus `"cached": true` when replayed |
| `error` | `{"code", "message"}` with code `quota_exceeded`, `internal` or `iteration_limit` |

- The agent stops after **5 model calls and 8 tool calls**. On its last turn the model
  is told to answer with what it has.
- Bad arguments, unknown references and unknown tools are returned to the model as tool
  errors, so it can correct itself; they never end the run.
- Citations get the same code-level check as in chat, now against every text the tools
  returned. A text from outside the wisdom books is **valid only if a tool returned
  it**, and it is flagged `outside_corpus: true` so the UI can present it as a cross
  reference.
- The answer arrives as a single `token` event, because `generate_with_tools` is not a
  streaming call. The `tool_call` and `tool_result` events show progress while the
  agent works.

## Protecting the public demo

The demo runs on free tiers, so `/chat` and `/agent` are guarded. All the state lives in
Postgres (migration `0003`), because Render Free restarts the process and loses memory.

- **Rate limit.** `RATE_LIMIT_PER_HOUR` (10) requests per client IP per fixed one-hour
  window, shared by `/chat` and `/agent`. The 11th gets `429` with a `Retry-After`
  header and a message saying when to try again. The IP comes from `X-Forwarded-For`:
  proxies append the address they saw, so only the last `TRUSTED_PROXY_HOPS` (1 on
  Render) entries are trusted and anything a client prepends is ignored; invalid values
  fall back to the socket address, and IPv6 clients are grouped by `/64`. `/search` is
  not rate limited (it costs at most one embedding).
- **Answer cache.** Key = SHA-256 of (mode, normalized question, provider, model);
  normalized means lowercase, single spaces and no final punctuation, so
  `What is love?` and `what is love` share it. A hit replays the stored events
  (`sources`, `token`, `citations`, `done` with `"cached": true`) without calling the
  LLM and without counting against the rate limit. Entries live `CACHE_TTL_DAYS` (30).
  Chat questions sent with `history` are neither cached nor replayed.
- **Query-embedding cache.** The same normalized text reuses its stored vector in
  search, chat and the agent's `search_passages`.
- **Quota status.** When the provider reports its quota exhausted, the visitor gets
  `error` / `quota_exceeded` ("The free AI quota for today is used up. Cached example
  questions still work: try one of them.") and the moment is stored. `GET /health`
  then reports `"llm_quota": "exhausted"` and `llm_quota_resets_at` until the next
  daily reset (midnight in `QUOTA_RESET_TIMEZONE`, Pacific time for Gemini), so the UI
  can warn before anyone asks. The next successful answer clears it (a short
  per-minute 429 looks the same as the daily one). Without a database `llm_quota` is
  omitted.
- **Examples.** `GET /api/v1/examples` returns six curated questions. Fill the cache
  with their real answers after deploying (and after changing the model):

```sh
uv run python -m app.scripts.warm_cache            # chat answers; skips cached ones
uv run python -m app.scripts.warm_cache --agent    # agent answers too
```

## MCP server

The same three tools are available to any MCP client (Claude Desktop, Claude Code, MCP
Inspector...) over **stdio**:

```sh
uv run python -m app.mcp_server
```

- **Tools.** `lookup_reference`, `search_passages` and `cross_references`. They are
  not redefined for MCP: each one publishes the JSON Schema from
  [app/tools/](app/tools/) and runs the same code as the agent.
- **Resource.** `wisdom://attribution` (Markdown) lists the licenses of the returned
  texts.
- **Configuration.** The server reads `DATABASE_URL` from `.env` (the same Neon
  database as the API). It uses `GEMINI_API_KEY` only to embed `search_passages`
  queries, whatever `LLM_PROVIDER` is set to. `lookup_reference` and
  `cross_references` work without any API key.
- **Logs.** stdout carries the protocol, so logs go to stderr.

The commands below run uv with `--directory` so that `.env` is found whatever the
client's working directory is. Find your uv path with `where uv` (Windows) or
`which uv` (macOS/Linux).

### MCP Inspector

```sh
# Interactive UI in the browser:
npx @modelcontextprotocol/inspector uv --directory C:\path\to\wisdom-assistant run python -m app.mcp_server

# Or from the terminal:
npx @modelcontextprotocol/inspector --cli uv --directory C:\path\to\wisdom-assistant run python -m app.mcp_server --method tools/list
```

In the UI, open **Tools**, pick `lookup_reference` and call it with
`{"reference": "Proverbs 3:5-6"}`. Under **Resources** you can read
`wisdom://attribution`.

### Claude Desktop (Windows)

Edit `%APPDATA%\Claude\claude_desktop_config.json` (Settings → Developer → Edit
Config). Use absolute paths and doubled backslashes:

```json
{
  "mcpServers": {
    "wisdom-assistant": {
      "command": "C:\\Users\\YOUR_USER\\.local\\bin\\uv.exe",
      "args": [
        "--directory",
        "C:\\path\\to\\wisdom-assistant",
        "run",
        "python",
        "-m",
        "app.mcp_server"
      ]
    }
  }
}
```

Restart Claude Desktop. The three tools appear under the tools (🔨) menu.

### Claude Code

```sh
claude mcp add wisdom-assistant --scope user -- uv --directory C:\path\to\wisdom-assistant run python -m app.mcp_server
```

Run `claude mcp list` to check that it connects. Use `--scope project` instead to
share the server through a `.mcp.json` file in a repository.

## Attributions

- **World English Bible** — [ebible.org](https://ebible.org/), public domain.
- **Cross references** — [OpenBible.info](https://www.openbible.info/labs/cross-references/),
  licensed under [CC-BY](https://creativecommons.org/licenses/by/4.0/).
