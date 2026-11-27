"""Public-demo protection: per-IP rate limit, answer cache, query-embedding cache and
LLM quota status, all kept in Postgres because Render Free restarts lose process memory.

Every operation is best effort: if the database fails, the guard logs it and lets
the request through (retrieval needs the same database, so an outage cannot turn
into unlimited LLM calls). Without a database the guard is disabled entirely.
"""

import hashlib
import ipaddress
import json
import logging
import unicodedata
from collections.abc import AsyncIterable, AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from time import monotonic
from typing import Any, Final, Literal
from zoneinfo import ZoneInfo

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.llm import Embedder, EmbedTask, QuotaExceededError
from app.repositories import demo_state

logger = logging.getLogger(__name__)

Mode = Literal["chat", "agent"]

RATE_LIMIT_WINDOW: Final = timedelta(hours=1)
# How long /health trusts its in-memory copy of the quota status before re-reading it.
QUOTA_STATUS_TTL_SECONDS: Final = 60.0
EMBEDDINGS_SOURCE: Final = "embeddings"


def utc_now() -> datetime:
    return datetime.now(UTC)


# --- Normalization and cache keys --------------------------------------------------------


def normalize_question(text: str) -> str:
    """Lowercase, single spaces, no trailing punctuation: `What is love?` -> `what is love`."""
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    while normalized and (
        unicodedata.category(normalized[-1]).startswith("P") or normalized[-1].isspace()
    ):
        normalized = normalized[:-1]
    return normalized


def _sha256(*parts: str) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()


def answer_cache_key(mode: Mode, question: str, provider: str, model: str) -> str:
    return _sha256("answer", mode, normalize_question(question), provider, model)


def query_embedding_key(model: str, text: str) -> str:
    return _sha256("query-embedding", model, normalize_question(text))


# --- Client identification --------------------------------------------------------------

_Address = ipaddress.IPv4Address | ipaddress.IPv6Address


def _parse_ip(value: str) -> _Address | None:
    value = value.strip()
    if value.startswith("[") and "]" in value:  # [2001:db8::1]:443
        value = value[1 : value.index("]")]
    elif value.count(":") == 1:  # 203.0.113.7:51234
        value = value.split(":", 1)[0]
    try:
        return ipaddress.ip_address(value)
    except ValueError:
        return None


def _bucket(address: _Address) -> str:
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return str(address.ipv4_mapped)
        # One subscriber usually gets a whole /64; limiting single addresses is trivial to evade.
        return str(ipaddress.IPv6Network((address, 64), strict=False))
    return str(address)


def client_key(forwarded_for: str | None, peer: str | None, trusted_hops: int) -> str:
    """The rate-limit identity of a request.

    Proxies append the address they saw to `X-Forwarded-For`, so only the last
    `trusted_hops` entries were written by infrastructure we trust; anything to
    their left is whatever the client sent. With `trusted_hops = 1` (Render) the
    last entry is the client. Unparseable values fall back to the socket peer.
    """
    candidates: list[str] = []
    if trusted_hops > 0 and forwarded_for:
        entries = [entry for entry in forwarded_for.split(",") if entry.strip()]
        if len(entries) >= trusted_hops:
            candidates.append(entries[-trusted_hops])
    if peer:
        candidates.append(peer)
    for candidate in candidates:
        address = _parse_ip(candidate)
        if address is not None:
            return _bucket(address)
    return "unknown"


# --- Rate limit and quota ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    limit: int
    retry_at: datetime  # end of the current window

    def retry_after_seconds(self, now: datetime) -> int:
        return max(1, int((self.retry_at - now).total_seconds() + 0.999))

    def message(self, now: datetime) -> str:
        minutes = max(1, round(self.retry_after_seconds(now) / 60))
        when = self.retry_at.strftime("%H:%M UTC")
        unit = "minute" if minutes == 1 else "minutes"
        return (
            f"Rate limit reached: {self.limit} questions per hour. "
            f"Try again in {minutes} {unit} (at {when}). "
            "Cached example questions still work."
        )


def window_start(now: datetime, window: timedelta = RATE_LIMIT_WINDOW) -> datetime:
    """Start of the fixed window containing `now` (windows are aligned to the epoch)."""
    seconds = int(window.total_seconds())
    epoch = int(now.timestamp())
    return datetime.fromtimestamp(epoch - epoch % seconds, UTC)


@dataclass(frozen=True, slots=True)
class QuotaStatus:
    exhausted: bool
    resets_at: datetime | None = None

    @property
    def label(self) -> Literal["available", "exhausted"]:
        return "exhausted" if self.exhausted else "available"


def next_quota_reset(exhausted_at: datetime, zone: ZoneInfo) -> datetime:
    """The first local midnight in `zone` after `exhausted_at`, in UTC."""
    local = exhausted_at.astimezone(zone)
    midnight = datetime.combine(local.date() + timedelta(days=1), time(0), tzinfo=zone)
    return midnight.astimezone(UTC)


def quota_status_at(exhausted_at: datetime | None, now: datetime, zone: ZoneInfo) -> QuotaStatus:
    """Exhausted from the recorded moment until the next daily reset."""
    if exhausted_at is None:
        return QuotaStatus(exhausted=False)
    resets_at = next_quota_reset(exhausted_at, zone)
    if now >= resets_at:
        return QuotaStatus(exhausted=False)
    return QuotaStatus(exhausted=True, resets_at=resets_at)


# --- Streams ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """One serialized SSE event: its name and JSON-ready data."""

    name: str
    data: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AnswerEntry:
    """Where a finished answer goes in the cache."""

    key: str
    mode: Mode
    question: str
    provider: str
    model: str

    @classmethod
    def build(cls, mode: Mode, question: str, provider: str, model: str) -> "AnswerEntry":
        return cls(
            key=answer_cache_key(mode, question, provider, model),
            mode=mode,
            question=normalize_question(question),
            provider=provider,
            model=model,
        )


def _coalesce_tokens(events: Sequence[StreamEvent]) -> list[dict[str, Any]]:
    """Events as stored: consecutive `token` fragments merged into one."""
    stored: list[dict[str, Any]] = []
    for event in events:
        previous = stored[-1] if stored else None
        if event.name == "token" and previous is not None and previous["event"] == "token":
            previous["data"] = {"text": previous["data"]["text"] + event.data["text"]}
        else:
            stored.append({"event": event.name, "data": dict(event.data)})
    return stored


def _replayed(stored: list[dict[str, Any]]) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    for item in stored:
        data: dict[str, Any] = dict(item["data"])
        if item["event"] == "done":
            data["cached"] = True
        events.append(StreamEvent(item["event"], data))
    return events


# --- The guard --------------------------------------------------------------------------


@dataclass(slots=True)
class _QuotaMemo:
    status: QuotaStatus
    read_at: float


@dataclass(slots=True)
class DemoGuard:
    """Shared, process-wide protection state backed by Postgres (disabled without one)."""

    sessions: async_sessionmaker[AsyncSession] | None
    rate_limit_per_hour: int = 10
    cache_ttl: timedelta = timedelta(days=30)
    quota_zone: ZoneInfo = field(default_factory=lambda: ZoneInfo("America/Los_Angeles"))
    clock: Callable[[], datetime] = utc_now
    _quota_memo: _QuotaMemo | None = None

    @classmethod
    def from_settings(
        cls,
        sessions: async_sessionmaker[AsyncSession] | None,
        settings: Settings,
        clock: Callable[[], datetime] = utc_now,
    ) -> "DemoGuard":
        return cls(
            sessions=sessions,
            rate_limit_per_hour=settings.rate_limit_per_hour,
            cache_ttl=timedelta(days=settings.cache_ttl_days),
            quota_zone=ZoneInfo(settings.quota_reset_timezone),
            clock=clock,
        )

    @property
    def enabled(self) -> bool:
        return self.sessions is not None

    # Answer cache

    async def cached_answer(self, key: str) -> list[StreamEvent] | None:
        """A previous complete answer for `key`, ready to replay with `done.cached = true`."""
        if self.sessions is None:
            return None
        try:
            async with self.sessions() as session:
                stored = await demo_state.get_answer(session, key, self.clock() - self.cache_ttl)
        except Exception:
            logger.exception("answer cache read failed")
            return None
        return None if stored is None else _replayed(stored)

    async def track(
        self, events: AsyncIterable[StreamEvent], *, source: str, entry: AnswerEntry | None
    ) -> AsyncIterator[StreamEvent]:
        """Pass a live answer through, caching it on `done` and recording quota errors.

        The answer is stored before `done` is sent, so a client that disconnects
        right after receiving it does not lose the cache entry.
        """
        seen: list[StreamEvent] = []
        async for event in events:
            if event.name == "error" and event.data.get("code") == "quota_exceeded":
                await self.record_quota_exhausted(source)
            elif event.name == "done":
                await self._answered(entry, [*seen, event])
            seen.append(event)
            yield event

    async def _answered(self, entry: AnswerEntry | None, events: list[StreamEvent]) -> None:
        if self.sessions is None:
            return
        now = self.clock()
        try:
            async with self.sessions.begin() as session:
                await demo_state.clear_quota_exhausted(session)
                if entry is not None:
                    await demo_state.put_answer(
                        session,
                        key=entry.key,
                        mode=entry.mode,
                        question=entry.question,
                        provider=entry.provider,
                        model=entry.model,
                        events=_coalesce_tokens(events),
                        created_at=now,
                    )
                    await demo_state.delete_answers_before(session, now - self.cache_ttl)
        except Exception:
            logger.exception("answer cache write failed")
            return
        if self._quota_memo is not None and self._quota_memo.status.exhausted:
            self._quota_memo = _QuotaMemo(QuotaStatus(exhausted=False), monotonic())

    # Rate limit

    async def consume_rate_limit(self, client: str) -> RateLimitDecision:
        """Count one request for `client`; `allowed` is False once the window is full."""
        now = self.clock()
        start = window_start(now)
        decision = RateLimitDecision(True, self.rate_limit_per_hour, start + RATE_LIMIT_WINDOW)
        if self.sessions is None:
            return decision
        try:
            async with self.sessions.begin() as session:
                count = await demo_state.consume_request(
                    session, client, start, self.rate_limit_per_hour
                )
                await demo_state.delete_windows_before(session, start)
        except Exception:
            logger.exception("rate limit check failed; letting the request through")
            return decision
        if count is None:
            return RateLimitDecision(False, decision.limit, decision.retry_at)
        return decision

    # Query embeddings

    async def cached_query_embedding(self, key: str) -> list[float] | None:
        if self.sessions is None:
            return None
        try:
            async with self.sessions() as session:
                return await demo_state.get_query_embedding(
                    session, key, self.clock() - self.cache_ttl
                )
        except Exception:
            logger.exception("query embedding cache read failed")
            return None

    async def store_query_embedding(self, key: str, model: str, vector: list[float]) -> None:
        if self.sessions is None:
            return
        now = self.clock()
        try:
            async with self.sessions.begin() as session:
                await demo_state.put_query_embedding(session, key, model, vector, now)
                await demo_state.delete_query_embeddings_before(session, now - self.cache_ttl)
        except Exception:
            logger.exception("query embedding cache write failed")

    # Quota

    async def record_quota_exhausted(self, source: str) -> None:
        if self.sessions is None:
            return
        now = self.clock()
        try:
            async with self.sessions.begin() as session:
                await demo_state.mark_quota_exhausted(session, source, now)
        except Exception:
            logger.exception("recording the quota status failed")
            return
        status = quota_status_at(now, now, self.quota_zone)
        self._quota_memo = _QuotaMemo(status, monotonic())

    async def quota_status(self) -> QuotaStatus | None:
        """Current status (memoized briefly so health checks barely touch the database).

        None when there is no database or it cannot be read.
        """
        if self.sessions is None:
            return None
        memo = self._quota_memo
        if memo is not None and monotonic() - memo.read_at < QUOTA_STATUS_TTL_SECONDS:
            return self._recheck(memo.status)
        try:
            async with self.sessions() as session:
                exhausted_at = await demo_state.quota_exhausted_at(session)
        except Exception:
            logger.exception("reading the quota status failed")
            return None
        status = quota_status_at(exhausted_at, self.clock(), self.quota_zone)
        self._quota_memo = _QuotaMemo(status, monotonic())
        return status

    def _recheck(self, status: QuotaStatus) -> QuotaStatus:
        """A memoized `exhausted` status expires at its reset time even inside the TTL."""
        if status.resets_at is not None and self.clock() >= status.resets_at:
            return QuotaStatus(exhausted=False)
        return status


class CachedQueryEmbedder:
    """`Embedder` that reuses query embeddings by normalized text and records quota errors.

    Only single-text `QUERY` calls are cached (what search, chat and the agent make);
    document embeddings for ingestion pass straight through.
    """

    def __init__(self, inner: Embedder, guard: DemoGuard) -> None:
        self._inner = inner
        self._guard = guard

    @property
    def embed_model(self) -> str:
        return self._inner.embed_model

    async def embed(self, texts: Sequence[str], task: EmbedTask) -> list[list[float]]:
        if task is not EmbedTask.QUERY or len(texts) != 1 or not self._guard.enabled:
            return await self._embed(texts, task)
        model = self.embed_model
        key = query_embedding_key(model, texts[0])
        cached = await self._guard.cached_query_embedding(key)
        if cached is not None:
            return [cached]
        vectors = await self._embed(texts, task)
        await self._guard.store_query_embedding(key, model, vectors[0])
        return vectors

    async def _embed(self, texts: Sequence[str], task: EmbedTask) -> list[list[float]]:
        try:
            return await self._inner.embed(texts, task)
        except QuotaExceededError:
            await self._guard.record_quota_exhausted(EMBEDDINGS_SOURCE)
            raise
