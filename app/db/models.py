"""SQLAlchemy models: the Bible data, cross references and the public-demo state."""

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    false,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_DIMENSIONS = 768


class Base(DeclarativeBase):
    pass


class Book(Base):
    __tablename__ = "books"
    __table_args__ = (CheckConstraint("testament IN ('OT', 'NT')", name="ck_books_testament"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    osis_code: Mapped[str] = mapped_column(String(10), unique=True)
    name: Mapped[str] = mapped_column(String(40))
    testament: Mapped[str] = mapped_column(String(2))
    position: Mapped[int] = mapped_column(Integer, unique=True)
    in_corpus: Mapped[bool] = mapped_column(Boolean, server_default=false())


class Verse(Base):
    __tablename__ = "verses"
    __table_args__ = (
        UniqueConstraint("book_id", "chapter", "verse", name="uq_verses_book_chapter_verse"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    book_id: Mapped[int] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"))
    chapter: Mapped[int] = mapped_column(Integer)
    verse: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)


class Passage(Base):
    __tablename__ = "passages"
    __table_args__ = (
        UniqueConstraint("book_id", "chapter", "start_verse", "end_verse", name="uq_passages_span"),
        Index(
            "ix_passages_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    book_id: Mapped[int] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"))
    chapter: Mapped[int] = mapped_column(Integer)
    start_verse: Mapped[int] = mapped_column(Integer)
    end_verse: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    reference: Mapped[str] = mapped_column(String(60))
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS), nullable=True
    )
    embedded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class CrossReference(Base):
    """OpenBible.info cross reference: one source verse to a target verse or range.

    Targets can be in any of the 66 books; ranges may cross chapters
    (`to_chapter:to_start_verse` .. `to_end_chapter:to_end_verse`).
    """

    __tablename__ = "cross_references"
    __table_args__ = (
        UniqueConstraint(
            "from_book_id",
            "from_chapter",
            "from_verse",
            "to_book_id",
            "to_chapter",
            "to_start_verse",
            "to_end_chapter",
            "to_end_verse",
            name="uq_cross_references_pair",
        ),
        Index(
            "ix_cross_references_source",
            "from_book_id",
            "from_chapter",
            "from_verse",
            "votes",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    from_book_id: Mapped[int] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"))
    from_chapter: Mapped[int] = mapped_column(SmallInteger)
    from_verse: Mapped[int] = mapped_column(SmallInteger)
    to_book_id: Mapped[int] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"))
    to_chapter: Mapped[int] = mapped_column(SmallInteger)
    to_start_verse: Mapped[int] = mapped_column(SmallInteger)
    to_end_chapter: Mapped[int] = mapped_column(SmallInteger)
    to_end_verse: Mapped[int] = mapped_column(SmallInteger)
    votes: Mapped[int] = mapped_column(Integer)


class RateLimitWindow(Base):
    """Requests one client made in one fixed window (`window_start` .. + 1 hour)."""

    __tablename__ = "rate_limit_windows"
    __table_args__ = (Index("ix_rate_limit_windows_start", "window_start"),)

    client_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    requests: Mapped[int] = mapped_column(Integer)


class CachedAnswer(Base):
    """A complete streamed answer (its SSE events), replayed for the same question."""

    __tablename__ = "answer_cache"
    __table_args__ = (Index("ix_answer_cache_created_at", "created_at"),)

    key: Mapped[str] = mapped_column(String(64), primary_key=True)  # SHA-256 hex
    mode: Mapped[str] = mapped_column(String(10))
    question: Mapped[str] = mapped_column(Text)  # normalized, for inspection only
    provider: Mapped[str] = mapped_column(String(20))
    model: Mapped[str] = mapped_column(String(80))
    events: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class CachedQueryEmbedding(Base):
    """Embedding of a normalized search query, reused instead of calling the API."""

    __tablename__ = "query_embedding_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)  # SHA-256 hex
    model: Mapped[str] = mapped_column(String(80))
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class LLMQuotaStatus(Base):
    """Single row (id = 1): when an LLM quota was last reported exhausted, and by whom.

    `source` is the chat provider's name, or `embeddings` for the query-embedding API.
    """

    __tablename__ = "llm_quota_status"
    __table_args__ = (CheckConstraint("id = 1", name="ck_llm_quota_status_single_row"),)

    id: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    source: Mapped[str] = mapped_column(String(20))
    exhausted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
