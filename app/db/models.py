"""SQLAlchemy models: books, verses and passages (chunks with embeddings)."""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    false,
)
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
