"""books, verses and passages with a pgvector HNSW index

Revision ID: 0001
Revises:
Create Date: 2026-11-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "books",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("osis_code", sa.String(10), nullable=False, unique=True),
        sa.Column("name", sa.String(40), nullable=False),
        sa.Column("testament", sa.String(2), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False, unique=True),
        sa.Column("in_corpus", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.CheckConstraint("testament IN ('OT', 'NT')", name="ck_books_testament"),
    )

    op.create_table(
        "verses",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "book_id", sa.Integer(), sa.ForeignKey("books.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("chapter", sa.Integer(), nullable=False),
        sa.Column("verse", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.UniqueConstraint("book_id", "chapter", "verse", name="uq_verses_book_chapter_verse"),
    )

    op.create_table(
        "passages",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "book_id", sa.Integer(), sa.ForeignKey("books.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("chapter", sa.Integer(), nullable=False),
        sa.Column("start_verse", sa.Integer(), nullable=False),
        sa.Column("end_verse", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("reference", sa.String(60), nullable=False),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column("embedded_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint(
            "book_id", "chapter", "start_verse", "end_verse", name="uq_passages_span"
        ),
    )
    op.create_index(
        "ix_passages_embedding_hnsw",
        "passages",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_passages_embedding_hnsw", table_name="passages")
    op.drop_table("passages")
    op.drop_table("verses")
    op.drop_table("books")
