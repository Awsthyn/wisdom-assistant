"""public-demo protection: rate limits, answer and embedding caches, quota status

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "rate_limit_windows",
        sa.Column("client_key", sa.String(64), primary_key=True),
        sa.Column("window_start", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("requests", sa.Integer(), nullable=False),
    )
    op.create_index("ix_rate_limit_windows_start", "rate_limit_windows", ["window_start"])

    op.create_table(
        "answer_cache",
        sa.Column("key", sa.String(64), primary_key=True),
        sa.Column("mode", sa.String(10), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("provider", sa.String(20), nullable=False),
        sa.Column("model", sa.String(80), nullable=False),
        sa.Column("events", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_answer_cache_created_at", "answer_cache", ["created_at"])

    op.create_table(
        "query_embedding_cache",
        sa.Column("key", sa.String(64), primary_key=True),
        sa.Column("model", sa.String(80), nullable=False),
        sa.Column("embedding", Vector(768), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "llm_quota_status",
        sa.Column("id", sa.SmallInteger(), primary_key=True),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column("exhausted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("id = 1", name="ck_llm_quota_status_single_row"),
    )


def downgrade() -> None:
    op.drop_table("llm_quota_status")
    op.drop_table("query_embedding_cache")
    op.drop_index("ix_answer_cache_created_at", table_name="answer_cache")
    op.drop_table("answer_cache")
    op.drop_index("ix_rate_limit_windows_start", table_name="rate_limit_windows")
    op.drop_table("rate_limit_windows")
