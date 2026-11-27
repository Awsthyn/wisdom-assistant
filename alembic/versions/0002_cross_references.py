"""cross_references from OpenBible.info

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "cross_references",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "from_book_id",
            sa.Integer(),
            sa.ForeignKey("books.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("from_chapter", sa.SmallInteger(), nullable=False),
        sa.Column("from_verse", sa.SmallInteger(), nullable=False),
        sa.Column(
            "to_book_id",
            sa.Integer(),
            sa.ForeignKey("books.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("to_chapter", sa.SmallInteger(), nullable=False),
        sa.Column("to_start_verse", sa.SmallInteger(), nullable=False),
        sa.Column("to_end_chapter", sa.SmallInteger(), nullable=False),
        sa.Column("to_end_verse", sa.SmallInteger(), nullable=False),
        sa.Column("votes", sa.Integer(), nullable=False),
        sa.UniqueConstraint(
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
    )
    op.create_index(
        "ix_cross_references_source",
        "cross_references",
        ["from_book_id", "from_chapter", "from_verse", "votes"],
    )


def downgrade() -> None:
    op.drop_index("ix_cross_references_source", table_name="cross_references")
    op.drop_table("cross_references")
