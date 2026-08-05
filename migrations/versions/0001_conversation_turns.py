"""Create conversation turn records."""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None


def upgrade() -> None:
    op.create_table(
        "conversation_turns",
        sa.Column("turn_id", sa.String(64), primary_key=True),
        sa.Column("session_id", sa.String(64), nullable=False),
        sa.Column("user_text", sa.Text(), nullable=False),
        sa.Column("assistant_text", sa.Text(), nullable=False),
        sa.Column("started_at", sa.Float(), nullable=False),
        sa.Column("asr_latency_ms", sa.Float()),
        sa.Column("llm_first_token_ms", sa.Float()),
        sa.Column("tts_first_audio_ms", sa.Float()),
        sa.Column("end_to_end_ms", sa.Float()),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completion_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_conversation_turns_session_id", "conversation_turns", ["session_id"])


def downgrade() -> None:
    op.drop_table("conversation_turns")
