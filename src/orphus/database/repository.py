"""Async PostgreSQL persistence boundary."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from orphus.config.settings import DatabaseSettings
from orphus.database.models import ConversationTurnRecord
from orphus.domain.types import ConversationTurn


class TurnRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._sessions = async_sessionmaker(engine, expire_on_commit=False)

    @classmethod
    def connect(cls, settings: DatabaseSettings) -> TurnRepository:
        engine = create_async_engine(
            settings.url,
            pool_size=settings.pool_size,
            max_overflow=settings.max_overflow,
            pool_timeout=settings.pool_timeout_s,
            echo=settings.echo,
            pool_pre_ping=True,
        )
        return cls(engine)

    async def save(self, turn: ConversationTurn) -> None:
        usage = turn.usage
        record = ConversationTurnRecord(
            turn_id=turn.turn_id,
            session_id=turn.session_id,
            user_text=turn.user_text,
            assistant_text=turn.assistant_text,
            started_at=turn.started_at,
            asr_latency_ms=turn.asr_latency_ms,
            llm_first_token_ms=turn.llm_first_token_ms,
            tts_first_audio_ms=turn.tts_first_audio_ms,
            end_to_end_ms=turn.end_to_end_ms,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
        async with self._sessions() as session:
            session.add(record)
            await session.commit()

    async def health(self) -> None:
        async with self._engine.connect() as connection:
            await connection.exec_driver_sql("SELECT 1")

    async def aclose(self) -> None:
        await self._engine.dispose()

