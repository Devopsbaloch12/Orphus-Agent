"""Concurrent session registry and lifecycle reaper."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from orphus.conversation.errors import SessionLimitError, SessionNotFoundError
from orphus.conversation.history import History
from orphus.conversation.session import Session
from orphus.domain.types import SessionId

__all__ = ["SessionManager"]


class SessionManager:
    """Own isolated sessions and enforce the configured concurrency ceiling."""

    def __init__(
        self,
        *,
        max_concurrent: int = 20,
        idle_timeout_s: float = 300.0,
        max_duration_s: float = 3600.0,
        history_max_turns: int = 20,
        history_max_chars: int = 8000,
        system_prompt: str | None = None,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self._limit = max_concurrent
        self._idle_timeout = idle_timeout_s
        self._max_duration = max_duration_s
        self._history_max_turns = history_max_turns
        self._history_max_chars = history_max_chars
        self._system_prompt = system_prompt
        self._sessions: dict[SessionId, Session] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    @property
    def capacity(self) -> int:
        return self._limit

    async def create(self, *, metadata: Mapping[str, Any] | None = None) -> Session:
        async with self._lock:
            if self._closed:
                raise RuntimeError("session manager is closed")
            if len(self._sessions) >= self._limit:
                raise SessionLimitError(len(self._sessions), self._limit)
            history = History(
                system_prompt=self._system_prompt,
                max_turns=self._history_max_turns,
                max_chars=self._history_max_chars,
            )
            session = Session(history=history, metadata=dict(metadata or {}))
            self._sessions[session.session_id] = session
            return session

    async def get(self, session_id: str) -> Session:
        async with self._lock:
            session = self._sessions.get(SessionId(session_id))
        if session is None:
            raise SessionNotFoundError(session_id)
        return session

    async def close(self, session_id: str, *, reason: str = "closed") -> None:
        async with self._lock:
            session = self._sessions.pop(SessionId(session_id), None)
        if session is None:
            raise SessionNotFoundError(session_id)
        await session.aclose(reason=reason)

    async def reap(self) -> int:
        """Close idle or over-duration sessions and return the number reaped."""
        async with self._lock:
            expired = [
                session
                for session in self._sessions.values()
                if session.idle_s >= self._idle_timeout
                or session.age_s >= self._max_duration
            ]
            for session in expired:
                self._sessions.pop(session.session_id, None)
        await asyncio.gather(
            *(session.aclose(reason="timeout") for session in expired),
            return_exceptions=True,
        )
        return len(expired)

    async def aclose(self) -> None:
        async with self._lock:
            self._closed = True
            sessions = list(self._sessions.values())
            self._sessions.clear()
        await asyncio.gather(
            *(session.aclose(reason="shutdown") for session in sessions),
            return_exceptions=True,
        )

