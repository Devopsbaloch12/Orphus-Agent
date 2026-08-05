from __future__ import annotations

import pytest

from orphus.conversation.errors import SessionLimitError, SessionNotFoundError
from orphus.conversation.manager import SessionManager
from orphus.conversation.session import SessionState


async def test_sessions_are_isolated_and_capacity_is_enforced() -> None:
    manager = SessionManager(max_concurrent=2, system_prompt="brief")
    first = await manager.create(metadata={"voice": "tara"})
    second = await manager.create()
    first.history.add_user("hello")
    assert len(second.history) == 0
    with pytest.raises(SessionLimitError):
        await manager.create()


async def test_close_removes_and_cleans_session() -> None:
    manager = SessionManager()
    session = await manager.create()
    await manager.close(session.session_id, reason="client_left")
    assert session.state is SessionState.CLOSED
    with pytest.raises(SessionNotFoundError):
        await manager.get(session.session_id)


async def test_shutdown_closes_every_session() -> None:
    manager = SessionManager()
    sessions = [await manager.create() for _ in range(3)]
    await manager.aclose()
    assert manager.active_count == 0
    assert all(session.state is SessionState.CLOSED for session in sessions)
    with pytest.raises(RuntimeError, match="closed"):
        await manager.create()
