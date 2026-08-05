"""Errors raised by the conversation layer.

These are separated from the classes that raise them so the API layer can
import and map them to HTTP status codes without pulling in the session
machinery.
"""

from __future__ import annotations

__all__ = [
    "CancellationError",
    "ConversationError",
    "SessionClosedError",
    "SessionLimitError",
    "SessionNotFoundError",
]


class ConversationError(Exception):
    """Base class for every conversation-layer failure."""


class SessionLimitError(ConversationError):
    """Raised when a new session would exceed ``session.max_concurrent``.

    This is a clean rejection, not a crash: the caller should return 503 with a
    ``Retry-After``. Admitting the session anyway would degrade every
    conversation already in flight, which is strictly worse than refusing one.

    Args:
        active: Sessions currently open.
        limit: Configured concurrency ceiling.
    """

    def __init__(self, active: int, limit: int) -> None:
        super().__init__(f"session limit reached: {active}/{limit} sessions active")
        self.active = active
        self.limit = limit


class SessionNotFoundError(ConversationError):
    """Raised when a session id does not resolve to a live session."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"no such session: {session_id}")
        self.session_id = session_id


class SessionClosedError(ConversationError):
    """Raised when work is submitted to a session that is shutting down."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"session {session_id} is closed")
        self.session_id = session_id


class CancellationError(ConversationError):
    """Raised when a cancellation token fires during an operation.

    Deliberately *not* :class:`asyncio.CancelledError`. Barge-in is an expected,
    recoverable conversational event -- the session survives it -- whereas
    ``CancelledError`` means the task itself is being torn down and must not be
    swallowed. Conflating them makes it impossible to write a correct
    ``except`` clause.

    Args:
        reason: Why the token was cancelled.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason
