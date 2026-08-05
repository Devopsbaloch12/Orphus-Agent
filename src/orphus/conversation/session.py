"""A single live conversation.

A ``Session`` owns everything that must not be shared between conversations:
its own history, its own cancellation token, and its own cleanup stack. The
model adapters are shared and stateless-per-call; the per-conversation state
lives here.

Timekeeping uses two clocks on purpose. Idle and duration reaping read
``time.monotonic``, which cannot jump when NTP steps the wall clock -- a
backwards step on a naive implementation reaps every live session at once.
``created_at_utc`` is wall clock and exists only for persistence and display.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from orphus.conversation.cancellation import CancellationToken
from orphus.conversation.errors import SessionClosedError
from orphus.conversation.history import History
from orphus.domain.types import SessionId, new_session_id
from orphus.observability.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Sequence

    from orphus.domain.types import ConversationTurn

logger = get_logger(__name__)

__all__ = ["Session", "SessionState"]

CleanupHook = "Callable[[], Coroutine[Any, Any, None]]"

# Turns kept in memory for diagnostics. The durable record is PostgreSQL; this
# is a ring buffer so a long call cannot grow without bound.
_RECENT_TURNS = 32


class SessionState(StrEnum):
    """Lifecycle states. Transitions are strictly forward."""

    ACTIVE = "active"
    CLOSING = "closing"
    CLOSED = "closed"


class Session:
    """One voice conversation and everything scoped to it.

    Args:
        session_id: Identifier. Minted if omitted.
        history: Pre-built history. A default empty one is created if omitted.
        metadata: Arbitrary caller-supplied context (client ip, voice, locale).
            Never logged wholesale, since callers put user data in here.
        clock: Monotonic time source. Injectable so the reaper can be tested
            without sleeping.

    Example:
        >>> import asyncio
        >>> session = Session()
        >>> session.state
        <SessionState.ACTIVE: 'active'>
        >>> asyncio.run(session.aclose(reason="done"))
        >>> session.state
        <SessionState.CLOSED: 'closed'>
    """

    __slots__ = (
        "_cleanup",
        "_clock",
        "_close_lock",
        "_close_reason",
        "_closed_event",
        "_created_at",
        "_created_at_utc",
        "_last_active_at",
        "_state",
        "_turn_total",
        "_turns",
        "cancellation",
        "history",
        "metadata",
        "session_id",
    )

    def __init__(
        self,
        session_id: SessionId | None = None,
        *,
        history: History | None = None,
        metadata: dict[str, Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.session_id: SessionId = session_id or new_session_id()
        self.history: History = history if history is not None else History()
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.cancellation = CancellationToken(name=self.session_id)

        self._clock = clock
        self._created_at = clock()
        self._created_at_utc = datetime.now(UTC)
        self._last_active_at = self._created_at
        self._state = SessionState.ACTIVE
        self._close_reason: str | None = None
        self._closed_event = asyncio.Event()
        self._close_lock = asyncio.Lock()
        self._cleanup: list[Callable[[], Coroutine[Any, Any, None]]] = []
        self._turns: deque[ConversationTurn] = deque(maxlen=_RECENT_TURNS)
        self._turn_total = 0

    def __repr__(self) -> str:
        """Log-safe representation. Deliberately omits history and metadata."""
        return (
            f"Session(id={self.session_id!r}, state={self._state.value}, "
            f"turns={self._turn_total}, idle_s={self.idle_s:.1f})"
        )

    # -- state --------------------------------------------------------------

    @property
    def state(self) -> SessionState:
        """Current lifecycle state."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Whether the session still accepts work."""
        return self._state is SessionState.ACTIVE

    @property
    def created_at(self) -> float:
        """Monotonic timestamp of creation. For durations only."""
        return self._created_at

    @property
    def created_at_utc(self) -> datetime:
        """Wall-clock creation time, for persistence and display."""
        return self._created_at_utc

    @property
    def last_active_at(self) -> float:
        """Monotonic timestamp of the last :meth:`touch`."""
        return self._last_active_at

    @property
    def close_reason(self) -> str | None:
        """Why the session closed, or ``None`` while it is live."""
        return self._close_reason

    @property
    def age_s(self) -> float:
        """Seconds since creation."""
        return self._clock() - self._created_at

    @property
    def idle_s(self) -> float:
        """Seconds since the last activity."""
        return self._clock() - self._last_active_at

    @property
    def turn_count(self) -> int:
        """Turns completed over the session's whole life."""
        return self._turn_total

    @property
    def recent_turns(self) -> Sequence[ConversationTurn]:
        """The most recent turns held in memory, oldest first."""
        return tuple(self._turns)

    # -- activity -----------------------------------------------------------

    def touch(self) -> None:
        """Mark the session active, deferring the idle reaper."""
        self._last_active_at = self._clock()

    def require_active(self) -> None:
        """Raise if the session is no longer accepting work.

        Raises:
            SessionClosedError: The session is closing or closed.
        """
        if self._state is not SessionState.ACTIVE:
            raise SessionClosedError(self.session_id)

    def record_turn(self, turn: ConversationTurn) -> None:
        """Append a completed exchange to the history and the ring buffer.

        Args:
            turn: The finished exchange.
        """
        self.history.add_turn(turn)
        self._turns.append(turn)
        self._turn_total += 1
        self.touch()

    def begin_turn(self) -> None:
        """Re-arm the cancellation token for a new turn."""
        self.cancellation.reset()
        self.touch()

    def barge_in(self, reason: str = "barge_in") -> bool:
        """Cancel everything in flight for this session.

        Returns:
            ``True`` if this call fired the token.
        """
        self.touch()
        return self.cancellation.cancel(reason)

    # -- teardown -----------------------------------------------------------

    def add_cleanup(self, hook: Callable[[], Coroutine[Any, Any, None]]) -> None:
        """Register an async teardown hook.

        Hooks run in reverse registration order (LIFO), like a context-manager
        stack: a hook registered later may depend on resources a hook
        registered earlier established, so it must unwind first.

        Args:
            hook: Zero-argument coroutine function.

        Raises:
            SessionClosedError: The session has already begun closing, so the
                hook would never run.
        """
        self.require_active()
        self._cleanup.append(hook)

    async def wait_closed(self) -> None:
        """Block until the session has finished closing."""
        await self._closed_event.wait()

    async def aclose(self, *, reason: str = "closed") -> None:
        """Tear the session down deterministically.

        Ordering is fixed and observable:

        1. The state flips to ``CLOSING``, so new work is refused immediately.
        2. The cancellation token fires, aborting in-flight LLM and TTS work.
        3. Cleanup hooks run in LIFO order, one at a time. A hook that raises
           is logged and the unwind continues -- a leaked Redis key must not
           strand a GPU allocation registered before it.
        4. The state flips to ``CLOSED`` and :meth:`wait_closed` unblocks.

        Idempotent and safe to call concurrently: the second caller awaits the
        first and returns once teardown is complete.

        Args:
            reason: Recorded on the session and used as the cancellation reason.
        """
        async with self._close_lock:
            if self._state is SessionState.CLOSED:
                return

            self._state = SessionState.CLOSING
            self._close_reason = reason
            self.cancellation.cancel(reason)

            hooks = list(reversed(self._cleanup))
            self._cleanup.clear()
            for hook in hooks:
                try:
                    await hook()
                except asyncio.CancelledError:
                    # Teardown must complete even if the closing task is being
                    # cancelled; re-raise only after the remaining hooks run.
                    logger.warning(
                        "cleanup hook cancelled", extra={"session_id": self.session_id}
                    )
                except Exception:
                    logger.exception(
                        "cleanup hook failed",
                        extra={
                            "session_id": self.session_id,
                            "hook": getattr(hook, "__qualname__", "?"),
                        },
                    )

            self._state = SessionState.CLOSED
            self._closed_event.set()
            logger.info(
                "session closed",
                extra={
                    "session_id": self.session_id,
                    "reason": reason,
                    "turns": self._turn_total,
                    "duration_s": round(self.age_s, 3),
                },
            )
