"""Cooperative cancellation, primarily for barge-in.

When the VAD reports ``SPEECH_STARTED`` mid-reply the system must stop three
things at once: LLM generation (to stop burning provider tokens), TTS synthesis
(to stop producing audio nobody will hear), and playback. They live in
different tasks, so a shared token is the coordination point.

The token is one-shot per turn and explicitly re-armed with :meth:`reset`. A
token that auto-rearmed would race: a late cancel from the previous turn would
silently kill the next one. :attr:`generation` exists so a consumer can tell
whether the fire it observed belongs to the turn it is serving.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from orphus.conversation.errors import CancellationError
from orphus.observability.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = get_logger(__name__)

__all__ = ["CancellationToken"]


class CancellationToken:
    """A re-armable, one-shot cancellation signal shared across tasks.

    Args:
        name: Label used in logs; usually the session id.

    Example:
        >>> token = CancellationToken(name="sess_1")
        >>> token.cancel("barge_in")
        True
        >>> token.is_cancelled, token.reason
        (True, 'barge_in')
    """

    __slots__ = ("_bound", "_callbacks", "_event", "_generation", "_name", "_reason")

    def __init__(self, *, name: str = "") -> None:
        self._name = name
        self._event = asyncio.Event()
        self._reason: str | None = None
        self._generation = 0
        self._callbacks: list[Callable[[str], None]] = []
        self._bound: set[asyncio.Task[Any]] = set()

    # -- state --------------------------------------------------------------

    @property
    def name(self) -> str:
        """Label supplied at construction."""
        return self._name

    @property
    def is_cancelled(self) -> bool:
        """Whether the token has fired and not yet been reset."""
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        """Why the token fired, or ``None`` if it has not."""
        return self._reason

    @property
    def generation(self) -> int:
        """How many times the token has been reset.

        A consumer that captured the generation when it started work can
        compare it later to distinguish "my turn was cancelled" from "a
        previous turn was cancelled and the token has since been re-armed".
        """
        return self._generation

    # -- firing -------------------------------------------------------------

    def cancel(self, reason: str = "cancelled") -> bool:
        """Fire the token.

        Idempotent: a second call while already cancelled is a no-op, so the
        VAD emitting two speech-start events does not double-cancel.

        Args:
            reason: Short machine-readable cause, e.g. ``barge_in``.

        Returns:
            ``True`` if this call fired the token, ``False`` if it was already
            cancelled.
        """
        if self._event.is_set():
            return False
        self._reason = reason
        self._event.set()

        # Bound tasks are cancelled before user callbacks run: the point of a
        # barge-in is to stop work immediately, and a slow callback must not
        # delay that.
        for task in tuple(self._bound):
            if not task.done():
                task.cancel()

        for callback in tuple(self._callbacks):
            try:
                callback(reason)
            except Exception:
                logger.exception("cancellation callback failed", extra={"token": self._name})
        return True

    def reset(self) -> None:
        """Re-arm the token for the next turn.

        Bound tasks are forgotten (they belonged to the finished turn);
        callbacks are kept, because they are registered for the token's
        lifetime rather than per turn.
        """
        self._event.clear()
        self._reason = None
        self._generation += 1
        self._bound.clear()

    # -- observing ----------------------------------------------------------

    async def wait(self) -> str:
        """Block until the token fires.

        Returns:
            The cancellation reason.
        """
        await self._event.wait()
        return self._reason or "cancelled"

    def raise_if_cancelled(self) -> None:
        """Raise :class:`CancellationError` if the token has fired.

        Raises:
            CancellationError: The token is cancelled.
        """
        if self._event.is_set():
            raise CancellationError(self._reason or "cancelled")

    def on_cancel(self, callback: Callable[[str], None]) -> Callable[[], None]:
        """Register a synchronous listener, fired once when the token fires.

        The callback must not block; it runs inside :meth:`cancel`, on the
        barge-in path. Use :meth:`bind` for anything that needs to await.

        Args:
            callback: Receives the cancellation reason.

        Returns:
            An unsubscribe function.
        """
        self._callbacks.append(callback)

        def unsubscribe() -> None:
            with suppress(ValueError):
                self._callbacks.remove(callback)

        return unsubscribe

    def bind(self, task: asyncio.Task[Any]) -> None:
        """Cancel ``task`` when this token fires.

        If the token has *already* fired, ``task`` is cancelled immediately --
        otherwise a task started in the same tick as a barge-in would run to
        completion.

        Args:
            task: The task whose lifetime is tied to this token.
        """
        if self._event.is_set():
            task.cancel()
            return
        self._bound.add(task)
        task.add_done_callback(self._bound.discard)

    async def guard[T](self, awaitable: Awaitable[T]) -> T:
        """Run ``awaitable``, aborting it the instant the token fires.

        Args:
            awaitable: The work to protect.

        Returns:
            Whatever ``awaitable`` returns.

        Raises:
            CancellationError: The token fired before the work completed. The
                underlying task is cancelled and awaited before this is raised,
                so no orphaned task outlives the call.
        """
        self.raise_if_cancelled()

        task = asyncio.ensure_future(awaitable)
        self.bind(task)
        try:
            return await task
        except asyncio.CancelledError:
            # Distinguish "the token stopped us" from "our caller was torn
            # down". Only the former is a recoverable conversational event.
            if self._event.is_set():
                raise CancellationError(self._reason or "cancelled") from None
            raise
