"""Shared machinery for the bounded queues.

Both :class:`~orphus.queue.bounded.BoundedQueue` and
:class:`~orphus.queue.priority.PriorityBoundedQueue` need the same consumer-side
behaviour -- awaiting getters, wakeup handoff on cancellation, close/drain
semantics, and Prometheus-facing counters. Only the storage discipline (FIFO vs
heap) and the overflow decision differ, so that is all the subclasses supply.

The queues are **not** thread-safe. They are event-loop objects: every mutation
happens in loop callbacks, so no lock is required and none is taken. A producer
running on another thread must marshal through ``loop.call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from orphus.observability.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = get_logger(__name__)

__all__ = [
    "OverflowPolicy",
    "PutOutcome",
    "QueueClosed",
    "QueueFull",
    "QueueStats",
]

# Drops are counted, not logged, on the hot path: at a 20ms frame cadence a
# log line per dropped frame would itself become the bottleneck. One warning is
# emitted when a queue first overflows, then one every _DROP_LOG_INTERVAL drops.
_DROP_LOG_INTERVAL = 100


class OverflowPolicy(StrEnum):
    """What a full queue does with an incoming item.

    Values match ``scheduler.overflow_policy`` in the YAML config.
    """

    DROP_OLDEST = "drop_oldest"
    """Evict the stalest item to make room. The realtime-audio default: a
    dropped frame is a click, a growing backlog is a broken conversation."""

    DROP_NEWEST = "drop_newest"
    """Refuse the incoming item and keep the backlog intact. Correct when
    earlier items are prerequisites for later ones."""

    REJECT = "reject"
    """Raise :class:`QueueFull` so the caller decides. Correct for work queues
    where silently losing a request would be a correctness bug."""


class PutOutcome(StrEnum):
    """What a queue did with an item handed to ``put``."""

    ACCEPTED = "accepted"
    """Stored with room to spare."""

    DISPLACED_OLDEST = "displaced_oldest"
    """Stored, but an older item was evicted to make room."""

    DROPPED_NEWEST = "dropped_newest"
    """Not stored; the queue was full and the policy favours the backlog."""

    @property
    def stored(self) -> bool:
        """Whether the item is now in the queue."""
        return self is not PutOutcome.DROPPED_NEWEST


class QueueClosed(RuntimeError):
    """Raised when putting into, or getting from a drained, closed queue."""


class QueueFull(RuntimeError):
    """Raised by ``put`` when the queue is full under ``REJECT``."""


@dataclass(frozen=True, slots=True)
class QueueStats:
    """Point-in-time queue counters, shaped for a Prometheus collector.

    The metrics registry lives elsewhere; this is deliberately plain data so
    that scraping it never reaches into queue internals.
    """

    name: str
    capacity: int
    depth: int
    high_water_mark: int
    enqueued: int
    dequeued: int
    dropped: int
    rejected: int
    closed: bool

    @property
    def utilisation(self) -> float:
        """Depth as a fraction of capacity, in ``[0.0, 1.0]``."""
        return self.depth / self.capacity if self.capacity else 0.0


def _wake_next(waiters: deque[asyncio.Future[None]]) -> None:
    """Resolve the first still-pending waiter, discarding dead ones."""
    while waiters:
        waiter = waiters.popleft()
        if not waiter.done():
            waiter.set_result(None)
            return


class _QueueBase[T](ABC):
    """Consumer side, counters, and lifecycle shared by the bounded queues."""

    __slots__ = (
        "_capacity",
        "_closed",
        "_dequeued",
        "_dropped",
        "_enqueued",
        "_getters",
        "_high_water_mark",
        "_name",
        "_policy",
        "_rejected",
    )

    def __init__(self, capacity: int, *, policy: OverflowPolicy, name: str) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._policy = policy
        self._name = name
        self._getters: deque[asyncio.Future[None]] = deque()
        self._closed = False
        self._enqueued = 0
        self._dequeued = 0
        self._dropped = 0
        self._rejected = 0
        self._high_water_mark = 0

    # -- storage hooks implemented by subclasses ---------------------------

    @abstractmethod
    def _size(self) -> int:
        """Number of items currently stored."""

    @abstractmethod
    def _take(self) -> T:
        """Remove and return the next item. Only called when non-empty."""

    @abstractmethod
    def _discard_oldest(self) -> None:
        """Evict the stalest item. Only called when full."""

    @abstractmethod
    def _store(self, item: T) -> None:
        """Add an item. Only called when there is room."""

    @abstractmethod
    def _clear_storage(self) -> None:
        """Drop every stored item."""

    # -- introspection ------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable queue name, used in logs and metric labels."""
        return self._name

    @property
    def capacity(self) -> int:
        """Maximum number of items the queue will hold."""
        return self._capacity

    @property
    def policy(self) -> OverflowPolicy:
        """Overflow policy in force."""
        return self._policy

    @property
    def depth(self) -> int:
        """Number of items currently queued."""
        return self._size()

    @property
    def dropped(self) -> int:
        """Items lost to overflow, under either drop policy."""
        return self._dropped

    @property
    def rejected(self) -> int:
        """``put`` calls refused with :class:`QueueFull`."""
        return self._rejected

    @property
    def is_closed(self) -> bool:
        """Whether the queue has been closed to new items."""
        return self._closed

    @property
    def is_full(self) -> bool:
        """Whether the next ``put`` would trigger the overflow policy."""
        return self._size() >= self._capacity

    def empty(self) -> bool:
        """Whether the queue currently holds no items."""
        return self._size() == 0

    def stats(self) -> QueueStats:
        """Snapshot every counter in one allocation."""
        return QueueStats(
            name=self._name,
            capacity=self._capacity,
            depth=self._size(),
            high_water_mark=self._high_water_mark,
            enqueued=self._enqueued,
            dequeued=self._dequeued,
            dropped=self._dropped,
            rejected=self._rejected,
            closed=self._closed,
        )

    # -- producer side ------------------------------------------------------

    def put(self, item: T) -> PutOutcome:
        """Offer an item, resolving overflow immediately.

        This never blocks and never awaits, so a capture callback can call it
        without an event loop turn and without ever stalling on a slow consumer.

        Args:
            item: The value to enqueue.

        Returns:
            How the queue handled the item.

        Raises:
            QueueClosed: The queue is closed to new items.
            QueueFull: The queue is full and the policy is
                :attr:`OverflowPolicy.REJECT`.
        """
        if self._closed:
            raise QueueClosed(f"queue {self._name!r} is closed")

        if self._size() >= self._capacity:
            outcome = self._overflow(item)
        else:
            self._store(item)
            outcome = PutOutcome.ACCEPTED

        if outcome.stored:
            self._enqueued += 1
            self._high_water_mark = max(self._high_water_mark, self._size())
            _wake_next(self._getters)
        return outcome

    def _overflow(self, item: T) -> PutOutcome:
        """Apply :attr:`policy` to an item that does not fit."""
        match self._policy:
            case OverflowPolicy.REJECT:
                self._rejected += 1
                raise QueueFull(
                    f"queue {self._name!r} is full ({self._capacity} items)"
                )
            case OverflowPolicy.DROP_NEWEST:
                self._record_drop()
                return PutOutcome.DROPPED_NEWEST
            case OverflowPolicy.DROP_OLDEST:
                self._discard_oldest()
                self._record_drop()
                self._store(item)
                return PutOutcome.DISPLACED_OLDEST

    def _record_drop(self) -> None:
        self._dropped += 1
        if self._dropped == 1 or self._dropped % _DROP_LOG_INTERVAL == 0:
            logger.warning(
                "queue overflow",
                extra={
                    "queue": self._name,
                    "policy": self._policy.value,
                    "capacity": self._capacity,
                    "dropped_total": self._dropped,
                },
            )

    # -- consumer side ------------------------------------------------------

    async def get(self) -> T:
        """Await the next item.

        Returns:
            The next item, by FIFO or priority order depending on the subclass.

        Raises:
            QueueClosed: The queue was closed and is now drained.
        """
        while self._size() == 0:
            if self._closed:
                raise QueueClosed(f"queue {self._name!r} is closed and drained")
            await self._wait_for_item()
        return self._pop()

    def get_nowait(self) -> T:
        """Take the next item without awaiting.

        Returns:
            The next item.

        Raises:
            QueueClosed: The queue was closed and is now drained.
            LookupError: The queue is momentarily empty but still open.
        """
        if self._size() == 0:
            if self._closed:
                raise QueueClosed(f"queue {self._name!r} is closed and drained")
            raise LookupError(f"queue {self._name!r} is empty")
        return self._pop()

    async def _wait_for_item(self) -> None:
        """Park until a producer or ``close`` wakes us.

        On cancellation the wakeup we may already have been handed has to be
        passed on, otherwise a concurrent consumer sleeps through an item that
        was queued for us. This mirrors ``asyncio.Queue``'s handoff protocol.
        """
        waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._getters.append(waiter)
        try:
            await waiter
        except BaseException:
            waiter.cancel()
            with suppress(ValueError):
                self._getters.remove(waiter)
            if self._size() > 0 and not waiter.cancelled():
                _wake_next(self._getters)
            raise

    def _pop(self) -> T:
        item = self._take()
        self._dequeued += 1
        return item

    def __aiter__(self) -> AsyncIterator[T]:
        """Iterate items until the queue is closed and drained."""
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[T]:
        while True:
            try:
                yield await self.get()
            except QueueClosed:
                return

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Stop accepting items. Queued items stay drainable.

        Idempotent. Every parked consumer is woken so it can drain what is left
        and then observe :class:`QueueClosed`.
        """
        if self._closed:
            return
        self._closed = True
        while self._getters:
            waiter = self._getters.popleft()
            if not waiter.done():
                waiter.set_result(None)

    def clear(self) -> int:
        """Discard every queued item, e.g. on barge-in.

        Returns:
            How many items were discarded. They count as drops.
        """
        discarded = self._size()
        if discarded:
            self._clear_storage()
            self._dropped += discarded
        return discarded

    def drain(self) -> list[T]:
        """Remove and return every queued item in order."""
        items: list[T] = []
        while self._size() > 0:
            items.append(self._pop())
        return items
