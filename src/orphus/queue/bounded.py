"""FIFO bounded queue for the streaming pipeline stages.

One of these sits between every pair of pipeline stages (capture -> VAD,
VAD -> ASR, ASR -> LLM, LLM -> TTS). Bounding them is what turns overload into
a measurable drop counter instead of unbounded memory growth and a latency
curve that never recovers.
"""

from __future__ import annotations

from collections import deque

from orphus.queue._base import OverflowPolicy, _QueueBase

__all__ = ["BoundedQueue"]


class BoundedQueue[T](_QueueBase[T]):
    """A bounded FIFO queue with a configurable overflow policy.

    Args:
        capacity: Maximum items held before the overflow policy applies.
        policy: What to do when the queue is full. Defaults to
            :attr:`~orphus.queue.OverflowPolicy.DROP_OLDEST`, which is right for
            realtime audio: a 20ms frame is worth far less than the latency a
            growing backlog would add.
        name: Label used in logs and metric labels.

    Example:
        >>> q: BoundedQueue[int] = BoundedQueue(2, name="audio")
        >>> _ = q.put(1), q.put(2), q.put(3)
        >>> q.depth, q.dropped
        (2, 1)
    """

    __slots__ = ("_items",)

    def __init__(
        self,
        capacity: int,
        *,
        policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
        name: str = "queue",
    ) -> None:
        super().__init__(capacity, policy=policy, name=name)
        self._items: deque[T] = deque()

    def _size(self) -> int:
        return len(self._items)

    def _take(self) -> T:
        return self._items.popleft()

    def _discard_oldest(self) -> None:
        self._items.popleft()

    def _store(self, item: T) -> None:
        self._items.append(item)

    def _clear_storage(self) -> None:
        self._items.clear()

    def peek(self) -> T | None:
        """Look at the head without removing it. ``None`` when empty."""
        return self._items[0] if self._items else None
