"""Bounded priority queue used by the scheduler's worker pool.

Priority alone is not enough for a voice pipeline: two barge-in cancellations
submitted a second apart must still run in submission order, or the second can
undo the first. Entries are ordered by ``(priority, sequence)``, which makes the
ordering total and the queue stable within a priority band.
"""

from __future__ import annotations

import heapq
import itertools
from typing import TYPE_CHECKING

from orphus.queue._base import OverflowPolicy, _QueueBase

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

__all__ = ["PriorityBoundedQueue"]


class PriorityBoundedQueue[T](_QueueBase[T]):
    """A bounded min-heap queue; the lowest ``priority`` value is served first.

    Args:
        capacity: Maximum items held before the overflow policy applies.
        priority_of: Extracts the ordering key from an item. Passing the
            extractor in (rather than a ``(priority, item)`` tuple API) keeps
            ``put`` identical to :class:`~orphus.queue.BoundedQueue`'s, so the
            two are interchangeable behind ``_QueueBase``.
        policy: What to do when the queue is full. Defaults to
            :attr:`~orphus.queue.OverflowPolicy.REJECT` -- for a *work* queue,
            silently dropping a request is a correctness bug, so overload is
            surfaced to admission control rather than swallowed.
        name: Label used in logs and metric labels.

    Note:
        ``DROP_OLDEST`` evicts the earliest-submitted entry regardless of its
        priority, matching the policy's name. Eviction is ``O(n)``: a heap has
        no cheap handle on its oldest member, and at the capacities used here
        (tens of entries) a scan beats maintaining a second index.
    """

    __slots__ = ("_counter", "_heap", "_priority_of")

    def __init__(
        self,
        capacity: int,
        priority_of: Callable[[T], int],
        *,
        policy: OverflowPolicy = OverflowPolicy.REJECT,
        name: str = "priority-queue",
    ) -> None:
        super().__init__(capacity, policy=policy, name=name)
        # (priority, sequence, item). The monotonic sequence breaks priority
        # ties in FIFO order and stops heapq ever comparing the payloads, which
        # therefore need not be orderable at all.
        self._heap: list[tuple[int, int, T]] = []
        self._counter = itertools.count()
        self._priority_of = priority_of

    def _size(self) -> int:
        return len(self._heap)

    def _take(self) -> T:
        return heapq.heappop(self._heap)[2]

    def _discard_oldest(self) -> None:
        oldest = min(range(len(self._heap)), key=lambda index: self._heap[index][1])
        self._heap[oldest] = self._heap[-1]
        self._heap.pop()
        heapq.heapify(self._heap)

    def _store(self, item: T) -> None:
        heapq.heappush(self._heap, (self._priority_of(item), next(self._counter), item))

    def _clear_storage(self) -> None:
        self._heap.clear()

    def peek_priority(self) -> int | None:
        """Priority of the next item to be served, or ``None`` when empty."""
        return self._heap[0][0] if self._heap else None

    def snapshot(self) -> Iterator[T]:
        """Yield every queued item in unspecified order.

        For cancellation sweeps and diagnostics, not for consumption.
        """
        return (item for _priority, _sequence, item in self._heap)

    def remove_if(self, predicate: Callable[[T], bool]) -> list[T]:
        """Remove every queued item matching ``predicate``.

        Used to drop all of a session's pending work in one pass when the
        session ends, so workers never pick up orphaned tasks.

        Args:
            predicate: Returns ``True`` for items to remove.

        Returns:
            The removed items. They are *not* counted as drops: they were
            cancelled, not shed under load.
        """
        kept: list[tuple[int, int, T]] = []
        removed: list[T] = []
        for entry in self._heap:
            if predicate(entry[2]):
                removed.append(entry[2])
            else:
                kept.append(entry)
        if removed:
            self._heap = kept
            heapq.heapify(self._heap)
        return removed
