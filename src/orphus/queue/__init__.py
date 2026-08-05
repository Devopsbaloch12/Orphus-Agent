"""Bounded async queues with explicit overflow policies.

Every hand-off point in the Orphus pipeline goes through one of these. Nothing
in the system uses an unbounded queue: under overload an unbounded queue does
not fail, it degrades silently -- latency climbs turn after turn until the
process is OOM-killed. A bounded queue forces the decision at the moment of
overload and records it in a counter the dashboards can see.

Two shapes are provided:

* :class:`BoundedQueue` -- FIFO, defaults to ``drop_oldest``, for media frames.
* :class:`PriorityBoundedQueue` -- min-heap, defaults to ``reject``, for work
  items where losing a request silently would be a correctness bug.

Both are event-loop objects and are not thread-safe; see
``orphus.queue._base`` for the reasoning.
"""

from __future__ import annotations

from orphus.queue._base import (
    OverflowPolicy,
    PutOutcome,
    QueueClosed,
    QueueFull,
    QueueStats,
)
from orphus.queue.bounded import BoundedQueue
from orphus.queue.priority import PriorityBoundedQueue

__all__ = [
    "BoundedQueue",
    "OverflowPolicy",
    "PriorityBoundedQueue",
    "PutOutcome",
    "QueueClosed",
    "QueueFull",
    "QueueStats",
]
