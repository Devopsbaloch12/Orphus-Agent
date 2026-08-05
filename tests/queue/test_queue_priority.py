"""Behaviour of the bounded priority queue used by the scheduler."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from orphus.queue import OverflowPolicy, PriorityBoundedQueue, QueueFull


@dataclass(frozen=True, slots=True)
class Job:
    """Deliberately unorderable, to prove the heap never compares payloads."""

    name: str
    priority: int
    session: str = "s1"


def _queue(
    capacity: int = 8,
    policy: OverflowPolicy = OverflowPolicy.REJECT,
) -> PriorityBoundedQueue[Job]:
    return PriorityBoundedQueue(capacity, lambda job: job.priority, policy=policy, name="work")


async def test_lowest_priority_value_is_served_first() -> None:
    q = _queue()
    q.put(Job("normal", 20))
    q.put(Job("realtime", 0))
    q.put(Job("low", 30))
    q.put(Job("high", 10))

    order = [(await q.get()).name for _ in range(4)]
    assert order == ["realtime", "high", "normal", "low"]


async def test_ties_are_broken_in_submission_order() -> None:
    """Two barge-ins at the same priority must run in the order submitted."""
    q = _queue()
    for index in range(5):
        q.put(Job(f"job-{index}", 0))

    order = [(await q.get()).name for _ in range(5)]
    assert order == [f"job-{index}" for index in range(5)]


def test_reject_is_the_default_policy() -> None:
    q = _queue(capacity=2)
    q.put(Job("a", 0))
    q.put(Job("b", 0))
    with pytest.raises(QueueFull):
        q.put(Job("c", 0))
    assert q.rejected == 1


async def test_drop_oldest_evicts_by_submission_not_priority() -> None:
    q = _queue(capacity=3, policy=OverflowPolicy.DROP_OLDEST)
    q.put(Job("first", 30))
    q.put(Job("second", 0))
    q.put(Job("third", 10))
    q.put(Job("fourth", 20))  # displaces "first", the earliest submission

    names = {job.name for job in q.snapshot()}
    assert names == {"second", "third", "fourth"}
    assert q.dropped == 1
    # Heap invariant survives the O(n) eviction.
    assert (await q.get()).name == "second"


def test_drop_newest_refuses_the_incoming_item() -> None:
    q = _queue(capacity=2, policy=OverflowPolicy.DROP_NEWEST)
    q.put(Job("a", 0))
    q.put(Job("b", 0))
    q.put(Job("c", 0))

    assert {job.name for job in q.snapshot()} == {"a", "b"}
    assert q.dropped == 1


def test_peek_priority() -> None:
    q = _queue()
    assert q.peek_priority() is None
    q.put(Job("a", 20))
    q.put(Job("b", 5))
    assert q.peek_priority() == 5


async def test_remove_if_cancels_a_sessions_work() -> None:
    q = _queue()
    q.put(Job("a", 0, session="doomed"))
    q.put(Job("b", 10, session="alive"))
    q.put(Job("c", 20, session="doomed"))

    removed = q.remove_if(lambda job: job.session == "doomed")

    assert {job.name for job in removed} == {"a", "c"}
    assert q.depth == 1
    # Cancellation is not load shedding, so it must not pollute the drop counter.
    assert q.dropped == 0
    assert (await q.get()).name == "b"


def test_remove_if_no_match_is_a_noop() -> None:
    q = _queue()
    q.put(Job("a", 0))
    assert q.remove_if(lambda job: False) == []
    assert q.depth == 1
