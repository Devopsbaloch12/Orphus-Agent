"""Behaviour of the FIFO bounded queue, focused on the overflow policies."""

from __future__ import annotations

import asyncio

import pytest

from orphus.queue import (
    BoundedQueue,
    OverflowPolicy,
    PutOutcome,
    QueueClosed,
    QueueFull,
)


def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError, match="capacity"):
        BoundedQueue[int](0)


async def test_fifo_order() -> None:
    q: BoundedQueue[int] = BoundedQueue(8)
    for value in range(5):
        assert q.put(value) is PutOutcome.ACCEPTED
    assert [await q.get() for _ in range(5)] == [0, 1, 2, 3, 4]
    assert q.depth == 0


async def test_drop_oldest_keeps_the_freshest_frames() -> None:
    q: BoundedQueue[int] = BoundedQueue(3, policy=OverflowPolicy.DROP_OLDEST, name="audio")
    for value in range(6):
        q.put(value)

    # The three most recent survive; the three stalest were shed.
    assert q.drain() == [3, 4, 5]
    assert q.dropped == 3
    assert q.rejected == 0


async def test_drop_oldest_reports_displacement() -> None:
    q: BoundedQueue[int] = BoundedQueue(1, policy=OverflowPolicy.DROP_OLDEST)
    assert q.put(1) is PutOutcome.ACCEPTED
    assert q.put(2) is PutOutcome.DISPLACED_OLDEST
    assert await q.get() == 2


async def test_drop_newest_preserves_the_backlog() -> None:
    q: BoundedQueue[int] = BoundedQueue(3, policy=OverflowPolicy.DROP_NEWEST)
    for value in range(6):
        q.put(value)

    assert q.drain() == [0, 1, 2]
    assert q.dropped == 3


def test_reject_raises_and_counts() -> None:
    q: BoundedQueue[int] = BoundedQueue(2, policy=OverflowPolicy.REJECT, name="llm")
    q.put(1)
    q.put(2)
    with pytest.raises(QueueFull, match="llm"):
        q.put(3)
    assert q.rejected == 1
    assert q.dropped == 0
    assert q.depth == 2


async def test_get_blocks_until_an_item_arrives() -> None:
    q: BoundedQueue[str] = BoundedQueue(4)
    getter = asyncio.ensure_future(q.get())
    await asyncio.sleep(0)
    assert not getter.done()

    q.put("hello")
    assert await getter == "hello"


async def test_cancelled_getter_hands_its_wakeup_on() -> None:
    """A cancelled consumer must not swallow the item queued for it."""
    q: BoundedQueue[int] = BoundedQueue(4)
    first = asyncio.ensure_future(q.get())
    second = asyncio.ensure_future(q.get())
    await asyncio.sleep(0)

    # Wake `first` with an item, then cancel it in the same tick, before it has
    # resumed. The item must reach `second` instead of being lost.
    q.put(7)
    first.cancel()
    assert await second == 7
    with pytest.raises(asyncio.CancelledError):
        await first


async def test_close_drains_then_raises() -> None:
    q: BoundedQueue[int] = BoundedQueue(4)
    q.put(1)
    q.put(2)
    q.close()

    with pytest.raises(QueueClosed):
        q.put(3)

    assert await q.get() == 1
    assert await q.get() == 2
    with pytest.raises(QueueClosed):
        await q.get()


async def test_close_wakes_parked_consumers() -> None:
    q: BoundedQueue[int] = BoundedQueue(4)
    getter = asyncio.ensure_future(q.get())
    await asyncio.sleep(0)

    q.close()
    with pytest.raises(QueueClosed):
        await asyncio.wait_for(getter, timeout=1.0)


async def test_async_iteration_stops_at_close() -> None:
    q: BoundedQueue[int] = BoundedQueue(8)
    for value in range(3):
        q.put(value)
    q.close()

    assert [value async for value in q] == [0, 1, 2]


async def test_iteration_is_live_until_closed() -> None:
    q: BoundedQueue[int] = BoundedQueue(8)
    seen: list[int] = []

    async def consume() -> None:
        async for value in q:
            seen.append(value)

    consumer = asyncio.ensure_future(consume())
    for value in range(3):
        q.put(value)
        await asyncio.sleep(0)
    q.close()
    await asyncio.wait_for(consumer, timeout=1.0)

    assert seen == [0, 1, 2]


def test_get_nowait_semantics() -> None:
    q: BoundedQueue[int] = BoundedQueue(4)
    with pytest.raises(LookupError):
        q.get_nowait()
    q.put(1)
    assert q.get_nowait() == 1
    q.close()
    with pytest.raises(QueueClosed):
        q.get_nowait()


def test_clear_counts_as_drops() -> None:
    """Barge-in flushes the queue; those frames are genuinely lost."""
    q: BoundedQueue[int] = BoundedQueue(8)
    for value in range(4):
        q.put(value)

    assert q.clear() == 4
    assert q.depth == 0
    assert q.dropped == 4


def test_stats_snapshot() -> None:
    q: BoundedQueue[int] = BoundedQueue(4, name="asr")
    for value in range(3):
        q.put(value)
    q.get_nowait()

    stats = q.stats()
    assert stats.name == "asr"
    assert stats.capacity == 4
    assert stats.depth == 2
    assert stats.high_water_mark == 3
    assert stats.enqueued == 3
    assert stats.dequeued == 1
    assert stats.utilisation == pytest.approx(0.5)
    assert not stats.closed


def test_peek_does_not_consume() -> None:
    q: BoundedQueue[int] = BoundedQueue(4)
    assert q.peek() is None
    q.put(9)
    assert q.peek() == 9
    assert q.depth == 1


def test_close_is_idempotent() -> None:
    q: BoundedQueue[int] = BoundedQueue(2)
    q.close()
    q.close()
    assert q.is_closed
