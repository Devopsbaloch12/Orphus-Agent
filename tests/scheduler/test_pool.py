from __future__ import annotations

from orphus.domain.types import SessionId
from orphus.scheduler import WorkerPool


async def test_pool_executes_work() -> None:
    pool = WorkerPool(workers=2, capacity=4)
    await pool.start()

    async def operation() -> int:
        return 42

    future = pool.submit(SessionId("one"), operation)
    assert await future == 42
    await pool.aclose()


async def test_pool_propagates_failure() -> None:
    pool = WorkerPool(workers=1, capacity=2)
    await pool.start()

    async def operation() -> None:
        raise ValueError("bad work")

    future = pool.submit(SessionId("one"), operation)
    try:
        await future
    except ValueError as exc:
        assert str(exc) == "bad work"
    else:
        raise AssertionError("failure was not propagated")
    await pool.aclose()

