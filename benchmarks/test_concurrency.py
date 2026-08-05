"""CPU-safe scheduler concurrency benchmark; GPU benchmarks run on RunPod."""

from __future__ import annotations

import asyncio

import pytest

from orphus.domain.types import SessionId
from orphus.scheduler import WorkerPool


@pytest.mark.slow
async def test_twenty_sessions_complete_without_starvation() -> None:
    pool = WorkerPool(workers=4, capacity=20)
    await pool.start()
    started: set[str] = set()

    def operation(session_id: str):
        async def run() -> str:
            started.add(session_id)
            await asyncio.sleep(0.005)
            return session_id

        return run

    futures = [
        pool.submit(SessionId(f"session-{index}"), operation(f"session-{index}"))
        for index in range(20)
    ]
    results = await asyncio.gather(*futures)
    assert len(results) == 20
    assert len(started) == 20
    await pool.aclose()
