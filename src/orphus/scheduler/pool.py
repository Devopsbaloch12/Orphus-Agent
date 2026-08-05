"""Bounded priority worker pool with cancellation and graceful draining."""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from orphus.domain.types import SessionId
from orphus.queue import OverflowPolicy, PriorityBoundedQueue, QueueClosed


@dataclass(slots=True)
class WorkItem[T]:
    priority: int
    sequence: int
    session_id: SessionId
    operation: Callable[[], Awaitable[T]] = field(repr=False)
    future: asyncio.Future[T] = field(repr=False)


class WorkerPool:
    """Execute bounded prioritized work without admitting an unbounded backlog."""

    def __init__(self, *, workers: int = 4, capacity: int = 64) -> None:
        if workers < 1:
            raise ValueError("workers must be >= 1")
        self._worker_count = workers
        self._queue: PriorityBoundedQueue[WorkItem[Any]] = PriorityBoundedQueue(
            capacity,
            priority_of=lambda item: item.priority,
            policy=OverflowPolicy.REJECT,
            name="worker-pool",
        )
        self._sequence = itertools.count()
        self._workers: list[asyncio.Task[None]] = []
        self._running: dict[SessionId, set[asyncio.Future[Any]]] = {}

    @property
    def queue_depth(self) -> int:
        return self._queue.depth

    async def start(self) -> None:
        if self._workers:
            return
        self._workers = [
            asyncio.create_task(self._worker(), name=f"orphus-worker-{index}")
            for index in range(self._worker_count)
        ]

    def submit[T](
        self,
        session_id: SessionId,
        operation: Callable[[], Awaitable[T]],
        *,
        priority: int = 100,
    ) -> asyncio.Future[T]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[T] = loop.create_future()
        item = WorkItem(priority, next(self._sequence), session_id, operation, future)
        self._queue.put(item)
        return future

    def cancel_session(self, session_id: SessionId) -> int:
        removed = self._queue.remove_if(lambda item: item.session_id == session_id)
        for item in removed:
            item.future.cancel()
        running = self._running.get(session_id, set())
        for task in tuple(running):
            task.cancel()
        return len(removed) + len(running)

    async def _worker(self) -> None:
        while True:
            try:
                item = await self._queue.get()
            except QueueClosed:
                return
            if item.future.cancelled():
                continue
            task: asyncio.Future[Any] = asyncio.ensure_future(item.operation())
            self._running.setdefault(item.session_id, set()).add(task)
            try:
                result = await task
            except asyncio.CancelledError:
                item.future.cancel()
            except Exception as exc:
                item.future.set_exception(exc)
            else:
                item.future.set_result(result)
            finally:
                tasks = self._running.get(item.session_id)
                if tasks is not None:
                    tasks.discard(task)
                    if not tasks:
                        self._running.pop(item.session_id, None)

    async def aclose(self, *, drain_timeout_s: float = 15.0) -> None:
        self._queue.close()
        if not self._workers:
            return
        try:
            async with asyncio.timeout(drain_timeout_s):
                await asyncio.gather(*self._workers)
        except TimeoutError:
            for worker in self._workers:
                worker.cancel()
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
