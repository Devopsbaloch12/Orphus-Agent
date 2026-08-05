"""Sliding-window rate limiting for public endpoints.

Token-bucket and fixed-window are both tempting for their simplicity, but each
has a failure mode that matters for a voice API:

* **Fixed window** lets a client burst up to 2x the limit across a window
  boundary (max at the end of window N, max again at the start of N+1).
* **Token bucket** needs a background refill process or careful lazy-refill
  math to stay correct under concurrency.

A sliding window over a deque of timestamps avoids both at the cost of O(log n)
eviction per request, which is irrelevant at the request rates this endpoint
sees (it gates *new* HTTP/WebSocket connections, not the audio frames inside
an established stream).

This is an in-process limiter. It is correct for a single API process; behind
multiple replicas it under-counts because each process tracks its own window.
That is an acceptable and explicitly noted limitation -- a distributed limiter
needs Redis-backed counters, which is real added complexity the current single
-process deployment target does not need yet. Revisit if/when the API layer is
horizontally scaled.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass

from fastapi import HTTPException, Request, status

from orphus.observability.logging import get_logger

__all__ = ["RateLimitExceeded", "RateLimiter"]

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RateLimitExceeded:
    """Details of a rejected request, for logging and the response body."""

    limit: int
    window_s: int
    retry_after_s: float


class RateLimiter:
    """Per-client sliding-window request limiter.

    Clients are identified by API key when authenticated, falling back to
    remote address. Using the API key when present means one client behind a
    shared NAT (a corporate proxy, a carrier-grade NAT) does not exhaust a
    single bucket for every other client behind the same address.
    """

    def __init__(self, *, max_requests: int, window_s: int) -> None:
        """Initialise the limiter.

        Args:
            max_requests: Requests allowed per client per window.
            window_s: Window length in seconds.
        """
        self._max_requests = max_requests
        self._window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    @staticmethod
    def _client_id(request: Request) -> str:
        api_key = request.headers.get("x-api-key")
        if api_key:
            # Keys can be long; only a stable, non-reversible slice is needed
            # to bucket by identity, and truncating keeps memory bounded even
            # if a client rotates through many keys.
            return f"key:{api_key[-16:]}"
        if request.client:
            return f"ip:{request.client.host}"
        return "unknown"

    def _evict_expired(self, hits: deque[float], now: float) -> None:
        cutoff = now - self._window_s
        while hits and hits[0] < cutoff:
            hits.popleft()

    def check(self, request: Request) -> RateLimitExceeded | None:
        """Record one hit and report whether it exceeds the limit.

        Returns:
            ``None`` if the request is admitted, otherwise the details of the
            violation for the caller to turn into a 429.
        """
        now = time.monotonic()
        client_id = self._client_id(request)
        hits = self._hits[client_id]
        self._evict_expired(hits, now)

        if len(hits) >= self._max_requests:
            retry_after = self._window_s - (now - hits[0])
            return RateLimitExceeded(
                limit=self._max_requests,
                window_s=self._window_s,
                retry_after_s=max(retry_after, 0.0),
            )

        hits.append(now)
        return None

    def sweep(self) -> int:
        """Drop tracking for clients with no hits in the current window.

        Call periodically (e.g. from the scheduler's background sweeper) so
        long-running processes do not accumulate one deque per distinct client
        forever.

        Returns:
            Number of client buckets removed.
        """
        now = time.monotonic()
        stale = []
        for client_id, hits in self._hits.items():
            self._evict_expired(hits, now)
            if not hits:
                stale.append(client_id)
        for client_id in stale:
            del self._hits[client_id]
        return len(stale)

    async def __call__(self, request: Request) -> None:
        """FastAPI dependency: raises 429 when the client is over budget."""
        violation = self.check(request)
        if violation is not None:
            logger.warning(
                "rate_limit.exceeded",
                extra={
                    "path": request.url.path,
                    "limit": violation.limit,
                    "window_s": violation.window_s,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Rate limit exceeded: {violation.limit} requests per "
                    f"{violation.window_s}s"
                ),
                headers={"Retry-After": str(int(violation.retry_after_s) + 1)},
            )
