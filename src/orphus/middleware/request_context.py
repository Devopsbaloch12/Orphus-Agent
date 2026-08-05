"""ASGI middleware binding a request ID and timing to every log line.

Sits above everything else in the stack so that even a 500 raised by another
middleware (auth, rate limiting) is logged with a request ID the client can
quote back for support. The ID is echoed in ``X-Request-ID`` so client and
server logs can be correlated without a distributed tracing backend.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from orphus.observability.logging import bind_session, get_logger
from orphus.observability.metrics import get_metrics

__all__ = ["RequestContextMiddleware"]

logger = get_logger(__name__)

_REQUEST_ID_HEADER = "x-request-id"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a request ID, times the request, and logs the outcome."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Wrap one request with an ID, timing, and structured access log."""
        request_id = request.headers.get(_REQUEST_ID_HEADER) or f"req_{uuid.uuid4().hex[:16]}"
        request.state.request_id = request_id
        started = time.perf_counter()

        # Reusing the session binding context var: for HTTP requests the
        # "session" in the log sense is the request itself, giving every log
        # line during this call the same correlation id without threading it
        # through every function signature.
        with bind_session(request_id):
            try:
                response = await call_next(request)
            except Exception:
                elapsed_ms = (time.perf_counter() - started) * 1000
                logger.exception(
                    "http.request_failed",
                    extra={
                        "method": request.method,
                        "path": request.url.path,
                        "elapsed_ms": round(elapsed_ms, 2),
                    },
                )
                get_metrics().record_error("api", RuntimeError("unhandled"))
                raise

            elapsed_ms = (time.perf_counter() - started) * 1000
            response.headers[_REQUEST_ID_HEADER] = request_id

            log_level = "warning" if response.status_code >= 400 else "info"
            getattr(logger, log_level)(
                "http.request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "elapsed_ms": round(elapsed_ms, 2),
                },
            )
            return response
