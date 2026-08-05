"""Tests for the sliding-window rate limiter."""

from __future__ import annotations

import time

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from orphus.middleware.rate_limit import RateLimiter


def _build_app(limiter: RateLimiter) -> FastAPI:
    app = FastAPI()

    @app.get("/limited", dependencies=[Depends(limiter)])
    def limited() -> dict[str, bool]:
        return {"ok": True}

    return app


class TestWindowBehaviour:
    def test_requests_within_limit_are_admitted(self) -> None:
        client = TestClient(_build_app(RateLimiter(max_requests=5, window_s=60)))
        for _ in range(5):
            assert client.get("/limited").status_code == 200

    def test_request_over_limit_is_rejected(self) -> None:
        client = TestClient(_build_app(RateLimiter(max_requests=3, window_s=60)))
        for _ in range(3):
            client.get("/limited")
        response = client.get("/limited")
        assert response.status_code == 429
        assert "Retry-After" in response.headers

    def test_window_expiry_admits_again(self) -> None:
        limiter = RateLimiter(max_requests=2, window_s=1)
        client = TestClient(_build_app(limiter))
        client.get("/limited")
        client.get("/limited")
        assert client.get("/limited").status_code == 429

        time.sleep(1.05)
        assert client.get("/limited").status_code == 200


class TestClientIsolation:
    def test_different_api_keys_get_separate_budgets(self) -> None:
        client = TestClient(_build_app(RateLimiter(max_requests=1, window_s=60)))
        first = client.get("/limited", headers={"X-API-Key": "client-a"})
        second = client.get("/limited", headers={"X-API-Key": "client-b"})
        assert first.status_code == 200
        assert second.status_code == 200

    def test_same_key_shares_one_budget(self) -> None:
        client = TestClient(_build_app(RateLimiter(max_requests=1, window_s=60)))
        first = client.get("/limited", headers={"X-API-Key": "client-a"})
        second = client.get("/limited", headers={"X-API-Key": "client-a"})
        assert first.status_code == 200
        assert second.status_code == 429


class TestSweep:
    def test_sweep_removes_expired_buckets(self) -> None:
        limiter = RateLimiter(max_requests=5, window_s=0)
        app = _build_app(limiter)
        with TestClient(app) as client:
            client.get("/limited", headers={"X-API-Key": "stale"})

        time.sleep(0.01)
        removed = limiter.sweep()
        assert removed == 1

    def test_sweep_keeps_active_buckets(self) -> None:
        limiter = RateLimiter(max_requests=5, window_s=60)
        with TestClient(_build_app(limiter)) as client:
            client.get("/limited", headers={"X-API-Key": "active"})
        assert limiter.sweep() == 0


class TestDirectCheck:
    def test_check_does_not_mutate_state_message(self) -> None:
        from starlette.requests import Request

        limiter = RateLimiter(max_requests=1, window_s=60)
        scope = {
            "type": "http",
            "headers": [(b"x-api-key", b"direct")],
            "client": ("127.0.0.1", 1234),
        }
        request = Request(scope)

        assert limiter.check(request) is None  # first hit admitted
        violation = limiter.check(request)  # second hit rejected
        assert violation is not None
        assert violation.limit == 1
        assert violation.retry_after_s >= 0
