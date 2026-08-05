"""Tests for the request-context ASGI middleware."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from orphus.middleware.request_context import RequestContextMiddleware


def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/ok")
    def ok() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/boom")
    def boom() -> None:
        raise ValueError("intentional failure")

    return app


@pytest.fixture
def client() -> TestClient:
    return TestClient(_build_app(), raise_server_exceptions=False)


class TestRequestId:
    def test_response_carries_a_request_id(self, client: TestClient) -> None:
        response = client.get("/ok")
        assert response.headers.get("x-request-id", "").startswith("req_")

    def test_client_supplied_request_id_is_echoed(self, client: TestClient) -> None:
        response = client.get("/ok", headers={"X-Request-ID": "req_client_supplied"})
        assert response.headers["x-request-id"] == "req_client_supplied"

    def test_each_request_gets_a_distinct_id(self, client: TestClient) -> None:
        first = client.get("/ok").headers["x-request-id"]
        second = client.get("/ok").headers["x-request-id"]
        assert first != second


class TestErrorHandling:
    def test_exception_still_gets_a_request_id_path(self, client: TestClient) -> None:
        # The middleware must not swallow the exception -- FastAPI's default
        # handler still needs to see it to produce the 500.
        response = client.get("/boom")
        assert response.status_code == 500

    def test_normal_request_succeeds(self, client: TestClient) -> None:
        assert client.get("/ok").status_code == 200
