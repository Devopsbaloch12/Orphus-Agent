"""Tests for API key authentication."""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from orphus.middleware.auth import ApiKeyAuthenticator


def _build_app(authenticator: ApiKeyAuthenticator) -> FastAPI:
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(authenticator)])
    def protected() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    return app


class TestDisabled:
    def test_no_keys_configured_means_open_access(self) -> None:
        client = TestClient(_build_app(ApiKeyAuthenticator([])))
        assert client.get("/protected").status_code == 200

    def test_enabled_is_false_when_empty(self) -> None:
        assert ApiKeyAuthenticator([]).enabled is False


class TestEnabled:
    @pytest.fixture
    def client(self) -> TestClient:
        return TestClient(_build_app(ApiKeyAuthenticator(["valid-key-1", "valid-key-2"])))

    def test_enabled_is_true_when_keys_present(self) -> None:
        assert ApiKeyAuthenticator(["k"]).enabled is True

    def test_missing_key_is_rejected(self, client: TestClient) -> None:
        response = client.get("/protected")
        assert response.status_code == 401

    def test_valid_bearer_token_is_accepted(self, client: TestClient) -> None:
        response = client.get(
            "/protected", headers={"Authorization": "Bearer valid-key-1"}
        )
        assert response.status_code == 200

    def test_valid_x_api_key_header_is_accepted(self, client: TestClient) -> None:
        response = client.get("/protected", headers={"X-API-Key": "valid-key-2"})
        assert response.status_code == 200

    def test_second_configured_key_also_works(self, client: TestClient) -> None:
        response = client.get("/protected", headers={"X-API-Key": "valid-key-1"})
        assert response.status_code == 200

    def test_wrong_key_is_rejected(self, client: TestClient) -> None:
        response = client.get("/protected", headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 401

    def test_wrong_scheme_is_rejected(self, client: TestClient) -> None:
        response = client.get(
            "/protected", headers={"Authorization": "Basic valid-key-1"}
        )
        assert response.status_code == 401

    def test_x_api_key_takes_precedence_over_bearer(self, client: TestClient) -> None:
        response = client.get(
            "/protected",
            headers={"X-API-Key": "valid-key-1", "Authorization": "Bearer wrong-key"},
        )
        assert response.status_code == 200

    def test_unprotected_route_is_unaffected(self, client: TestClient) -> None:
        assert client.get("/health").status_code == 200

    def test_response_includes_www_authenticate(self, client: TestClient) -> None:
        response = client.get("/protected")
        assert response.headers.get("www-authenticate") == "Bearer"
