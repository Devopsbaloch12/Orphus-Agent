from fastapi.testclient import TestClient

from orphus.api.app import create_app
from orphus.config import Settings


def test_session_crud() -> None:
    with TestClient(create_app(Settings())) as client:
        created = client.post("/v1/sessions", json={"voice": "tara"})
        assert created.status_code == 201
        session_id = created.json()["session_id"]
        assert client.get(f"/v1/sessions/{session_id}").status_code == 200
        assert client.delete(f"/v1/sessions/{session_id}").status_code == 204
        assert client.get(f"/v1/sessions/{session_id}").status_code == 404


def test_capacity_returns_service_unavailable() -> None:
    settings = Settings(session={"max_concurrent": 1})
    with TestClient(create_app(settings)) as client:
        assert client.post("/v1/sessions", json={}).status_code == 201
        assert client.post("/v1/sessions", json={}).status_code == 503


def test_health_and_metrics_are_public() -> None:
    with TestClient(create_app(Settings())) as client:
        assert client.get("/health").status_code == 200
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "text/plain" in metrics.headers["content-type"]
