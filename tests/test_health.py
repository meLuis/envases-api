from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_health_exposes_model_and_version() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["service"] == "envases_backend"
    assert payload["model"] == app.title
    assert payload["version"] == app.version
