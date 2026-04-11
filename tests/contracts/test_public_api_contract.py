from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.router_public import router
from src.api import database


def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def sample_row() -> dict:
    return {
        "event_id": "us6000test1",
        "event_datetime": "2026-04-11T09:10:04.765000Z",
        "latitude": 39.9153,
        "longitude": 141.5227,
        "depth": 79.374,
        "magnitude": 4.3,
        "prob_7d": 0.443,
        "prob_30d": 0.6169,
        "prob_365d": 0.9919,
        "risk_7d": "🟡 Modéré",
        "risk_30d": "🟠 Élevé",
        "risk_365d": "🔴 Très élevé",
        "scored_at": "2026-04-11T09:50:12.636086Z",
        "model_version": "benchmark_v2__compact22__bestparams_v2",
        "benchmark_id": "benchmark_v2",
        "feature_set_version": "candidate_feature_set_v1",
        "dataset_version": "dataset_v5_dedup",
    }


def expected_event_keys() -> set[str]:
    return {
        "event_id",
        "event_datetime",
        "latitude",
        "longitude",
        "depth",
        "magnitude",
        "prob_7d",
        "prob_30d",
        "prob_365d",
        "risk_7d",
        "risk_30d",
        "risk_365d",
        "risk_7d_code",
        "risk_30d_code",
        "risk_365d_code",
        "scored_at",
        "model_version",
        "benchmark_id",
        "feature_set_version",
        "dataset_version",
    }


def test_api_v1_health_contract():
    client = TestClient(make_app())

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_scored_events_contract(monkeypatch):
    def fake_list_scored_events_public(limit: int = 50):
        assert limit == 50
        return [sample_row()]

    monkeypatch.setattr(database, "list_scored_events_public", fake_list_scored_events_public)

    client = TestClient(make_app())
    response = client.get("/api/v1/scored-events?limit=50")

    assert response.status_code == 200

    payload = response.json()
    assert set(payload.keys()) == {"data", "meta"}
    assert payload["meta"] == {"count": 1, "limit": 50, "rejected": 0}
    assert isinstance(payload["data"], list)
    assert len(payload["data"]) == 1

    item = payload["data"][0]
    assert set(item.keys()) == expected_event_keys()

    assert item["event_id"] == "us6000test1"
    assert item["risk_7d_code"] == "moderate"
    assert item["risk_30d_code"] == "high"
    assert item["risk_365d_code"] == "very_high"

    assert item["model_version"] == "benchmark_v2__compact22__bestparams_v2"
    assert item["benchmark_id"] == "benchmark_v2"
    assert item["feature_set_version"] == "candidate_feature_set_v1"
    assert item["dataset_version"] == "dataset_v5_dedup"

    assert "mlflow_run_id" not in item


def test_list_scored_events_rejects_invalid_rows(monkeypatch):
    bad_row = sample_row()
    bad_row.pop("event_id")

    def fake_list_scored_events_public(limit: int = 50):
        return [sample_row(), bad_row]

    monkeypatch.setattr(database, "list_scored_events_public", fake_list_scored_events_public)

    client = TestClient(make_app())
    response = client.get("/api/v1/scored-events?limit=50")

    assert response.status_code == 200

    payload = response.json()
    assert payload["meta"] == {"count": 1, "limit": 50, "rejected": 1}
    assert len(payload["data"]) == 1
    assert payload["data"][0]["event_id"] == "us6000test1"


def test_get_scored_event_detail_contract(monkeypatch):
    def fake_get_scored_event_public(event_id: str):
        assert event_id == "us6000test1"
        return sample_row()

    monkeypatch.setattr(database, "get_scored_event_public", fake_get_scored_event_public)

    client = TestClient(make_app())
    response = client.get("/api/v1/scored-events/us6000test1")

    assert response.status_code == 200

    payload = response.json()
    assert set(payload.keys()) == {"data"}

    item = payload["data"]
    assert set(item.keys()) == expected_event_keys()
    assert item["event_id"] == "us6000test1"
    assert item["risk_7d_code"] == "moderate"
    assert item["risk_30d_code"] == "high"
    assert item["risk_365d_code"] == "very_high"
    assert "mlflow_run_id" not in item


def test_get_scored_event_not_found(monkeypatch):
    def fake_get_scored_event_public(event_id: str):
        return None

    monkeypatch.setattr(database, "get_scored_event_public", fake_get_scored_event_public)

    client = TestClient(make_app())
    response = client.get("/api/v1/scored-events/unknown-event")

    assert response.status_code == 404
    assert response.json() == {"detail": "Event unknown-event not found"}