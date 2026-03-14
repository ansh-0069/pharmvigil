"""
Tests for the FastAPI application endpoints.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure src is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI test client."""
    from src.api.app import app
    return TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def ensure_models(sample_raw_df):
    """Run a quick training before API tests so models & scores exist."""
    from src.models.train_model import train_pipeline
    train_pipeline(df=sample_raw_df.copy())


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_health_reports_model_status(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert isinstance(data["models_loaded"], bool)


class TestPredictEndpoint:
    def test_predict_valid_pair(self, client):
        resp = client.post(
            "/predict",
            json={"drug_name": "warfarin", "adverse_event": "thrombocytopenia"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "risk_score" in data
        assert "signal_strength" in data
        assert "alert_level" in data
        assert 0 <= data["risk_score"] <= 1

    def test_predict_unknown_pair(self, client):
        resp = client.post(
            "/predict",
            json={"drug_name": "unknowndrug_xyz", "adverse_event": "unknownevent_xyz"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "risk_score" in data

    def test_predict_missing_fields(self, client):
        resp = client.post("/predict", json={"drug_name": "warfarin"})
        assert resp.status_code == 422  # validation error

    def test_predict_empty_body(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422


class TestTopSignalsEndpoint:
    def test_top_signals_default(self, client):
        resp = client.get("/top-signals")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        if data:
            assert "drug_name" in data[0]
            assert "risk_score" in data[0]

    def test_top_signals_with_n(self, client):
        resp = client.get("/top-signals?n=5")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) <= 5

    def test_top_signals_sorted(self, client):
        resp = client.get("/top-signals?n=10")
        data = resp.json()
        if len(data) >= 2:
            scores = [item["risk_score"] for item in data]
            assert scores == sorted(scores, reverse=True)


class TestStatsEndpoint:
    def test_stats_returns_200(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_scored_pairs" in data
        assert "alert_distribution" in data
