"""Tests d'intégration — endpoints de l'API FastAPI.

Utilise un mock du modèle sklearn pour tester l'API indépendamment de MLflow.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def mock_model():
    """Simule un Pipeline sklearn avec predict_proba et feature_names_in_."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    model.feature_names_in_ = np.array([
        "Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation",
        "Sunshine", "WindGustDir", "WindGustSpeed", "WindDir9am", "WindDir3pm",
        "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm",
        "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm",
        "Temp9am", "Temp3pm", "RainToday", "Month",
    ])
    return model


@pytest.fixture()
def client(mock_model):
    """TestClient avec le modèle mocké et réseau bloqué (pas de MLflow requis)."""
    with patch("src.api.main._state", {"model": mock_model, "version": "42"}):
        with patch("src.api.main.load_model", return_value=mock_model):
            from src.api.main import app
            with TestClient(app) as c:
                yield c


# ---------------------------------------------------------------------------
# Tests — /health
# ---------------------------------------------------------------------------
class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_contains_status(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_shows_model_loaded(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True


# ---------------------------------------------------------------------------
# Tests — /predict
# ---------------------------------------------------------------------------
class TestPredictEndpoint:
    PAYLOAD = {
        "Location": "Sydney", "Month": 7, "RainToday": "Yes",
        "Humidity3pm": 80, "Sunshine": 3.5, "Pressure3pm": 1008.0,
        "Rainfall": 12.0, "WindGustSpeed": 56, "Cloud3pm": 8, "Temp3pm": 16.0,
    }

    def test_predict_returns_200(self, client):
        r = client.post("/predict", json=self.PAYLOAD)
        assert r.status_code == 200

    def test_predict_response_schema(self, client):
        data = client.post("/predict", json=self.PAYLOAD).json()
        assert "rain_tomorrow" in data
        assert "probability" in data
        assert "threshold" in data

    def test_predict_probability_in_range(self, client):
        data = client.post("/predict", json=self.PAYLOAD).json()
        assert 0.0 <= data["probability"] <= 1.0

    def test_predict_empty_payload_accepted(self, client):
        """Tous les champs sont optionnels — un payload vide est valide."""
        r = client.post("/predict", json={})
        assert r.status_code == 200

    def test_predict_returns_model_version(self, client):
        data = client.post("/predict", json=self.PAYLOAD).json()
        assert data["model_version"] == "42"


# ---------------------------------------------------------------------------
# Tests — /predict sans modèle chargé
# ---------------------------------------------------------------------------
class TestPredictNoModel:
    def test_returns_503_when_model_unavailable(self):
        """Si le modèle n'est pas chargé et le reload échoue → 503."""
        with patch("src.api.main._state", {"model": None, "version": None}):
            with patch("src.api.main.load_model", side_effect=RuntimeError("no model")):
                from src.api.main import app
                with TestClient(app) as c:
                    r = c.post("/predict", json={"Location": "Sydney"})
                    assert r.status_code == 503


# ---------------------------------------------------------------------------
# Tests — /reload
# ---------------------------------------------------------------------------
class TestReloadEndpoint:
    def test_reload_success(self, client):
        r = client.post("/reload")
        assert r.status_code == 200
        assert r.json()["reloaded"] is True

    def test_reload_failure_returns_503(self, client):
        with patch("src.api.main.load_model", side_effect=RuntimeError("fail")):
            r = client.post("/reload")
            assert r.status_code == 503
