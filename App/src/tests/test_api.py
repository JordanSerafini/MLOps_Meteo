"""Tests d'intégration — endpoints de l'API FastAPI.

Le modèle est mocké (pas de MLflow requis) et les fixtures d'authentification
viennent de conftest.py.
"""
import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api import auth


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

    def test_predict_returns_200(self, client, entetes_client):
        r = client.post("/predict", json=self.PAYLOAD, headers=entetes_client)
        assert r.status_code == 200

    def test_predict_response_schema(self, client, entetes_client):
        data = client.post("/predict", json=self.PAYLOAD, headers=entetes_client).json()
        assert "rain_tomorrow" in data
        assert "probability" in data
        assert "threshold" in data

    def test_predict_probability_in_range(self, client, entetes_client):
        data = client.post("/predict", json=self.PAYLOAD, headers=entetes_client).json()
        assert 0.0 <= data["probability"] <= 1.0

    def test_predict_empty_payload_accepted(self, client, entetes_client):
        """Tous les champs sont optionnels — un payload vide est valide."""
        r = client.post("/predict", json={}, headers=entetes_client)
        assert r.status_code == 200

    def test_predict_returns_model_version(self, client, entetes_client):
        data = client.post("/predict", json=self.PAYLOAD, headers=entetes_client).json()
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
                jeton = auth.creer_jeton("client", ["predict"])
                with TestClient(app) as c:
                    r = c.post("/predict", json={"Location": "Sydney"},
                               headers={"Authorization": f"Bearer {jeton}"})
                    assert r.status_code == 503


# ---------------------------------------------------------------------------
# Tests — journal des prédictions (JSONL)
# ---------------------------------------------------------------------------
class TestPredictionLog:
    PAYLOAD = {"Location": "Sydney", "Month": 7, "Humidity3pm": 80}

    def test_ecrit_une_ligne_par_prediction(self, client, entetes_client, tmp_path):
        journal = tmp_path / "sous-dossier" / "predictions.jsonl"
        with patch("src.api.main.config.PREDICTION_LOG_PATH", str(journal)):
            client.post("/predict", json=self.PAYLOAD, headers=entetes_client)
            client.post("/predict", json=self.PAYLOAD, headers=entetes_client)

        lignes = journal.read_text(encoding="utf-8").strip().splitlines()
        assert len(lignes) == 2

    def test_contenu_de_la_ligne(self, client, entetes_client, tmp_path):
        journal = tmp_path / "predictions.jsonl"
        with patch("src.api.main.config.PREDICTION_LOG_PATH", str(journal)):
            client.post("/predict", json=self.PAYLOAD, headers=entetes_client)

        ligne = json.loads(journal.read_text(encoding="utf-8").strip())
        assert set(ligne) == {"ts", "request_id", "model_version", "threshold",
                              "proba", "label", "features"}
        assert 0.0 <= ligne["proba"] <= 1.0
        assert isinstance(ligne["label"], bool)
        assert ligne["features"]["Location"] == "Sydney"

    def test_desactive_si_chemin_vide(self, client, entetes_client, tmp_path):
        """Chemin vide = pas de journalisation, et surtout pas d'erreur."""
        with patch("src.api.main.config.PREDICTION_LOG_PATH", ""):
            r = client.post("/predict", json=self.PAYLOAD, headers=entetes_client)
        assert r.status_code == 200
        assert list(tmp_path.iterdir()) == []

    def test_erreur_d_ecriture_ne_casse_pas_la_prediction(self, client, entetes_client, tmp_path):
        """Un chemin inaccessible désactive le journal mais laisse l'API répondre."""
        fichier = tmp_path / "occupe"
        fichier.write_text("")
        with patch("src.api.main.config.PREDICTION_LOG_PATH", str(fichier / "impossible.jsonl")):
            with patch("src.api.main._log_desactive", False):
                r = client.post("/predict", json=self.PAYLOAD, headers=entetes_client)
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Tests — /reload
# ---------------------------------------------------------------------------
class TestReloadEndpoint:
    def test_reload_success(self, client, entetes_admin):
        r = client.post("/reload", headers=entetes_admin)
        assert r.status_code == 200
        assert r.json()["reloaded"] is True

    def test_reload_failure_returns_503(self, client, entetes_admin):
        with patch("src.api.main.load_model", side_effect=RuntimeError("fail")):
            r = client.post("/reload", headers=entetes_admin)
            assert r.status_code == 503
