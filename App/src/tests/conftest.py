"""Configuration commune aux tests.

Les comptes et le secret de signature sont fixés ici pour toute la session : sans eux
l'API refuse de démarrer, ce qui est justement le comportement attendu en production.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import config
from src.api import auth

MDP_CLIENT = "mot-de-passe-client"
MDP_ADMIN = "mot-de-passe-admin"


@pytest.fixture(autouse=True, scope="session")
def _configuration_auth():
    with patch.multiple(
        config,
        JWT_SECRET_KEY="secret-de-test-sans-valeur-en-production",
        JWT_EXPIRE_MINUTES=60,
        API_CLIENT_USER="client",
        API_CLIENT_PASSWORD_HASH=auth.hacher(MDP_CLIENT),
        API_ADMIN_USER="admin",
        API_ADMIN_PASSWORD_HASH=auth.hacher(MDP_ADMIN),
    ):
        yield


@pytest.fixture()
def mock_model():
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
    with patch("src.api.main._state", {"model": mock_model, "version": "42"}):
        with patch("src.api.main.load_model", return_value=mock_model):
            from src.api.main import app
            with TestClient(app) as c:
                yield c


def _jeton(client, identifiant, mot_de_passe):
    r = client.post("/token", data={"username": identifiant, "password": mot_de_passe})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


@pytest.fixture()
def entetes_client(client):
    return {"Authorization": f"Bearer {_jeton(client, 'client', MDP_CLIENT)}"}


@pytest.fixture()
def entetes_admin(client):
    return {"Authorization": f"Bearer {_jeton(client, 'admin', MDP_ADMIN)}"}
