"""Tests de l'authentification OAuth2 et des portées."""
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import jwt
import pytest

from src import config
from src.api import auth
from src.tests.conftest import MDP_ADMIN, MDP_CLIENT

PAYLOAD = {"Location": "Sydney", "Month": 7, "Humidity3pm": 80}


class TestObtentionDuJeton:
    def test_identifiants_valides(self, client):
        r = client.post("/token", data={"username": "client", "password": MDP_CLIENT})
        assert r.status_code == 200
        corps = r.json()
        assert corps["token_type"] == "bearer"
        assert corps["scope"] == "predict"

    def test_admin_recoit_les_deux_portees(self, client):
        r = client.post("/token", data={"username": "admin", "password": MDP_ADMIN})
        assert set(r.json()["scope"].split()) == {"predict", "admin"}

    def test_mauvais_mot_de_passe(self, client):
        r = client.post("/token", data={"username": "client", "password": "faux"})
        assert r.status_code == 401

    def test_compte_inconnu(self, client):
        r = client.post("/token", data={"username": "personne", "password": MDP_CLIENT})
        assert r.status_code == 401

    def test_le_message_ne_distingue_pas_les_deux_cas(self, client):
        """Un message différent selon que le compte existe permet de les énumérer."""
        inconnu = client.post("/token", data={"username": "x", "password": "y"}).json()
        mauvais = client.post("/token", data={"username": "client", "password": "y"}).json()
        assert inconnu["detail"] == mauvais["detail"]


class TestAccesPredict:
    def test_jeton_valide(self, client, entetes_client):
        assert client.post("/predict", json=PAYLOAD, headers=entetes_client).status_code == 200

    def test_sans_jeton(self, client):
        assert client.post("/predict", json=PAYLOAD).status_code == 401

    def test_jeton_bidon(self, client):
        entetes = {"Authorization": "Bearer pas-un-jeton"}
        assert client.post("/predict", json=PAYLOAD, headers=entetes).status_code == 401

    def test_jeton_expire(self, client):
        expire = jwt.encode(
            {"sub": "client", "scopes": ["predict"],
             "exp": datetime.now(UTC) - timedelta(minutes=1)},
            config.JWT_SECRET_KEY, algorithm=config.JWT_ALGORITHM)
        entetes = {"Authorization": f"Bearer {expire}"}
        r = client.post("/predict", json=PAYLOAD, headers=entetes)
        assert r.status_code == 401
        assert "expiré" in r.json()["detail"]

    def test_signature_falsifiee(self, client):
        """Un jeton bien formé mais signé avec une autre clé doit être rejeté."""
        faux = jwt.encode(
            {"sub": "admin", "scopes": ["predict", "admin"],
             "exp": datetime.now(UTC) + timedelta(hours=1)},
            "une-autre-cle", algorithm=config.JWT_ALGORITHM)
        entetes = {"Authorization": f"Bearer {faux}"}
        assert client.post("/predict", json=PAYLOAD, headers=entetes).status_code == 401


class TestAccesReload:
    def test_le_client_est_refuse(self, client, entetes_client):
        """Authentifié mais pas autorisé : 403 et non 401."""
        r = client.post("/reload", headers=entetes_client)
        assert r.status_code == 403
        assert "admin" in r.json()["detail"]

    def test_l_admin_est_accepte(self, client, entetes_admin):
        assert client.post("/reload", headers=entetes_admin).status_code == 200

    def test_sans_jeton(self, client):
        assert client.post("/reload").status_code == 401


class TestEndpointsOuverts:
    def test_health_reste_accessible(self, client):
        """Le healthcheck Docker et la sonde Airflow n'ont pas de jeton."""
        assert client.get("/health").status_code == 200

    def test_metrics_reste_accessible(self, client):
        """Prometheus scrute par le réseau interne ; nginx ferme l'accès externe."""
        assert client.get("/metrics").status_code == 200


class TestVerificationConfiguration:
    def test_refuse_un_secret_absent(self):
        with patch.object(config, "JWT_SECRET_KEY", ""):
            with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
                auth.verifier_configuration()

    def test_refuse_le_secret_d_exemple(self):
        with patch.object(config, "JWT_SECRET_KEY", "CHANGEME"):
            with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
                auth.verifier_configuration()

    def test_refuse_l_absence_de_compte(self):
        with patch.multiple(config, API_CLIENT_PASSWORD_HASH="", API_ADMIN_PASSWORD_HASH=""):
            with pytest.raises(RuntimeError, match="Aucun compte"):
                auth.verifier_configuration()

    def test_detecte_un_hachage_tronque(self):
        """Le cas du $ non doublé dans .env.api, qui refuserait tout sans rien dire."""
        with patch.object(config, "API_CLIENT_PASSWORD_HASH", "$2b$12"):
            with pytest.raises(RuntimeError, match="doublés"):
                auth.verifier_configuration()


class TestHachage:
    def test_aller_retour(self):
        assert auth.mot_de_passe_valide("coucou", auth.hacher("coucou"))

    def test_mauvais_mot_de_passe(self):
        assert not auth.mot_de_passe_valide("autre", auth.hacher("coucou"))

    def test_hachage_malforme(self):
        assert not auth.mot_de_passe_valide("coucou", "pas-un-hachage")

    def test_mot_de_passe_trop_long(self):
        """bcrypt refuse au-delà de 72 octets : ne pas laisser remonter l'exception."""
        assert not auth.mot_de_passe_valide("x" * 100, auth.hacher("coucou"))
