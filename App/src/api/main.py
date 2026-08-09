"""API d'inférence FastAPI.

Charge le modèle depuis le MLflow Model Registry (alias 'champion') et expose :
  POST /token    -> jeton JWT (identifiant + mot de passe)
  GET  /health   -> état + version du modèle (ouvert)
  POST /predict  -> probabilité de pluie demain (portée 'predict')
  POST /reload   -> recharge le modèle après un entraînement (portée 'admin')
  GET  /metrics  -> métriques Prometheus (ouvert en interne, fermé par nginx)
"""
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import OAuth2PasswordRequestForm
from mlflow.tracking import MlflowClient
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from .. import config
from .auth import authentifier, creer_jeton, utilisateur_courant, verifier_configuration
from .schemas import PredictionOut, WeatherFeatures

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

_state = {"model": None, "version": None}

PRED_COUNTER = Counter("rain_predictions_total", "Nombre de prédictions", ["outcome"])
PROBA_HIST = Histogram("rain_prediction_proba", "Distribution des probabilités prédites",
                       buckets=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# /health dit déjà si le modèle est chargé, mais Prometheus ne lit pas /health :
# sans cette jauge on ne peut pas alerter sur une API debout mais inutilisable.
MODELE_CHARGE = Gauge("rain_model_loaded", "1 si un modèle est chargé en mémoire")

_log_desactive = False


def journalise(features: dict, proba: float, rain: bool):
    """Ajoute une ligne au journal JSONL des prédictions.

    Ce fichier est le jeu de données "courant" comparé à la référence par le job
    de détection de dérive. Une erreur d'écriture ne doit jamais faire échouer une
    prédiction : on désactive la journalisation et on continue.
    """
    global _log_desactive
    if not config.PREDICTION_LOG_PATH or _log_desactive:
        return
    ligne = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "request_id": str(uuid.uuid4()),
        "model_version": _state["version"],
        "threshold": config.DECISION_THRESHOLD,
        "proba": round(proba, 6),
        "label": rain,
        "features": features,
    }
    try:
        chemin = Path(config.PREDICTION_LOG_PATH)
        chemin.parent.mkdir(parents=True, exist_ok=True)
        with chemin.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ligne, ensure_ascii=False) + "\n")
    except OSError as e:
        _log_desactive = True
        log.warning("Journalisation désactivée (%s) : %s", config.PREDICTION_LOG_PATH, e)


def load_model():
    """Charge le Pipeline sklearn depuis le registry + récupère sa version."""
    model = mlflow.sklearn.load_model(config.MODEL_URI)
    version = None
    try:
        mv = MlflowClient().get_model_version_by_alias(config.MODEL_NAME, config.MODEL_ALIAS)
        version = mv.version
    except Exception:  # noqa: BLE001
        pass
    _state["model"], _state["version"] = model, version
    MODELE_CHARGE.set(1)
    log.info("Modèle chargé depuis %s (v%s)", config.MODEL_URI, version)
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    verifier_configuration()
    MODELE_CHARGE.set(0)
    try:
        load_model()
    except Exception as e:  # noqa: BLE001
        log.warning("Modèle non chargé au démarrage (entraînement requis ?): %s", e)
    yield


app = FastAPI(title="Rain in Australia — Inference API", version="0.2.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)  # GET /metrics


@app.post("/token")
def token(formulaire: OAuth2PasswordRequestForm = Depends()):
    """Échange identifiant + mot de passe contre un jeton valable JWT_EXPIRE_MINUTES."""
    utilisateur = authentifier(formulaire.username, formulaire.password)
    if utilisateur is None:
        # Message volontairement vague : ne pas indiquer lequel des deux est faux.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiants invalides",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {
        "access_token": creer_jeton(utilisateur["identifiant"], utilisateur["portees"]),
        "token_type": "bearer",
        "expires_in": config.JWT_EXPIRE_MINUTES * 60,
        "scope": " ".join(utilisateur["portees"]),
    }


@app.get("/health")
def health():
    """Reste ouvert : le healthcheck Docker, la sonde Airflow et Prometheus s'en servent."""
    return {
        "status": "ok",
        "model_loaded": _state["model"] is not None,
        "model_version": _state["version"],
        "model_uri": config.MODEL_URI,
        "decision_threshold": config.DECISION_THRESHOLD,
    }


@app.post("/reload")
def reload(utilisateur=Security(utilisateur_courant, scopes=["admin"])):
    try:
        load_model()
        log.info("Modèle rechargé par %s", utilisateur["identifiant"])
        return {"reloaded": True, "model_version": _state["version"]}
    except Exception as e:  # noqa: BLE001
        MODELE_CHARGE.set(0)
        raise HTTPException(status_code=503, detail=f"reload impossible: {e}")


@app.post("/predict", response_model=PredictionOut)
def predict(features: WeatherFeatures,
            utilisateur=Security(utilisateur_courant, scopes=["predict"])):
    if _state["model"] is None:
        try:
            load_model()
        except Exception as e:  # noqa: BLE001
            MODELE_CHARGE.set(0)
            raise HTTPException(status_code=503, detail=f"modèle indisponible: {e}")
    model = _state["model"]
    recu = features.model_dump()
    # reindex sur les colonnes vues à l'entraînement ; champs manquants -> NaN -> imputés
    df = pd.DataFrame([recu]).reindex(columns=list(model.feature_names_in_))
    proba = float(model.predict_proba(df)[0, 1])
    rain = bool(proba >= config.DECISION_THRESHOLD)
    PRED_COUNTER.labels(outcome="rain" if rain else "norain").inc()
    PROBA_HIST.observe(proba)
    journalise(recu, proba, rain)
    return PredictionOut(rain_tomorrow=rain, probability=round(proba, 4),
                         threshold=config.DECISION_THRESHOLD,
                         model_version=_state["version"])
