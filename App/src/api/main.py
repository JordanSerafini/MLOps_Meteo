"""API d'inférence FastAPI.

Charge le modèle depuis le MLflow Model Registry (alias 'champion') et expose :
  GET  /health   -> état + version du modèle
  POST /predict  -> probabilité de pluie demain
  POST /reload    -> recharge le modèle (après un nouvel entraînement)
  GET  /metrics  -> métriques Prometheus (pré-câblage monitoring Phase 4)
"""
import logging
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from .. import config
from .schemas import PredictionOut, WeatherFeatures

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")

mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

_state = {"model": None, "version": None}

PRED_COUNTER = Counter("rain_predictions_total", "Nombre de prédictions", ["outcome"])
PROBA_HIST = Histogram("rain_prediction_proba", "Distribution des probabilités prédites",
                       buckets=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


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
    log.info("Modèle chargé depuis %s (v%s)", config.MODEL_URI, version)
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except Exception as e:  # noqa: BLE001
        log.warning("Modèle non chargé au démarrage (entraînement requis ?): %s", e)
    yield


app = FastAPI(title="Rain in Australia — Inference API", version="0.1.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)  # GET /metrics


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _state["model"] is not None,
        "model_version": _state["version"],
        "model_uri": config.MODEL_URI,
        "decision_threshold": config.DECISION_THRESHOLD,
    }


@app.post("/reload")
def reload():
    try:
        load_model()
        return {"reloaded": True, "model_version": _state["version"]}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"reload impossible: {e}")


@app.post("/predict", response_model=PredictionOut)
def predict(features: WeatherFeatures):
    if _state["model"] is None:
        try:
            load_model()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"modèle indisponible: {e}")
    model = _state["model"]
    # reindex sur les colonnes vues à l'entraînement ; champs manquants -> NaN -> imputés
    df = pd.DataFrame([features.model_dump()]).reindex(columns=list(model.feature_names_in_))
    proba = float(model.predict_proba(df)[0, 1])
    rain = bool(proba >= config.DECISION_THRESHOLD)
    PRED_COUNTER.labels(outcome="rain" if rain else "norain").inc()
    PROBA_HIST.observe(proba)
    return PredictionOut(rain_tomorrow=rain, probability=round(proba, 4),
                         threshold=config.DECISION_THRESHOLD,
                         model_version=_state["version"])
