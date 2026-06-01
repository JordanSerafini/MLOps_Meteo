"""Configuration centralisée (lue par train.py et l'API)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.getenv("DATA_PATH", ROOT / "Data" / "weatherAUS.csv"))

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "rain-australia")
MODEL_NAME = os.getenv("MODEL_NAME", "rain-australia")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
# URI de chargement côté API (alias = recommandé MLflow 2.9+, remplace les stages dépréciés)
MODEL_URI = os.getenv("MODEL_URI", f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

# Données / modèle
TARGET = "RainTomorrow"
RANDOM_STATE = 42
# Colonnes traitées comme catégorielles (le reste = numérique)
CATEGORICAL_FEATURES = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "Month"]
# Colonnes retirées des features (Date -> feature Month ; TARGET = cible)
DROP_COLS = ["Date", TARGET]
