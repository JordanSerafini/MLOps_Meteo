"""Configuration centralisée (lue par train.py et l'API)."""
import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent       # .../MlOps_Meteo-Liora/App
PROJECT_ROOT = APP_DIR.parent                           # .../MlOps_Meteo-Liora (racine du repo)
# Par défaut le CSV est à la racine du projet (Data/) ; en conteneur DATA_PATH est forcé (/data/...)
DATA_PATH = Path(os.getenv("DATA_PATH", PROJECT_ROOT / "Data" / "weatherAUS.csv"))

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "rain-australia")
MODEL_NAME = os.getenv("MODEL_NAME", "rain-australia")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
# URI de chargement côté API (alias = recommandé MLflow 2.9+, remplace les stages dépréciés)
MODEL_URI = os.getenv("MODEL_URI", f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

# Décision / promotion
# Seuil de décision pour classer "pluie" (levier principal contre le déséquilibre).
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
# Gating de l'alias champion : ne promeut le modèle que si recall_rain >= ce seuil.
# Défaut 0.0 = pas de gating (ne casse pas la démo). Mettre p.ex. 0.6 pour exiger un recall pluie.
MIN_RECALL_FOR_CHAMPION = float(os.getenv("MIN_RECALL_FOR_CHAMPION", "0.0"))

# Données / modèle
TARGET = "RainTomorrow"
RANDOM_STATE = 42
# Colonnes traitées comme catégorielles (le reste = numérique)
CATEGORICAL_FEATURES = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "Month"]
# Colonnes retirées des features (Date -> feature Month ; TARGET = cible)
DROP_COLS = ["Date", TARGET]
