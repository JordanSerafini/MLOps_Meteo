import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent       # .../MlOps_Meteo-Liora/App
PROJECT_ROOT = APP_DIR.parent                           # .../MlOps_Meteo-Liora (racine du repo)
DATA_PATH = Path(os.getenv("DATA_PATH", PROJECT_ROOT / "Data" / "weatherAUS.csv"))

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "rain-australia")
MODEL_NAME = os.getenv("MODEL_NAME", "rain-australia")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MODEL_URI = os.getenv("MODEL_URI", f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

# Décision / promotion
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
MIN_RECALL_FOR_CHAMPION = float(os.getenv("MIN_RECALL_FOR_CHAMPION", "0.0"))

# Données / modèle
TARGET = "RainTomorrow"
RANDOM_STATE = 42
CATEGORICAL_FEATURES = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "Month"]
DROP_COLS = ["Date", TARGET]
