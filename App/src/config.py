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

# Journal des prédictions : sert de jeu "courant" à la détection de dérive.
# Chaîne vide -> pas de journalisation (cas des tests).
PREDICTION_LOG_PATH = os.getenv("PREDICTION_LOG_PATH", "")

# Authentification. Pas de valeur par défaut pour le secret : l'API refuse de démarrer
# sans, plutôt que de servir avec un secret d'exemple.
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

API_CLIENT_USER = os.getenv("API_CLIENT_USER", "client")
API_CLIENT_PASSWORD_HASH = os.getenv("API_CLIENT_PASSWORD_HASH", "")
API_ADMIN_USER = os.getenv("API_ADMIN_USER", "admin")
API_ADMIN_PASSWORD_HASH = os.getenv("API_ADMIN_PASSWORD_HASH", "")

# Dérive
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "")
# Seuil calibré, pas choisi au hasard : sur 15 tirages de trafic normal la part de
# colonnes en dérive monte jusqu'à 0,33 (bruit d'échantillonnage), alors qu'un trafic
# volontairement décalé descend rarement sous 0,89. 0,5 sépare les deux sans faux positif.
# Voir Docs/CALIBRATION_DRIFT.md.
DRIFT_SHARE_THRESHOLD = float(os.getenv("DRIFT_SHARE_THRESHOLD", "0.5"))
# Échantillon figé (versionné) : une référence stable dans le temps, et qui permet au job
# de tourner sur le serveur, où le CSV complet n'est pas déployé.
REFERENCE_PATH = os.getenv("REFERENCE_PATH", str(APP_DIR / "reference" / "reference.csv"))

# Données / modèle
TARGET = "RainTomorrow"
RANDOM_STATE = 42
CATEGORICAL_FEATURES = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "Month"]
DROP_COLS = ["Date", TARGET]
