"""
DAG Airflow — Pipeline d'entraînement de bout en bout.

Orchestration :
  1. validate_data   → vérifie que le CSV existe et n'est pas vide
  2. train_model     → entraîne le modèle (RandomForest) et l'enregistre dans MLflow
  3. reload_api      → recharge le modèle dans l'API de production
  4. smoke_test      → vérifie qu'une prédiction fonctionne après le déploiement

Planification : chaque dimanche à 02h00 (modifiable via schedule).
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.providers.http.sensors.http import HttpSensor
import json

# ─── Configuration par défaut ─────────────────────────────────────────
default_args = {
    "owner": "liora-mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ─── Fonctions Python ────────────────────────────────────────────────

def validate_data(**kwargs):
    """Vérifie que le fichier de données existe et contient des lignes."""
    import pandas as pd
    import os

    data_path = os.getenv("DATA_PATH", "/data/weatherAUS.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Fichier de données introuvable : {data_path}")

    df = pd.read_csv(data_path, nrows=5)
    n_cols = len(df.columns)
    if n_cols < 10:
        raise ValueError(f"Le fichier semble corrompu : seulement {n_cols} colonnes")

    # Compter le nombre total de lignes (rapide avec wc -l ou pandas)
    n_rows = sum(1 for _ in open(data_path)) - 1  # -1 pour le header
    print(f"✅ Données validées : {n_rows} lignes, {n_cols} colonnes")

    # Passer les infos au XCom pour les tâches suivantes
    kwargs["ti"].xcom_push(key="n_rows", value=n_rows)
    kwargs["ti"].xcom_push(key="n_cols", value=n_cols)


def check_smoke_test_response(**kwargs):
    """Vérifie que la réponse de l'API est cohérente."""
    response = kwargs["ti"].xcom_pull(task_ids="smoke_test")
    data = json.loads(response) if isinstance(response, str) else response
    assert "rain_tomorrow" in data, "Réponse invalide : champ 'rain_tomorrow' absent"
    assert "probability" in data, "Réponse invalide : champ 'probability' absent"
    prob = data["probability"]
    assert 0.0 <= prob <= 1.0, f"Probabilité hors bornes : {prob}"
    print(f"✅ Smoke test OK — probabilité: {prob}, pluie: {data['rain_tomorrow']}")


# ─── DAG ──────────────────────────────────────────────────────────────
with DAG(
    dag_id="rain_australia_training_pipeline",
    description="Pipeline MLOps : validation → entraînement → déploiement → smoke test",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="0 2 * * 0",  # chaque dimanche à 02h00
    catchup=False,
    tags=["mlops", "training", "rain-australia"],
) as dag:

    # 1. Validation des données
    validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    # 2. Entraînement du modèle (via le conteneur trainer existant)
    train = DockerOperator(
        task_id="train_model",
        image="meteo-liora-trainer:latest",
        command="--model rf --register",
        environment={
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "DATA_PATH": "/data/weatherAUS.csv",
        },
        network_mode="meteo-liora_mlops",
        mounts=[
            # Monte le dossier Data en lecture seule
            {"source": "/data", "target": "/data", "type": "bind", "read_only": True},
        ],
        auto_remove="success",
        docker_url="unix://var/run/docker.sock",
    )

    # 3. Attendre que l'API soit disponible puis recharger le modèle
    wait_api = HttpSensor(
        task_id="wait_for_api",
        http_conn_id="rain_api",
        endpoint="/health",
        response_check=lambda response: response.json().get("status") == "ok",
        poke_interval=10,
        timeout=120,
    )

    reload_api = SimpleHttpOperator(
        task_id="reload_api",
        http_conn_id="rain_api",
        endpoint="/reload",
        method="POST",
        response_check=lambda response: response.json().get("reloaded") is True,
        log_response=True,
    )

    # 4. Smoke test — une prédiction de bout en bout
    smoke_test = SimpleHttpOperator(
        task_id="smoke_test",
        http_conn_id="rain_api",
        endpoint="/predict",
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "Location": "Sydney", "Month": 7, "RainToday": "Yes",
            "Humidity3pm": 80, "Sunshine": 3.5, "Pressure3pm": 1008.0,
            "Rainfall": 12.0, "WindGustSpeed": 56, "Cloud3pm": 8, "Temp3pm": 16.0,
        }),
        response_check=lambda response: "rain_tomorrow" in response.json(),
        log_response=True,
    )

    verify = PythonOperator(
        task_id="verify_prediction",
        python_callable=check_smoke_test_response,
    )

    # ─── Orchestration ────────────────────────────────────────────────
    validate >> train >> wait_api >> reload_api >> smoke_test >> verify
