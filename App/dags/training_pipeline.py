"""
DAG Airflow — Pipeline de déploiement (split strict).

L'entraînement est DÉPORTÉ sur le PC de dev (specs serveur limitées) : il pousse le
modèle `@champion` directement dans le registry MLflow du serveur (mode --serve-artifacts).
Ce DAG ne fait donc QUE le déploiement, déclenché à distance après un entraînement réussi :

  1. wait_for_api       → attend que l'API réponde (/health status ok)
  2. reload_api         → recharge le modèle @champion depuis le registry (/reload)
  3. check_model_loaded → confirme que le modèle est bien chargé (/health model_loaded)
  4. smoke_test         → une prédiction de bout en bout (/predict)
  5. verify_prediction  → valide la cohérence de la réponse

Déclenchement : manuel ou via l'API REST Airflow depuis le PC de dev
(`POST /api/v1/dags/rain_australia_deploy_pipeline/dagRuns`). Aucun entraînement,
aucun accès Docker ni CSV côté serveur.
"""
import json
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.providers.http.sensors.http import HttpSensor

# ─── Configuration par défaut ─────────────────────────────────────────
default_args = {
    "owner": "liora-mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


# ─── Fonctions Python ────────────────────────────────────────────────

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
    dag_id="rain_australia_deploy_pipeline",
    description="Pipeline MLOps (déploiement) : reload → check → smoke test",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=None,  # déclenché à distance après un entraînement sur le PC de dev
    catchup=False,
    tags=["mlops", "deploy", "rain-australia"],
) as dag:

    # 1. Attendre que l'API soit disponible
    wait_api = HttpSensor(
        task_id="wait_for_api",
        http_conn_id="rain_api",
        endpoint="/health",
        response_check=lambda response: response.json().get("status") == "ok",
        poke_interval=10,
        timeout=120,
    )

    # 2. Recharger le modèle @champion depuis le registry MLflow
    reload_api = SimpleHttpOperator(
        task_id="reload_api",
        http_conn_id="rain_api",
        endpoint="/reload",
        method="POST",
        response_check=lambda response: response.json().get("reloaded") is True,
        log_response=True,
    )

    # 3. Confirmer que le modèle est effectivement chargé
    check_model_loaded = SimpleHttpOperator(
        task_id="check_model_loaded",
        http_conn_id="rain_api",
        endpoint="/health",
        method="GET",
        response_check=lambda response: response.json().get("model_loaded") is True,
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
    wait_api >> reload_api >> check_model_loaded >> smoke_test >> verify
