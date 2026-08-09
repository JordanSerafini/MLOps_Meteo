"""
DAG Airflow — Pipeline de déploiement (split strict).

L'entraînement est DÉPORTÉ sur le PC de dev (specs serveur limitées) : il pousse le
modèle `@champion` directement dans le registry MLflow du serveur (mode --serve-artifacts).
Ce DAG ne fait donc QUE le déploiement, déclenché à distance après un entraînement réussi :

  1. wait_for_api       → attend que l'API réponde (/health status ok)
  2. obtenir_jeton      → échange les identifiants admin contre un jeton JWT
  3. reload_api         → recharge le modèle @champion depuis le registry (/reload)
  4. check_model_loaded → confirme que le modèle est bien chargé (/health model_loaded)
  5. smoke_test         → une prédiction de bout en bout (/predict)
  6. verify_prediction  → valide la cohérence de la réponse

/reload exige la portée admin, /predict la portée predict : le jeton demandé est donc
celui du compte admin, qui a les deux.

Déclenchement : manuel ou via l'API REST Airflow depuis le PC de dev
(`POST /api/v1/dags/rain_australia_deploy_pipeline/dagRuns`). Aucun entraînement,
aucun accès Docker ni CSV côté serveur.
"""
import json
import os
from datetime import datetime, timedelta

import requests
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.providers.http.sensors.http import HttpSensor

API_URL = os.getenv("API_URL", "http://api:8000")
JETON = "{{ ti.xcom_pull(task_ids='obtenir_jeton') }}"

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

def recuperer_jeton(**kwargs):
    """Échange les identifiants admin contre un jeton, poussé en XCom pour la suite."""
    identifiant = os.getenv("API_ADMIN_USER", "admin")
    mot_de_passe = os.getenv("API_ADMIN_PASSWORD", "")
    if not mot_de_passe:
        raise RuntimeError(
            "API_ADMIN_PASSWORD absent de l'environnement Airflow : le DAG ne peut pas "
            "s'authentifier auprès de l'API."
        )

    r = requests.post(f"{API_URL}/token",
                      data={"username": identifiant, "password": mot_de_passe},
                      timeout=15)
    if r.status_code == 401:
        raise RuntimeError("Identifiants admin refusés par l'API.")
    r.raise_for_status()

    corps = r.json()
    print(f"Jeton obtenu pour '{identifiant}', portées : {corps.get('scope')}")
    return corps["access_token"]


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

    # 2. S'authentifier (le jeton sert aux étapes 3 et 5)
    jeton = PythonOperator(
        task_id="obtenir_jeton",
        python_callable=recuperer_jeton,
    )

    # 3. Recharger le modèle @champion depuis le registry MLflow
    reload_api = SimpleHttpOperator(
        task_id="reload_api",
        http_conn_id="rain_api",
        endpoint="/reload",
        method="POST",
        headers={"Authorization": f"Bearer {JETON}"},
        response_check=lambda response: response.json().get("reloaded") is True,
        log_response=True,
    )

    # 4. Confirmer que le modèle est effectivement chargé
    check_model_loaded = SimpleHttpOperator(
        task_id="check_model_loaded",
        http_conn_id="rain_api",
        endpoint="/health",
        method="GET",
        response_check=lambda response: response.json().get("model_loaded") is True,
        log_response=True,
    )

    # 5. Smoke test — une prédiction de bout en bout
    smoke_test = SimpleHttpOperator(
        task_id="smoke_test",
        http_conn_id="rain_api",
        endpoint="/predict",
        method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {JETON}"},
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
    wait_api >> jeton >> reload_api >> check_model_loaded >> smoke_test >> verify
