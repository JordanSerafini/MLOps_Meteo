"""
DAG Airflow — surveillance de la dérive.

Lit le résumé produit par le job `src/drift.py` (conteneur `drift`, profil compose du même
nom) et décide s'il faut réentraîner :

  1. lire_resume        → charge dernier_resume.json et vérifie qu'il est récent
  2. decider            → branche selon `retrain_recommande`
  3a. signaler_derive   → trace ce qu'il faut faire (branche « dérive »)
  3b. rien_a_faire      → branche « pas de dérive »

Pourquoi le DAG ne lance-t-il pas le job lui-même ? Le socket Docker a été retiré du
conteneur Airflow au moment du passage en production (cf. deployement/README.md) : lui
redonner la main sur le démon Docker de l'hôte reviendrait à lui donner l'équivalent du root.
Le job tourne donc à côté, déclenché par le Makefile ou une tâche planifiée, et dépose son
résumé dans un volume que ce DAG lit en seule lecture. Pour aller jusqu'au déclenchement
automatique, la solution propre serait un docker-socket-proxy limité à la création de
conteneurs, ou un worker Airflow dédié — hors périmètre ici.

Le réentraînement lui-même reste manuel (`make deploy-model` depuis le PC de dev), puisque
l'entraînement est déporté par conception : le serveur n'a ni le CSV ni l'image trainer.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.python import BranchPythonOperator, PythonOperator

RESUME = Path("/opt/airflow/drift-reports/dernier_resume.json")
AGE_MAX_HEURES = 36

default_args = {
    "owner": "liora-mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def lire_resume(**kwargs):
    """Charge le dernier résumé de dérive et vérifie qu'il n'est pas périmé."""
    if not RESUME.exists():
        raise AirflowSkipException(
            f"{RESUME} absent — le job de dérive n'a jamais tourné. "
            "Lancer `make drift` depuis le PC de dev."
        )

    resume = json.loads(RESUME.read_text(encoding="utf-8"))

    if resume.get("statut") == "donnees_insuffisantes":
        raise AirflowSkipException(
            f"Trop peu de prédictions pour conclure ({resume.get('n_courant')} observations)."
        )

    age = datetime.now().astimezone() - datetime.fromisoformat(resume["ts"])
    if age > timedelta(hours=AGE_MAX_HEURES):
        raise AirflowSkipException(
            f"Résumé vieux de {age.total_seconds() / 3600:.0f} h "
            f"(maximum {AGE_MAX_HEURES} h) — relancer le job de dérive."
        )

    print(f"Dérive mesurée le {resume['ts']} : "
          f"{resume['n_colonnes_derivees']} colonnes sur {resume['n_courant']} observations, "
          f"soit {100 * resume['part_colonnes_derivees']:.0f} % "
          f"(seuil {100 * resume['seuil']:.0f} %)")
    return resume


def decider(**kwargs):
    resume = kwargs["ti"].xcom_pull(task_ids="lire_resume")
    return "signaler_derive" if resume["retrain_recommande"] else "rien_a_faire"


def signaler_derive(**kwargs):
    resume = kwargs["ti"].xcom_pull(task_ids="lire_resume")
    print(
        "Dérive confirmée.\n"
        f"  part de colonnes dérivées : {resume['part_colonnes_derivees']} "
        f"(seuil {resume['seuil']})\n"
        f"  rapport détaillé          : {resume['rapport_html']}\n"
        "  action attendue           : `make deploy-model` depuis le PC de dev "
        "(réentraîne, pousse au registry, recharge l'API)."
    )


def rien_a_faire(**kwargs):
    resume = kwargs["ti"].xcom_pull(task_ids="lire_resume")
    print(f"Pas de dérive ({resume['part_colonnes_derivees']} < {resume['seuil']}). "
          "Le modèle en place reste valable.")


with DAG(
    dag_id="rain_australia_drift_check",
    description="Surveillance de la dérive : lit le rapport Evidently et décide d'un réentraînement",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="0 6 * * *",       # tous les jours à 6 h
    catchup=False,
    tags=["mlops", "drift", "rain-australia"],
) as dag:

    lecture = PythonOperator(task_id="lire_resume", python_callable=lire_resume)

    branche = BranchPythonOperator(task_id="decider", python_callable=decider)

    alerte = PythonOperator(task_id="signaler_derive", python_callable=signaler_derive)
    rien = PythonOperator(task_id="rien_a_faire", python_callable=rien_a_faire)

    lecture >> branche >> [alerte, rien]
