"""Entraînement + tracking MLflow + enregistrement au Model Registry.

Usage:
    python -m src.train --model rf --register
    python -m src.train --model logreg

Logue params/metrics/artefact dans MLflow ; si --register, enregistre une nouvelle
version du modèle et pose l'alias (par défaut 'champion') que l'API charge.
"""
import argparse

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from . import config
from .data import build_preprocessor, get_feature_lists, load_dataset

MODELS = {
    "logreg": lambda: LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
    "logreg_balanced": lambda: LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE),
    "rf": lambda: RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=config.RANDOM_STATE),
}


def parse_args():
    p = argparse.ArgumentParser(description="Entraîne et trace un modèle de prévision de pluie")
    p.add_argument("--model", choices=list(MODELS), default="rf")
    p.add_argument("--register", action="store_true", help="enregistre + alias dans le Model Registry")
    p.add_argument("--alias", default=config.MODEL_ALIAS)
    return p.parse_args()


def main():
    args = parse_args()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    X, y = load_dataset()
    numeric, categorical = get_feature_lists(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y)

    pipe = Pipeline([
        ("prep", build_preprocessor(numeric, categorical)),
        ("clf", MODELS[args.model]()),
    ])

    with mlflow.start_run(run_name=args.model) as run:
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "recall_rain": recall_score(y_test, pred),
            "f1_rain": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba),
        }
        mlflow.log_param("model_type", args.model)
        mlflow.log_param("n_features_in", X.shape[1])
        mlflow.log_param("n_train", X_train.shape[0])
        mlflow.log_metrics(metrics)

        print(f"\n=== {args.model} ===")
        print(classification_report(y_test, pred, target_names=["No", "Yes"]))
        print("metrics:", {k: round(v, 4) for k, v in metrics.items()})

        signature = infer_signature(X_test, pred)
        mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            signature=signature,
            input_example=X_test.head(2),
            registered_model_name=config.MODEL_NAME if args.register else None,
        )

        if args.register:
            client = MlflowClient()
            versions = client.search_model_versions(f"name='{config.MODEL_NAME}'")
            latest = str(max(int(v.version) for v in versions))  # l'API registry exige un str
            client.set_registered_model_alias(config.MODEL_NAME, args.alias, latest)
            print(f"Modèle '{config.MODEL_NAME}' v{latest} -> alias '@{args.alias}' "
                  f"(run {run.info.run_id})")


if __name__ == "__main__":
    main()
