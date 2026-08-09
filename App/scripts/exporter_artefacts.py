"""Génère les artefacts JSON consommés par le Streamlit de soutenance.

Le Streamlit tourne dans un conteneur qui n'a ni le CSV de 145 000 lignes, ni les
notebooks, ni scikit-learn. Tout ce qu'il affiche doit donc être figé ici, versionné,
et rechargeable sans rien recalculer : une démo qui dépend d'un entraînement à chaud
est une démo qui tombe le jour de l'oral.

    python App/scripts/exporter_artefacts.py        # ~1 min (la forêt aléatoire domine)

Sorties dans App/src/app/artefacts/ :
    eda.json          copie de Data/eda_stats.json (exploration, notebook 1)
    modeles.json      comparatif des quatre modèles + temps d'entraînement
    seuil.json        matrice de confusion pour 91 seuils de décision
    courbes.json      ROC et précision-rappel échantillonnées
    importances.json  variables les plus importantes (forêt aléatoire)
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # importe src.* depuis App/

from src import config  # noqa: E402
from src.data import build_preprocessor, get_feature_lists, load_dataset  # noqa: E402

SORTIE = config.APP_DIR / "src" / "app" / "artefacts"
EDA_SOURCE = config.PROJECT_ROOT / "Data" / "eda_stats.json"


def ecrire(nom, contenu):
    chemin = SORTIE / nom
    chemin.write_text(json.dumps(contenu, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {chemin.relative_to(config.PROJECT_ROOT)}  ({chemin.stat().st_size // 1024} ko)")


def preprocesseur(X, dense=False):
    """dense=True pour HistGradientBoosting, qui refuse les matrices creuses."""
    numeric, categorical = get_feature_lists(X)
    prep = build_preprocessor(numeric, categorical)
    if dense:
        prep.transformers[1][1].named_steps["onehot"].set_params(sparse_output=False)
    return prep


def echantillonner(x, y, n=200):
    """Réduit une courbe à n points en gardant les extrémités : le JSON reste léger."""
    if len(x) <= n:
        idx = np.arange(len(x))
    else:
        idx = np.unique(np.linspace(0, len(x) - 1, n).astype(int))
    return [round(float(v), 4) for v in np.asarray(x)[idx]], [round(float(v), 4) for v in np.asarray(y)[idx]]


def main():
    SORTIE.mkdir(parents=True, exist_ok=True)
    print("Chargement du dataset…")
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y
    )
    print(f"  train {X_train.shape[0]} lignes / test {X_test.shape[0]} lignes")

    modeles = {
        "Régression logistique": (LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE), False),
        "Régression logistique (balanced)": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE),
            False,
        ),
        "Forêt aléatoire": (
            RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=config.RANDOM_STATE),
            False,
        ),
        "Gradient boosting": (HistGradientBoostingClassifier(random_state=config.RANDOM_STATE), True),
    }

    lignes = []
    probas = {}
    pipelines = {}
    for nom, (clf, dense) in modeles.items():
        print(f"Entraînement — {nom}…")
        pipe = Pipeline([("prep", preprocesseur(X_train, dense=dense)), ("clf", clf)])
        depart = time.perf_counter()
        pipe.fit(X_train, y_train)
        duree = time.perf_counter() - depart
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        probas[nom] = proba
        pipelines[nom] = pipe
        lignes.append({
            "modele": nom,
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "rappel": round(float(recall_score(y_test, pred)), 4),
            "f1": round(float(f1_score(y_test, pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "secondes_entrainement": round(duree, 1),
        })
        print(f"  accuracy {lignes[-1]['accuracy']}  rappel {lignes[-1]['rappel']}  en {duree:.1f} s")

    # Baseline "toujours Non" : le point de comparaison qui rend l'accuracy suspecte.
    baseline = {
        "modele": "Baseline « toujours Non »",
        "accuracy": round(float(accuracy_score(y_test, np.zeros_like(y_test))), 4),
        "precision": 0.0, "rappel": 0.0, "f1": 0.0, "roc_auc": None,
        "secondes_entrainement": 0.0,
    }
    ecrire("modeles.json", {
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "taux_pluie_test": round(float(y_test.mean()), 4),
        "seuil_evalue": 0.5,
        "lignes": [baseline, *lignes],
    })

    # --- Effet du seuil, modèle de production constant -------------------------------
    # On exporte les quatre cases de la matrice de confusion, pas seulement les scores :
    # « 1 300 jours de pluie manqués au lieu de 2 900 » parle mieux qu'un rappel de 0,71.
    reference = "Gradient boosting"
    proba = probas[reference]
    grille = []
    for seuil in np.round(np.arange(0.05, 0.96, 0.01), 2):
        pred = (proba >= seuil).astype(int)
        vp = int(((pred == 1) & (y_test == 1)).sum())
        fp = int(((pred == 1) & (y_test == 0)).sum())
        vn = int(((pred == 0) & (y_test == 0)).sum())
        fn = int(((pred == 0) & (y_test == 1)).sum())
        grille.append({
            "seuil": float(seuil), "vp": vp, "fp": fp, "vn": vn, "fn": fn,
            "precision": round(vp / (vp + fp), 4) if vp + fp else 0.0,
            "rappel": round(vp / (vp + fn), 4) if vp + fn else 0.0,
            "f1": round(2 * vp / (2 * vp + fp + fn), 4) if vp else 0.0,
            "accuracy": round((vp + vn) / len(y_test), 4),
        })
    meilleur_f1 = max(grille, key=lambda g: g["f1"])
    rappel70 = min((g for g in grille if g["rappel"] >= 0.70), key=lambda g: abs(g["seuil"] - 0.5), default=None)
    ecrire("seuil.json", {
        "modele": reference,
        "seuil_production": config.DECISION_THRESHOLD,
        "seuil_f1_max": meilleur_f1["seuil"],
        "seuil_rappel_70": rappel70["seuil"] if rappel70 else None,
        "grille": grille,
    })

    # --- Courbes ROC / précision-rappel ---------------------------------------------
    courbes = {"taux_pluie_test": round(float(y_test.mean()), 4), "modeles": {}}
    for nom in ["Régression logistique", "Forêt aléatoire", "Gradient boosting"]:
        fpr, tpr, _ = roc_curve(y_test, probas[nom])
        prec, rec, _ = precision_recall_curve(y_test, probas[nom])
        fpr_e, tpr_e = echantillonner(fpr, tpr)
        rec_e, prec_e = echantillonner(rec, prec)
        courbes["modeles"][nom] = {
            "auc": round(float(roc_auc_score(y_test, probas[nom])), 4),
            "ap": round(float(average_precision_score(y_test, probas[nom])), 4),
            "roc": {"fpr": fpr_e, "tpr": tpr_e},
            "pr": {"rappel": rec_e, "precision": prec_e},
        }
    ecrire("courbes.json", courbes)

    # --- Importances de la forêt aléatoire ------------------------------------------
    pipe_rf = pipelines["Forêt aléatoire"]
    noms = pipe_rf.named_steps["prep"].get_feature_names_out()
    poids = pipe_rf.named_steps["clf"].feature_importances_
    ordre = np.argsort(poids)[::-1][:15]
    ecrire("importances.json", {
        "modele": "Forêt aléatoire",
        "variables": [
            {"variable": str(noms[i]).split("__", 1)[-1], "importance": round(float(poids[i]), 4)}
            for i in ordre
        ],
    })

    # --- Exploration : on recopie la sortie de Data/eda_explore.py -------------------
    if EDA_SOURCE.exists():
        shutil.copyfile(EDA_SOURCE, SORTIE / "eda.json")
        print(f"  {(SORTIE / 'eda.json').relative_to(config.PROJECT_ROOT)}  (copie de Data/eda_stats.json)")
    else:
        print(f"  ATTENTION : {EDA_SOURCE} absent, lancer d'abord python Data/eda_explore.py")

    print("\nTerminé. Versionner App/src/app/artefacts/ : le conteneur Streamlit les embarque.")


if __name__ == "__main__":
    main()
