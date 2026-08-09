"""Détection de dérive entre les données d'entraînement et les prédictions servies.

Job batch : compare un échantillon de référence (tiré du dataset d'entraînement) aux
observations reçues par l'API, loguées en JSONL par `api/main.py`.

Produit trois choses :
  - un rapport HTML Evidently, horodaté, archivé dans REPORTS_DIR ;
  - un résumé JSON (dernier état), lu par le DAG Airflow pour décider d'un réentraînement ;
  - trois métriques poussées au Pushgateway, que Prometheus récupère pour Grafana.

Usage:
    python -m src.drift
    python -m src.drift --min-samples 50 --seuil 0.3
"""
import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from . import config
from .data import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("drift")

# Colonnes suivies : les plus prédictives d'après l'EDA, plus la station.
# Inutile de surveiller les 22 colonnes, on veut un signal lisible.
COLONNES_SUIVIES = [
    "Humidity3pm", "Humidity9am", "Sunshine", "Cloud3pm",
    "Pressure3pm", "Temp3pm", "Rainfall", "WindGustSpeed", "Location",
]


def charge_reference(chemin, n, graine):
    """Distribution de référence à laquelle on compare le trafic.

    On lit de préférence l'échantillon figé (`reference/reference.csv`, versionné) plutôt
    que de retirer au hasard dans le dataset à chaque exécution : une référence qui bouge
    d'un run à l'autre rend les comparaisons dans le temps inexploitables. Le repli sur le
    CSV complet sert au développement, où le fichier est disponible.
    """
    chemin = Path(chemin)
    if chemin.exists():
        log.info("Référence figée : %s", chemin)
        return pd.read_csv(chemin)

    log.warning("%s absent, repli sur un tirage dans le dataset complet", chemin)
    X, _ = load_dataset()
    colonnes = [c for c in COLONNES_SUIVIES if c in X.columns]
    return X[colonnes].sample(min(n, len(X)), random_state=graine)


def charge_courant(chemin, colonnes):
    """Relit le journal JSONL des prédictions et en extrait les features."""
    chemin = Path(chemin)
    if not chemin.exists():
        return pd.DataFrame(columns=colonnes)

    lignes = []
    with chemin.open(encoding="utf-8") as f:
        for i, brut in enumerate(f, start=1):
            brut = brut.strip()
            if not brut:
                continue
            try:
                lignes.append(json.loads(brut)["features"])
            except (json.JSONDecodeError, KeyError):
                log.warning("Ligne %d du journal illisible, ignorée", i)

    if not lignes:
        return pd.DataFrame(columns=colonnes)
    return pd.DataFrame(lignes).reindex(columns=colonnes)


def extrait_metriques(resultat):
    """Sort (nombre, part) de colonnes en dérive du dictionnaire renvoyé par Evidently."""
    for metrique in resultat.get("metrics", []):
        if metrique.get("metric_name", "").startswith("DriftedColumnsCount"):
            valeur = metrique["value"]
            return int(valeur["count"]), float(valeur["share"])
    raise RuntimeError("DriftedColumnsCount absent du rapport Evidently")


def pousse_vers_prometheus(url, derive, nb_colonnes, part, nb_obs):
    """Envoie les métriques au Pushgateway. Un échec ne fait pas échouer le job."""
    if not url:
        return
    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

        registre = CollectorRegistry()
        jauges = {
            "rain_dataset_drift": ("1 si le jeu de données a dérivé", float(derive)),
            "rain_n_drifted_columns": ("Nombre de colonnes en dérive", nb_colonnes),
            "rain_share_drifted_columns": ("Part de colonnes en dérive", part),
            "rain_drift_samples": ("Observations comparées", nb_obs),
            # Permet d'alerter quand le job ne tourne plus : sans ça, un job planté
            # laisse les dernières valeurs en place et tout paraît normal.
            "rain_drift_last_run_timestamp": ("Horodatage de la dernière exécution",
                                              datetime.now(UTC).timestamp()),
        }
        for nom, (aide, valeur) in jauges.items():
            Gauge(nom, aide, registry=registre).set(valeur)

        push_to_gateway(url, job="drift", registry=registre)
        log.info("Métriques poussées vers %s", url)
    except Exception as e:  # noqa: BLE001 - le monitoring ne doit pas bloquer le job
        log.warning("Push Prometheus impossible (%s) : %s", url, e)


def parse_args():
    p = argparse.ArgumentParser(description="Compare les prédictions servies aux données d'entraînement")
    p.add_argument("--journal", default=config.PREDICTION_LOG_PATH or "/app/logs/predictions.jsonl",
                   help="journal JSONL écrit par l'API")
    p.add_argument("--rapports", default="/app/reports", help="dossier des rapports HTML")
    p.add_argument("--reference", default=config.REFERENCE_PATH,
                   help="échantillon de référence figé (CSV)")
    p.add_argument("--pushgateway", default=config.PUSHGATEWAY_URL)
    p.add_argument("--min-samples", type=int, default=400,
                   help="en dessous, le bruit d'échantillonnage domine (cf. Docs/CALIBRATION_DRIFT.md)")
    p.add_argument("--taille-reference", type=int, default=5000)
    p.add_argument("--seuil", type=float, default=config.DRIFT_SHARE_THRESHOLD,
                   help="part de colonnes en dérive au-delà de laquelle on réentraîne")
    return p.parse_args()


def main():
    args = parse_args()
    from evidently import Report
    from evidently.presets import DataDriftPreset

    reference = charge_reference(args.reference, args.taille_reference, config.RANDOM_STATE)
    courant = charge_courant(args.journal, reference.columns)
    log.info("Référence : %d lignes | Courant : %d lignes", len(reference), len(courant))

    dossier = Path(args.rapports)
    dossier.mkdir(parents=True, exist_ok=True)

    if len(courant) < args.min_samples:
        # Pas assez de trafic : on le dit explicitement plutot que de renvoyer
        # un "pas de derive" qui serait interprete comme un feu vert.
        resume = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "statut": "donnees_insuffisantes",
            "n_courant": len(courant),
            "min_samples": args.min_samples,
            "retrain_recommande": False,
        }
        (dossier / "dernier_resume.json").write_text(json.dumps(resume, indent=2), encoding="utf-8")
        log.warning("Seulement %d observations (minimum %d) — pas de conclusion",
                    len(courant), args.min_samples)
        print(json.dumps(resume, indent=2))
        return

    rapport = Report([DataDriftPreset()], include_tests=True)
    resultat = rapport.run(reference_data=reference, current_data=courant)

    nb_colonnes, part = extrait_metriques(resultat.dict())
    derive = part >= args.seuil

    horodatage = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    html = dossier / f"drift-{horodatage}.html"
    resultat.save_html(str(html))

    resume = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "statut": "ok",
        "n_reference": len(reference),
        "n_courant": len(courant),
        "n_colonnes_derivees": nb_colonnes,
        "part_colonnes_derivees": round(part, 4),
        "seuil": args.seuil,
        "retrain_recommande": bool(derive),
        "rapport_html": html.name,
    }
    (dossier / "dernier_resume.json").write_text(json.dumps(resume, indent=2), encoding="utf-8")

    pousse_vers_prometheus(args.pushgateway, derive, nb_colonnes, part, len(courant))

    log.info("%d/%d colonnes en dérive (%.0f %%) — seuil %.0f %% — réentraînement %s",
             nb_colonnes, len(reference.columns), 100 * part, 100 * args.seuil,
             "recommandé" if derive else "non nécessaire")
    print(json.dumps(resume, indent=2))


if __name__ == "__main__":
    main()
