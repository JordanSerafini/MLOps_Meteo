"""Accès aux artefacts figés affichés par le Streamlit.

Les fichiers de ce dossier sont produits par `python App/scripts/exporter_artefacts.py`
et versionnés. Le conteneur Streamlit n'a ni scikit-learn ni le CSV : il lit, il n'entraîne
pas. Un artefact manquant n'est pas une erreur fatale — l'onglet le signale et le reste de
l'application continue de fonctionner.
"""
import json
from pathlib import Path

import streamlit as st

DOSSIER = Path(__file__).resolve().parent / "artefacts"
CAPTURES = DOSSIER / "captures"


@st.cache_data(show_spinner=False)
def charger(nom):
    """Retourne le contenu de artefacts/<nom>.json, ou None si le fichier est absent."""
    chemin = DOSSIER / f"{nom}.json"
    if not chemin.exists():
        return None
    return json.loads(chemin.read_text(encoding="utf-8"))


def capture(nom):
    """Chemin de artefacts/captures/<nom>.png, ou None si la capture est absente.

    Les captures sont des relevés d'interfaces qui ne tournent pas en production sur la VM
    (Grafana, Prometheus, Alertmanager appartiennent au profil `monitoring`, lancé en local).
    Elles sont figées et versionnées pour la même raison que les JSON : une démonstration qui
    dépend d'un service à démarrer est une démonstration qui tombe le jour de l'oral.
    """
    chemin = CAPTURES / f"{nom}.png"
    return str(chemin) if chemin.exists() else None


def signaler_absent(nom):
    """Message d'erreur actionnable plutôt qu'une exception au milieu de l'oral."""
    st.warning(
        f"Artefact `{nom}.json` absent. Le régénérer avec :\n\n"
        "```bash\nmake artefacts\n```"
    )
