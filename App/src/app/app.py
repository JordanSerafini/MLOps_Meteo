"""Streamlit de démonstration et de soutenance — Rain in Australia.

Cinq onglets : la démonstration, puis une partie par membre de l'équipe, calée sur les quatre
phases de la feuille de route. Le support de l'oral, c'est cette application : elle est vivante
(elle interroge l'API réelle) là où des diapositives seraient mortes.

Deux règles de construction, qui expliquent l'organisation du dossier :
  - rien n'est calculé ici. Les chiffres viennent d'artefacts JSON versionnés (`artefacts/`),
    la seule dépendance vivante est l'API d'inférence. Une démonstration qui entraîne un modèle
    à chaud est une démonstration qui tombe le jour de l'oral ;
  - un onglet par fichier dans `onglets/`, pour que quatre personnes puissent travailler leur
    partie sans se marcher dessus.

Lancement local : streamlit run App/src/app/app.py
"""
import streamlit as st

import api_client
from onglets import donnees, industrialisation, modelisation, monitoring, prediction

# "presentateur" peut rester vide : l'en-tête et le plan omettent alors la mention, plutôt que
# d'afficher un nom d'emprunt au vidéoprojecteur.
# La durée est indicative — 20 minutes à quatre, en gardant de la marge pour les questions.
PARTIES = [
    {
        "cle": "prediction",
        "titre": "Démonstration",
        "intitule": "Une prédiction de bout en bout",
        "presentateur": "toute l'équipe",
        "minutes": 2,
        "module": prediction,
        "resume": "Le produit fini : le formulaire appelle l'API, qui interroge le modèle promu "
                  "dans MLflow.",
    },
    {
        "cle": "donnees",
        "titre": "1 · Données",
        "intitule": "Exploration, cible déséquilibrée et préprocessing",
        "presentateur": "",
        "minutes": 4,
        "module": donnees,
        "resume": "145 460 relevés, 49 stations, 22 % de jours de pluie : pourquoi l'accuracy est "
                  "un mauvais juge.",
    },
    {
        "cle": "modelisation",
        "titre": "2 · Modélisation",
        "intitule": "Modèles, seuil de décision, MLflow et DVC",
        "presentateur": "",
        "minutes": 4,
        "module": modelisation,
        "resume": "Quatre modèles se tiennent en quatre points ; le seuil de décision en vaut "
                  "dix-sept.",
    },
    {
        "cle": "industrialisation",
        "titre": "3 · Industrialisation",
        "intitule": "Microservices, orchestration, CI et sécurisation",
        "presentateur": "",
        "minutes": 4,
        "module": industrialisation,
        "resume": "Neuf services, une responsabilité chacun, un reverse proxy et une API "
                  "authentifiée.",
    },
    {
        "cle": "monitoring",
        "titre": "4 · Monitoring",
        "intitule": "Métriques, détection de dérive et alertes",
        "presentateur": "Jordan",
        "minutes": 5,
        "module": monitoring,
        "resume": "Comment un détecteur de dérive mal calibré nous a annoncé 78 % de dérive sur "
                  "des données normales.",
    },
]

st.set_page_config(page_title="Rain in Australia — MLOps", page_icon="🌧️", layout="wide")


def barre_laterale():
    st.sidebar.title("🌧️ Rain in Australia")
    st.sidebar.caption("Projet MLOps — prédire la pluie du lendemain sur 49 stations australiennes")

    sante = api_client.sante()
    if sante is None:
        st.sidebar.error("API injoignable")
        st.sidebar.caption(f"`{api_client.API_URL}`")
    elif sante.get("model_loaded"):
        st.sidebar.success(f"API en ligne — modèle v{sante.get('model_version')}")
        st.sidebar.caption(f"seuil servi : {sante.get('decision_threshold')} · "
                           f"`{sante.get('model_uri')}`")
    else:
        st.sidebar.warning("API en ligne, aucun modèle chargé")
        st.sidebar.caption("Lancer un entraînement, puis `POST /reload`.")

    st.sidebar.divider()
    st.sidebar.markdown("**Déroulé de l'oral**")
    total = sum(p["minutes"] for p in PARTIES)
    debut = 0
    for partie in PARTIES:
        fin = debut + partie["minutes"]
        qui = f" · {partie['presentateur']}" if partie["presentateur"] else ""
        st.sidebar.markdown(f"`{debut:02d}–{fin:02d}`  **{partie['titre']}**{qui}")
        debut = fin
    st.sidebar.caption(f"{total} minutes de présentation, le reste pour les questions.")


def entete(partie):
    st.markdown(f"### {partie['intitule']}")
    qui = f"présenté par {partie['presentateur']} · " if partie["presentateur"] else ""
    st.caption(f"{partie['titre']} · {qui}≈ {partie['minutes']} min — {partie['resume']}")


def main():
    barre_laterale()
    st.title("Va-t-il pleuvoir demain en Australie ?")
    st.caption(
        "Modèle de classification servi par une API FastAPI, entraînement suivi dans MLflow, "
        "orchestration Airflow, monitoring Prometheus / Grafana et détection de dérive Evidently."
    )
    for onglet, partie in zip(st.tabs([p["titre"] for p in PARTIES]), PARTIES, strict=True):
        with onglet:
            entete(partie)
            st.divider()
            partie["module"].afficher()


main()
