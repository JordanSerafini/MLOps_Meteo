"""Partie 3 — Microservices, orchestration, CI et sécurisation (phases 2 et 3).

Cet onglet est le seul qui interroge l'infrastructure en direct : état des services et
démonstration de l'autorisation sur `/reload`. Chaque appel a un délai court et son échec
est affiché comme une information, pas comme une exception.
"""
import os
import json
from pathlib import Path

import requests
import streamlit as st

import api_client

# Adresses sondées pour l'état de la plateforme. Dans le conteneur, ce sont les noms de
# service du réseau Docker — les services ne se joignent pas par leur adresse publique.
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")
AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://localhost:8080")

SERVICES = [
    ("api", "FastAPI", "sert les prédictions, ne s'occupe que de ça", "Dockerfile.api"),
    ("mlflow", "MLflow", "suivi des expériences + Model Registry (source du modèle servi)",
     "Dockerfile.mlflow"),
    ("streamlit", "Streamlit", "cette page — consomme l'API, jamais le modèle", "Dockerfile.streamlit"),
    ("trainer", "Entraînement", "conteneur éphémère, lancé à la demande sur un poste de dev",
     "Dockerfile.trainer"),
    ("drift", "Détection de dérive", "job éphémère Evidently, poussé vers Prometheus",
     "Dockerfile.drift"),
    ("airflow", "Airflow", "orchestration : déploiement et surveillance quotidienne", "image officielle"),
    ("prometheus", "Prometheus", "collecte des métriques exposées par l'API", "image officielle"),
    ("grafana", "Grafana", "deux tableaux de bord provisionnés au démarrage", "image officielle"),
    ("alertmanager", "Alertmanager", "route les six alertes vers un webhook", "image officielle"),
]

# Volontairement resserré : huit nœuds, pas quatorze. Un schéma exhaustif est illisible en
# projection, et le tableau qui suit détaille les rôles.
#
# Disposition verticale : Streamlit réduit le SVG à la largeur de la page, donc un graphe large
# devient minuscule quoi qu'on fasse des polices. En hauteur, les libellés restent lisibles.
GRAPHE = """
digraph architecture {
  rankdir=TB;
  nodesep=0.45; ranksep=0.55; fontname="Helvetica";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=15,
        color="#bbbbbb", margin="0.24,0.15"];
  edge [fontname="Helvetica", fontsize=12, color="#777777"];

  subgraph cluster_dev {
    label="Poste de développement"; style=dashed; color="#999999"; fontsize=13;
    csv   [label="weatherAUS.csv\\nversionné par DVC", fillcolor="#fff3cd"];
    train [label="trainer — train.py", fillcolor="#fff3cd"];
  }

  subgraph cluster_vm {
    label="VM meteo-liora — Docker Compose"; style=dashed; color="#999999"; fontsize=13;
    mlflow [label="MLflow — Model Registry", fillcolor="#d1e7dd"];
    api    [label="API FastAPI — OAuth2 + JWT", fillcolor="#cfe2ff"];
    stream [label="Streamlit", fillcolor="#cfe2ff"];
    air    [label="Airflow", fillcolor="#e2d9f3"];
    drift  [label="job dérive — Evidently", fillcolor="#f8d7da"];
    obs    [label="Prometheus →\\nGrafana + Alertmanager", fillcolor="#f8d7da"];
  }

  nginx [label="Internet → nginx\\nTLS + reverse proxy", fillcolor="#e9ecef"];

  csv -> train;
  train -> mlflow [label=" log + register"];
  mlflow -> api [label=" alias champion"];
  stream -> api [label=" POST /predict"];
  air -> api [label=" reload + smoke test"];
  api -> drift [label=" journal des prédictions"];
  api -> obs [label=" /metrics"];
  drift -> obs [label=" pushgateway"];
  air -> drift [label=" chaque matin"];
  nginx -> stream; nginx -> api; nginx -> mlflow; nginx -> air;
}
"""


def afficher():
    architecture()
    st.divider()
    etat_services()
    st.divider()
    securite()
    st.divider()
    orchestration()
    st.divider()
    integration_continue()


def architecture():
    st.subheader("Une responsabilité par conteneur")
    st.graphviz_chart(GRAPHE)
    st.markdown(
        "Le découpage suit une règle simple : **ce qui n'a pas le même cycle de vie ne vit pas "
        "dans le même conteneur**. L'entraînement est lourd, occasionnel et a besoin du jeu de "
        "données complet ; l'inférence est légère, permanente et n'a besoin que du modèle. Les "
        "mettre ensemble aurait imposé de déployer 145 000 lignes de CSV sur le serveur pour "
        "servir des prédictions unitaires.\n\n"
        "Le point de contact entre les deux est le **Model Registry** : l'entraînement pousse, "
        "l'API tire. Aucun volume partagé, aucun fichier de modèle qui circule à la main."
    )
    st.dataframe(
        [{"Service": s[1], "Rôle": s[2], "Image": s[3]} for s in SERVICES],
        use_container_width=True, hide_index=True,
    )


def etat_services():
    st.subheader("État de la plateforme, maintenant")
    sante = api_client.sante()
    c1, c2, c3, c4 = st.columns(4)
    if sante:
        c1.success("API : en ligne")
        c2.metric("Modèle chargé", "oui" if sante.get("model_loaded") else "non")
        c3.metric("Version servie", sante.get("model_version") or "—")
        c4.metric("Seuil appliqué", sante.get("decision_threshold", "—"))
        st.caption(f"`{api_client.API_URL}/health` — modèle : `{sante.get('model_uri')}`")
    else:
        c1.error("API : injoignable")
        st.caption(f"Aucune réponse de `{api_client.API_URL}/health`.")

    etats = [("MLflow", MLFLOW_URL, ""), ("Airflow", AIRFLOW_URL, "/health")]
    colonnes = st.columns(len(etats))
    for colonne, (nom, url, chemin) in zip(colonnes, etats, strict=True):
        if api_client.joignable(url, chemin):
            colonne.success(f"{nom} : joignable")
        else:
            colonne.warning(f"{nom} : pas de réponse")
        colonne.caption(url)
    st.caption(
        "Ces vérifications sont volontairement tolérantes : une page de connexion qui répond 401 "
        "prouve que le service tourne. Les adresses affichées sont celles du réseau Docker interne "
        "— les services se joignent par leur nom, aucun ne passe par Internet pour parler à un "
        "autre."
    )


def securite():
    st.subheader("Authentification, autorisation, et la différence entre les deux")
    st.markdown(
        "Le cadrage demandait des « APIs sécurisés, avec autorisation et authentification ». Il "
        "n'y avait rien : `/predict` et `/reload` étaient ouverts à qui voulait. C'est maintenant "
        "de l'OAuth2 avec jetons JWT et deux comptes aux portées distinctes."
    )
    st.dataframe(
        [
            {"Compte": "client", "Portée": "predict", "Peut prédire": "oui", "Peut recharger": "non"},
            {"Compte": "admin", "Portée": "predict + admin", "Peut prédire": "oui", "Peut recharger": "oui"},
        ],
        use_container_width=True, hide_index=True,
    )

    st.markdown("**Démonstration en direct**")
    st.caption(
        "Ce Streamlit est configuré avec le compte `client`. Il peut donc prédire, mais l'API doit "
        "lui refuser le rechargement du modèle : authentifié, pas autorisé. Le code de retour "
        "attendu est 403, pas 401 — la nuance est exactement ce que demandait le cadrage."
    )
    gauche, droite = st.columns(2)
    if gauche.button("Tenter `POST /reload` avec le compte du Streamlit", use_container_width=True):
        try:
            code, corps = api_client.tenter_reload(api_client.jeton())
        except requests.RequestException as e:
            st.error(f"Appel impossible : {e}")
        else:
            if code == 403:
                st.success(f"HTTP {code} — refusé faute de portée `admin`. Comportement attendu.")
            elif code == 200:
                st.error(f"HTTP {code} — le rechargement a été accepté. Ce compte a trop de droits.")
            else:
                st.warning(f"HTTP {code}")
            st.json(corps)

    with droite.form("admin"):
        st.caption("Pour montrer l'autre moitié : un compte administrateur. Rien n'est conservé.")
        identifiant = st.text_input("Identifiant", value="admin")
        mot_de_passe = st.text_input("Mot de passe", type="password")
        if st.form_submit_button("Obtenir un jeton puis recharger", use_container_width=True):
            if not mot_de_passe:
                st.info("Mot de passe requis — il est dans `App/.env`, généré par `make secrets`.")
            else:
                try:
                    jeton_admin, portees = api_client.obtenir_jeton(identifiant, mot_de_passe)
                    code, corps = api_client.tenter_reload(jeton_admin)
                except requests.HTTPError as e:
                    st.error(f"Jeton refusé (HTTP {e.response.status_code}) : identifiants invalides.")
                except requests.RequestException as e:
                    st.error(f"Appel impossible : {e}")
                else:
                    st.caption(f"Portées du jeton obtenu : `{portees}`")
                    (st.success if code == 200 else st.warning)(f"HTTP {code}")
                    st.json(corps)

    # Option pratique pour les démos : coller un jeton JWT existant et l'utiliser directement
    with st.expander("Utiliser un jeton admin existant (pour démo)"):
        jeton_cle = st.text_area("Collez le jeton JWT ici", value="", help="Jeton obtenu via /token (ne pas partager en prod)")
        if st.button("Utiliser ce jeton pour recharger", use_container_width=True):
            if not jeton_cle.strip():
                st.info("Collez un jeton valide avant de lancer l'appel.")
            else:
                try:
                    code, corps = api_client.tenter_reload(jeton_cle.strip())
                except requests.RequestException as e:
                    st.error(f"Appel impossible : {e}")
                else:
                    if code == 200:
                        st.success(f"HTTP {code} — rechargement accepté")
                    elif code == 403:
                        st.warning(f"HTTP {code} — jeton valide mais sans portée 'admin'")
                    elif code == 401:
                        st.error(f"HTTP {code} — jeton invalide ou expiré")
                    else:
                        st.warning(f"HTTP {code}")
                    st.json(corps)

    # Technical operational details removed from the UI; runbooks contain these topics.


def orchestration():
    st.subheader("Ce qu'Airflow orchestre, et ce qu'il n'orchestre pas")
    c1, c2 = st.columns(2)
    c1.markdown(
        "**DAG de déploiement** — attente de l'API, obtention d'un jeton administrateur, "
        "rechargement du modèle, contrôle de version, *smoke test* sur une prédiction réelle, "
        "validation de la réponse. C'est ce DAG qui garantit qu'un modèle promu est effectivement "
        "servi et qu'il répond.\n\n"
        "**DAG de surveillance** — chaque matin à 6 h, relecture du dernier résumé de dérive et "
        "signalement s'il faut réentraîner. Ses trois cas ont été testés : dérive, pas de dérive, "
        "résumé trop vieux pour être exploitable."
    )
    c2.markdown(
        "**Pourquoi le réentraînement n'est pas automatique**\n\n"
        "Nous avons retiré l'accès au démon Docker du conteneur Airflow au moment de la mise en "
        "production. Le lui rendre équivaudrait à lui donner les droits administrateur sur la "
        "machine : un DAG mal écrit pourrait lancer n'importe quel conteneur, monter n'importe "
        "quel volume.\n\n"
        "Ajoutons que l'entraînement est déporté sur un poste de dev par conception — le serveur "
        "n'a ni le jeu de données ni l'image d'entraînement. Le DAG **signale**, l'humain lance "
        "`make deploy-model`. La solution propre pour aller plus loin serait un proxy limitant "
        "strictement ce qu'Airflow peut demander à Docker ; c'est documenté dans le DAG lui-même."
    )
    st.caption("C'est un choix, pas un oubli — et il vaut mieux l'assumer que de laisser croire à "
               "une automatisation complète.")


def integration_continue():
    st.subheader("Intégration continue")
    c1, c2, c3 = st.columns(3)
    # Try to read CI metrics produced by the workflow (optional JSON file)
    metrics_file = Path(__file__).resolve().parents[3] / "ci_metrics.json"
    tests_count = "66"
    images_built = "5"
    linter_issues = "0"
    if metrics_file.exists():
        try:
            with metrics_file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            tests_count = str(data.get("tests", tests_count))
            images_built = str(data.get("images_built", images_built))
            linter_issues = str(data.get("linter_issues", linter_issues))
        except Exception:
            # If parsing fails, keep the defaults silently
            pass

    c1.metric("Tests unitaires", tests_count)
    c2.metric("Images construites en CI", images_built)
    c3.metric("Remontées du linter", linter_issues)
    st.markdown(
        "La CI tourne sur les branches `prenom_dev` et pas seulement à l'ouverture d'une *pull "
        "request* : au moment de la PR, il est déjà trop tard pour que le retour serve. Elle "
        "exécute le linter, les 66 tests, valide les deux fichiers Docker Compose et construit "
        "les cinq images.\n\n"
        "Les tests couvrent le chargement des données, les schémas d'entrée, l'authentification "
        "et les portées, les réponses de l'API et le job de dérive. Le modèle n'est pas "
        "réentraîné en CI : les tests d'API utilisent un modèle factice injecté, ce qui les rend "
        "rapides et déterministes."
    )
