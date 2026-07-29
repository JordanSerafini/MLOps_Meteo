"""Onglet de démonstration : une prédiction de bout en bout, servie par l'API.

C'est la vitrine. Rien n'est calculé ici : le formulaire construit le corps JSON du
contrat `/predict`, l'API répond, on affiche. Si l'écran plante, c'est l'API ou le
modèle qui est en cause, pas la page.
"""
import pandas as pd
import requests
import streamlit as st

import api_client
import artefacts
from formats import nb, pct

# Deux situations opposées, pour montrer que le modèle réagit à autre chose qu'au hasard.
# Les valeurs suivent les moyennes par classe mesurées au notebook 1 (class_means).
PRESETS = {
    "Journée humide et couverte": {
        "Location": "Portland", "Month": 7, "RainToday": "Yes", "Humidity3pm": 85,
        "Sunshine": 2.0, "Pressure3pm": 1006.0, "Rainfall": 14.0, "WindGustSpeed": 63,
        "Cloud3pm": 8, "Temp3pm": 14.0,
    },
    "Journée sèche et ensoleillée": {
        "Location": "Woomera", "Month": 1, "RainToday": "No", "Humidity3pm": 25,
        "Sunshine": 11.0, "Pressure3pm": 1019.0, "Rainfall": 0.0, "WindGustSpeed": 31,
        "Cloud3pm": 1, "Temp3pm": 33.0,
    },
    "Cas limite": {
        "Location": "Sydney", "Month": 4, "RainToday": "No", "Humidity3pm": 62,
        "Sunshine": 6.0, "Pressure3pm": 1013.0, "Rainfall": 1.0, "WindGustSpeed": 44,
        "Cloud3pm": 5, "Temp3pm": 22.0,
    },
}


def stations():
    """Liste des stations connues du modèle, tirée de l'exploration."""
    eda = artefacts.charger("eda")
    if not eda:
        return ["Sydney", "Portland", "Woomera"]
    return sorted(eda["geography"]["rain_rate_by_location_pct"])


def afficher():
    st.subheader("Prédire la pluie de demain")
    st.caption(
        "Le formulaire envoie un `POST /predict` authentifié à l'API, qui interroge le modèle "
        "chargé depuis le Model Registry MLflow. Streamlit ne voit jamais le modèle."
    )

    liste = stations()
    choix = st.radio("Situation de départ", list(PRESETS), horizontal=True,
                     help="Trois jeux de valeurs pour comparer les réponses du modèle.")
    base = PRESETS[choix]

    with st.form("predict"):
        c1, c2 = st.columns(2)
        index_station = liste.index(base["Location"]) if base["Location"] in liste else 0
        location = c1.selectbox("Station (Location)", liste, index=index_station)
        month = c2.slider("Mois", 1, 12, base["Month"])
        humidity3pm = c1.slider("Humidité 15h (%)", 0, 100, base["Humidity3pm"])
        sunshine = c2.slider("Ensoleillement (h)", 0.0, 14.5, base["Sunshine"])
        pressure3pm = c1.slider("Pression 15h (hPa)", 977.0, 1040.0, base["Pressure3pm"])
        rainfall = c2.number_input("Pluie aujourd'hui (mm)", 0.0, 400.0, base["Rainfall"])
        windgust = c1.slider("Rafale max (km/h)", 6, 135, base["WindGustSpeed"])
        cloud3pm = c2.slider("Nuages 15h (octas)", 0, 9, base["Cloud3pm"])
        temp3pm = c1.slider("Température 15h (°C)", -5.0, 47.0, base["Temp3pm"])
        raintoday = c2.selectbox("A-t-il plu aujourd'hui ?", ["Yes", "No"],
                                 index=0 if base["RainToday"] == "Yes" else 1)
        envoye = st.form_submit_button("Prédire", type="primary")

    if not envoye:
        st.info("Les seize autres variables du dataset sont absentes du formulaire : "
                "le pipeline les impute avec les statistiques mémorisées à l'entraînement.")
        return

    charge = {
        "Location": location, "Month": month, "Humidity3pm": humidity3pm,
        "Sunshine": sunshine, "Pressure3pm": pressure3pm, "Rainfall": rainfall,
        "WindGustSpeed": windgust, "Cloud3pm": cloud3pm, "Temp3pm": temp3pm,
        "RainToday": raintoday,
    }
    try:
        reponse = api_client.predire(charge)
    except requests.RequestException as e:
        st.error(f"Erreur de prédiction : {e}")
        st.caption(f"API interrogée : `{api_client.API_URL}`")
        return

    proba = reponse["probability"]
    seuil = reponse.get("threshold", 0.5)
    resultat(proba, seuil, reponse)
    comparaison_seuil(proba, seuil)
    with st.expander("Le corps de la requête envoyée à l'API"):
        st.json(charge)
    historique(charge, proba, reponse)


def resultat(proba, seuil, reponse):
    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilité de pluie demain", pct(proba))
    c2.metric("Seuil de décision appliqué", nb(seuil))
    c3.metric("Version du modèle", reponse.get("model_version") or "—")
    st.progress(min(proba, 1.0))
    if reponse["rain_tomorrow"]:
        st.success("Pluie annoncée demain — la probabilité dépasse le seuil.")
    else:
        st.info("Pas de pluie annoncée — la probabilité reste sous le seuil.")


def comparaison_seuil(proba, seuil_actuel):
    """Le même modèle, la même probabilité, deux décisions selon le seuil.

    C'est le point de bascule vers l'onglet modélisation : la décision n'est pas dans
    le modèle, elle est dans le seuil.
    """
    donnees = artefacts.charger("seuil")
    if not donnees:
        return
    alternatif = donnees.get("seuil_f1_max")
    if alternatif is None or abs(alternatif - seuil_actuel) < 1e-9:
        return
    if (proba >= seuil_actuel) != (proba >= alternatif):
        decision = "pluie" if proba >= alternatif else "pas de pluie"
        st.warning(
            f"Avec le seuil recommandé par notre analyse ({nb(alternatif)} au lieu de "
            f"{nb(seuil_actuel)}), cette même probabilité donnerait **{decision}**. "
            "Le détail est dans l'onglet Modélisation."
        )


def historique(charge, proba, reponse):
    """Chaque prédiction est aussi une ligne du journal côté API : c'est ce journal qui
    alimente la détection de dérive présentée au dernier onglet."""
    lignes = st.session_state.setdefault("historique", [])
    lignes.append({
        "Station": charge["Location"], "Mois": charge["Month"],
        "Humidité 15h": charge["Humidity3pm"], "Probabilité": round(proba, 4),
        "Décision": "pluie" if reponse["rain_tomorrow"] else "sec",
    })
    if len(lignes) > 1:
        st.markdown("**Prédictions de cette session**")
        st.dataframe(pd.DataFrame(lignes[::-1]), use_container_width=True, hide_index=True)
        st.caption("Côté API, chacune de ces requêtes a produit une ligne JSON dans le journal "
                   "des prédictions — le jeu de données « courant » du job de dérive.")
