"""Streamlit — site de display / démo. Consomme l'API (jamais le modèle en direct)."""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Pluie en Australie", page_icon="🌧️")
st.title("🌧️ Va-t-il pleuvoir demain ?")
st.caption("Démo MLOps — modèle servi via l'API FastAPI (MLflow Model Registry)")

# État de l'API / du modèle
try:
    h = requests.get(f"{API_URL}/health", timeout=3).json()
    if h.get("model_loaded"):
        st.sidebar.success(f"API OK — modèle v{h.get('model_version')}")
    else:
        st.sidebar.warning("API en ligne mais modèle non chargé (lancer l'entraînement).")
    st.sidebar.caption(f"URI: {h.get('model_uri')}")
except Exception as e:  # noqa: BLE001
    st.sidebar.error(f"API injoignable : {e}")

with st.form("predict"):
    c1, c2 = st.columns(2)
    location = c1.text_input("Station (Location)", "Sydney")
    month = c2.slider("Mois", 1, 12, 7)
    humidity3pm = c1.slider("Humidité 15h (%)", 0, 100, 80)
    sunshine = c2.slider("Ensoleillement (h)", 0.0, 14.5, 3.5)
    pressure3pm = c1.slider("Pression 15h (hPa)", 977.0, 1040.0, 1008.0)
    rainfall = c2.number_input("Pluie aujourd'hui (mm)", 0.0, 400.0, 12.0)
    windgust = c1.slider("Rafale max (km/h)", 6, 135, 56)
    cloud3pm = c2.slider("Nuages 15h (octas)", 0, 9, 8)
    temp3pm = c1.slider("Température 15h (°C)", -5.0, 47.0, 16.0)
    raintoday = c2.selectbox("A-t-il plu aujourd'hui ?", ["Yes", "No"], index=0)
    submitted = st.form_submit_button("Prédire")

if submitted:
    payload = {
        "Location": location, "Month": month, "Humidity3pm": humidity3pm,
        "Sunshine": sunshine, "Pressure3pm": pressure3pm, "Rainfall": rainfall,
        "WindGustSpeed": windgust, "Cloud3pm": cloud3pm, "Temp3pm": temp3pm,
        "RainToday": raintoday,
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        out = r.json()
        st.metric("Probabilité de pluie demain", f"{out['probability'] * 100:.1f}%")
        if out["rain_tomorrow"]:
            st.success("☔ Pluie probable demain")
        else:
            st.info("🌤️ Probablement sec demain")
        st.caption(f"modèle v{out.get('model_version')}")
    except Exception as e:  # noqa: BLE001
        st.error(f"Erreur de prédiction : {e}")
