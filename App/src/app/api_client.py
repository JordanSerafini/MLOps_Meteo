"""Client HTTP de l'API d'inférence.

Le Streamlit ne charge jamais le modèle : il passe par l'API, comme n'importe quel
consommateur. C'est ce qui permet de montrer en soutenance que le service est bien
découpé — et de démontrer l'autorisation, puisque le compte utilisé ici n'a que la
portée `predict`.
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_USER = os.getenv("API_USER", "client")
API_PASSWORD = os.getenv("API_PASSWORD", "")

DELAI = 10


def obtenir_jeton(identifiant, mot_de_passe):
    """Appel brut à POST /token. Retourne (jeton, portees) ou lève."""
    r = requests.post(f"{API_URL}/token",
                      data={"username": identifiant, "password": mot_de_passe},
                      timeout=DELAI)
    r.raise_for_status()
    charge = r.json()
    return charge["access_token"], charge.get("scope", "")


def jeton(rafraichir=False):
    """Jeton du compte de service, gardé en session ; renouvelé si l'API l'a rejeté."""
    if rafraichir:
        st.session_state.pop("jeton", None)
    if "jeton" not in st.session_state:
        st.session_state["jeton"], st.session_state["portees"] = obtenir_jeton(API_USER, API_PASSWORD)
    return st.session_state["jeton"]


def predire(charge):
    """POST /predict avec réessai unique si le jeton a expiré pendant la session."""
    r = requests.post(f"{API_URL}/predict", json=charge, timeout=DELAI,
                      headers={"Authorization": f"Bearer {jeton()}"})
    if r.status_code == 401:
        r = requests.post(f"{API_URL}/predict", json=charge, timeout=DELAI,
                          headers={"Authorization": f"Bearer {jeton(rafraichir=True)}"})
    r.raise_for_status()
    return r.json()


def tenter_reload(jeton_utilise):
    """POST /reload sans lever : c'est le code de retour qui nous intéresse (403 attendu
    avec un jeton `client`, 200 avec un jeton `admin`)."""
    r = requests.post(f"{API_URL}/reload", timeout=DELAI,
                      headers={"Authorization": f"Bearer {jeton_utilise}"})
    try:
        corps = r.json()
    except ValueError:
        corps = {"detail": r.text[:200]}
    return r.status_code, corps


@st.cache_data(ttl=10, show_spinner=False)
def sante():
    """État de l'API et du modèle servi. Retourne None si l'API est injoignable."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


@st.cache_data(ttl=30, show_spinner=False)
def joignable(url, chemin=""):
    """Teste la présence d'un service sans juger son code de retour : une page de
    connexion répond 401 ou 302, le service est debout quand même."""
    try:
        r = requests.get(f"{url}{chemin}", timeout=3, allow_redirects=False)
        return r.status_code < 500
    except requests.RequestException:
        return False
