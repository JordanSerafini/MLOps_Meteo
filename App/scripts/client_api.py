"""Accès à l'API authentifiée, partagé par les scripts du dossier."""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

URL_DEFAUT = os.getenv("API_URL", "http://localhost:8000")


def obtenir_jeton(url=URL_DEFAUT, role="client", timeout=15):
    identifiant = os.getenv(f"API_{role.upper()}_USER", role)
    mot_de_passe = os.getenv(f"API_{role.upper()}_PASSWORD", "")
    if not mot_de_passe:
        sys.exit(f"API_{role.upper()}_PASSWORD absent. Lancer `make secrets` puis recharger App/.env.")

    donnees = urllib.parse.urlencode({"username": identifiant, "password": mot_de_passe}).encode()
    requete = urllib.request.Request(f"{url}/token", data=donnees, method="POST")
    try:
        with urllib.request.urlopen(requete, timeout=timeout) as r:
            return json.loads(r.read())["access_token"]
    except urllib.error.HTTPError as e:
        if e.code == 401:
            sys.exit(f"Identifiants '{identifiant}' refusés par l'API.")
        raise
    except urllib.error.URLError as e:
        sys.exit(f"API injoignable sur {url} : {e.reason}")


def appel(chemin, methode="GET", corps=None, jeton=None, url=URL_DEFAUT, timeout=30):
    entetes = {}
    if jeton:
        entetes["Authorization"] = f"Bearer {jeton}"
    donnees = None
    if corps is not None:
        donnees = json.dumps(corps).encode()
        entetes["Content-Type"] = "application/json"

    requete = urllib.request.Request(f"{url}{chemin}", data=donnees,
                                     headers=entetes, method=methode)
    try:
        with urllib.request.urlopen(requete, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"{methode} {chemin} -> {e.code} {e.read().decode()[:200]}")
    except urllib.error.URLError as e:
        sys.exit(f"API injoignable sur {url} : {e.reason}")
