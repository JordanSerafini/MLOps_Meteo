"""Envoie des prédictions à l'API pour alimenter le journal.

Sert à deux choses : vérifier que la chaîne tourne, et provoquer une dérive à la demande
pour la démonstration. Les données étant arrêtées en 2017, on ne peut pas attendre qu'une
vraie dérive se produise — on rejoue donc les observations d'une station atypique.

Usage:
    python scripts/simuler_trafic.py --n 200
    python scripts/simuler_trafic.py --n 200 --station Portland   # provoque la dérive
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

CHAMPS = [
    "Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
    "WindGustDir", "WindGustSpeed", "WindDir9am", "WindDir3pm", "WindSpeed9am",
    "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
    "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm", "RainToday",
]


def charge(csv, station, n, graine):
    df = pd.read_csv(csv, na_values=["NA"])
    if station:
        df = df[df["Location"] == station]
        if df.empty:
            sys.exit(f"Station '{station}' absente du dataset")
    df = df.sample(min(n, len(df)), random_state=graine)
    df["Month"] = pd.to_datetime(df["Date"], errors="coerce").dt.month
    return df[[c for c in CHAMPS if c in df.columns] + ["Month"]]


def obtenir_jeton(url, identifiant, mot_de_passe, timeout):
    donnees = urllib.parse.urlencode({"username": identifiant, "password": mot_de_passe}).encode()
    requete = urllib.request.Request(f"{url}/token", data=donnees, method="POST")
    try:
        with urllib.request.urlopen(requete, timeout=timeout) as r:
            return json.loads(r.read())["access_token"]
    except urllib.error.HTTPError as e:
        if e.code == 401:
            sys.exit("Identifiants refusés. Vérifier API_CLIENT_PASSWORD dans App/.env.")
        raise


def envoie(url, ligne, jeton, timeout):
    charge_utile = {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                    for k, v in ligne.items()}
    requete = urllib.request.Request(
        f"{url}/predict",
        data=json.dumps(charge_utile).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {jeton}"},
        method="POST",
    )
    with urllib.request.urlopen(requete, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    p = argparse.ArgumentParser(description="Génère du trafic vers l'API d'inférence")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--csv", default=str(Path(__file__).resolve().parents[2] / "Data" / "weatherAUS.csv"))
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--station", default=None,
                   help="ne tirer que dans cette station (Portland = 37 %% de pluie, provoque la dérive)")
    p.add_argument("--graine", type=int, default=0)
    p.add_argument("--timeout", type=float, default=10)
    p.add_argument("--utilisateur", default=os.getenv("API_CLIENT_USER", "client"))
    p.add_argument("--mot-de-passe", default=os.getenv("API_CLIENT_PASSWORD", ""))
    args = p.parse_args()

    donnees = charge(args.csv, args.station, args.n, args.graine)
    jeton = obtenir_jeton(args.url, args.utilisateur, args.mot_de_passe, args.timeout)
    envoyees = pluie = 0

    for _, ligne in donnees.iterrows():
        try:
            reponse = envoie(args.url, ligne, jeton, args.timeout)
        except (urllib.error.URLError, TimeoutError) as e:
            sys.exit(f"Appel à {args.url} impossible : {e}")
        envoyees += 1
        pluie += bool(reponse["rain_tomorrow"])

    origine = args.station or "toutes stations"
    print(f"{envoyees} prédictions envoyées ({origine}) — "
          f"{pluie} annoncées pluvieuses, soit {100 * pluie / max(envoyees, 1):.0f} %")


if __name__ == "__main__":
    main()
