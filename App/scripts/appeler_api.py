"""Appels à l'API depuis la ligne de commande, avec authentification.

Usage:
    python scripts/appeler_api.py health
    python scripts/appeler_api.py predict
    python scripts/appeler_api.py reload
"""
import json
import sys

from client_api import appel, obtenir_jeton

EXEMPLE = {
    "Location": "Sydney", "Month": 7, "RainToday": "Yes",
    "Humidity3pm": 80, "Sunshine": 3.5, "Pressure3pm": 1008.0,
    "Rainfall": 12.0, "WindGustSpeed": 56, "Cloud3pm": 8, "Temp3pm": 16.0,
}


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "health"

    if action == "health":
        resultat = appel("/health")
    elif action == "predict":
        resultat = appel("/predict", "POST", EXEMPLE, obtenir_jeton(role="client"))
    elif action == "reload":
        resultat = appel("/reload", "POST", jeton=obtenir_jeton(role="admin"))
    else:
        sys.exit(f"Action inconnue : {action} (health, predict ou reload)")

    print(json.dumps(resultat, ensure_ascii=False))


if __name__ == "__main__":
    main()
