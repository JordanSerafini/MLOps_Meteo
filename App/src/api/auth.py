"""Authentification OAuth2 et jetons JWT.

Deux rôles : `client` peut demander des prédictions, `admin` peut en plus recharger le
modèle. Le jeton porte l'authentification, les portées OAuth2 portent l'autorisation.
Les comptes sont déclarés par variables d'environnement, mots de passe hachés.
"""
import secrets
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, SecurityScopes

from .. import config

PORTEES = {
    "predict": "Demander une prédiction",
    "admin": "Recharger le modèle servi",
}

LONGUEUR_HACHAGE_BCRYPT = 60

oauth2 = OAuth2PasswordBearer(tokenUrl="token", scopes=PORTEES, auto_error=False)


def hacher(mot_de_passe: str) -> str:
    return bcrypt.hashpw(mot_de_passe.encode(), bcrypt.gensalt()).decode()


def mot_de_passe_valide(propose: str, hache: str) -> bool:
    try:
        return bcrypt.checkpw(propose.encode(), hache.encode())
    except ValueError:
        return False


def comptes() -> dict[str, dict]:
    declares = {
        config.API_CLIENT_USER: (config.API_CLIENT_PASSWORD_HASH, ["predict"]),
        config.API_ADMIN_USER: (config.API_ADMIN_PASSWORD_HASH, ["predict", "admin"]),
    }
    return {
        nom: {"hachage": hachage, "portees": portees}
        for nom, (hachage, portees) in declares.items()
        if nom and hachage
    }


def verifier_configuration():
    """Appelée au démarrage : mieux vaut ne pas démarrer que servir sans protection."""
    if not config.JWT_SECRET_KEY or config.JWT_SECRET_KEY == "CHANGEME":
        raise RuntimeError(
            "JWT_SECRET_KEY absent ou laissé à sa valeur d'exemple. "
            "Lancer `make secrets`."
        )

    declares = comptes()
    if not declares:
        raise RuntimeError("Aucun compte configuré. Lancer `make secrets`.")

    for nom, compte in declares.items():
        if len(compte["hachage"]) != LONGUEUR_HACHAGE_BCRYPT:
            # Cas classique : un $ non doublé dans .env.api, mangé par l'interpolation
            # de compose. Le hachage arrive tronqué et tous les mots de passe sont
            # refusés, sans message. Autant refuser de démarrer.
            raise RuntimeError(
                f"Hachage du compte '{nom}' long de {len(compte['hachage'])} caractères "
                f"au lieu de {LONGUEUR_HACHAGE_BCRYPT}. Les $ du hachage doivent être "
                "doublés dans .env.api — `make secrets` s'en charge."
            )


def authentifier(identifiant: str, mot_de_passe: str) -> dict | None:
    compte = comptes().get(identifiant)
    if compte is None:
        # Hachage à vide : sans ça le temps de réponse trahit l'existence du compte.
        bcrypt.checkpw(b"x", bcrypt.hashpw(b"x", bcrypt.gensalt()))
        return None
    if not mot_de_passe_valide(mot_de_passe, compte["hachage"]):
        return None
    return {"identifiant": identifiant, "portees": compte["portees"]}


def creer_jeton(identifiant: str, portees: list[str]) -> str:
    maintenant = datetime.now(UTC)
    charge = {
        "sub": identifiant,
        "scopes": portees,
        "iat": maintenant,
        "exp": maintenant + timedelta(minutes=config.JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(charge, config.JWT_SECRET_KEY, algorithm=config.JWT_ALGORITHM)


def _refus(detail, portees, code=status.HTTP_401_UNAUTHORIZED):
    entete = f'Bearer scope="{portees.scope_str}"' if portees.scopes else "Bearer"
    return HTTPException(status_code=code, detail=detail, headers={"WWW-Authenticate": entete})


def utilisateur_courant(portees: SecurityScopes, jeton: str | None = Depends(oauth2)) -> dict:
    if not jeton:
        raise _refus("Jeton absent", portees)

    try:
        charge = jwt.decode(jeton, config.JWT_SECRET_KEY, algorithms=[config.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise _refus("Jeton expiré", portees) from None
    except jwt.InvalidTokenError:
        raise _refus("Jeton invalide", portees) from None

    identifiant = charge.get("sub")
    if not identifiant:
        raise _refus("Jeton sans sujet", portees)

    accordees = charge.get("scopes", [])
    for exigee in portees.scopes:
        if not any(secrets.compare_digest(exigee, a) for a in accordees):
            raise _refus(f"Portée '{exigee}' requise", portees, status.HTTP_403_FORBIDDEN)

    return {"identifiant": identifiant, "portees": accordees}
