"""Génère App/.env et App/.env.api avec des secrets neufs.

Les hachages bcrypt contiennent des `$`. Docker Compose interpole le contenu des
`env_file`, donc un `$` non doublé fait disparaître la fin du hachage : le conteneur
reçoit `$2b$12` et refuse tous les mots de passe sans rien dire. D'où le doublement
systématique ici, et la vérification de longueur côté API.
"""
import secrets
import string
import sys
from pathlib import Path

import bcrypt

APP = Path(__file__).resolve().parents[1]


def mot_de_passe(longueur=24):
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(longueur))


def main():
    for fichier in (APP / ".env", APP / ".env.api"):
        if fichier.exists() and "--force" not in sys.argv:
            sys.exit(f"{fichier} existe déjà. Relancer avec --force pour l'écraser.")

    jwt_secret = secrets.token_urlsafe(48)
    client, admin = mot_de_passe(), mot_de_passe()
    h_client = bcrypt.hashpw(client.encode(), bcrypt.gensalt()).decode()
    h_admin = bcrypt.hashpw(admin.encode(), bcrypt.gensalt()).decode()

    (APP / ".env.api").write_text(
        "# Lu directement par le conteneur. Les $ sont doublés pour survivre à\n"
        "# l'interpolation de docker compose.\n"
        f"JWT_SECRET_KEY={jwt_secret}\n"
        "JWT_EXPIRE_MINUTES=60\n"
        "API_CLIENT_USER=client\n"
        f"API_CLIENT_PASSWORD_HASH={h_client.replace('$', '$$')}\n"
        "API_ADMIN_USER=admin\n"
        f"API_ADMIN_PASSWORD_HASH={h_admin.replace('$', '$$')}\n"
    )

    (APP / ".env").write_text(
        "# Variables d'interpolation de docker compose.\n"
        "API_BIND=127.0.0.1\n"
        "DECISION_THRESHOLD=0.5\n"
        "DRIFT_SHARE_THRESHOLD=0.5\n"
        "ALERT_WEBHOOK_URL=\n"
        "\n"
        "# Comptes API. Les mots de passe en clair servent aux appels locaux du\n"
        "# Makefile et de Streamlit ; sur le serveur, seuls les hachages sont déployés.\n"
        "API_CLIENT_USER=client\n"
        f"API_CLIENT_PASSWORD={client}\n"
        "API_ADMIN_USER=admin\n"
        f"API_ADMIN_PASSWORD={admin}\n"
    )

    print("App/.env et App/.env.api regénérés")
    print(f"  client : {client}")
    print(f"  admin  : {admin}")
    print("\nSur le serveur, recopier JWT_SECRET_KEY et les deux hachages dans .env.api ;")
    print("les mots de passe en clair n'ont rien à y faire.")


if __name__ == "__main__":
    main()
