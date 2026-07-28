# Déploiement — Meteo Liora 🌧️

> Documentation du déploiement en production du projet MLOps « Rain in Australia ».
> Rédigée pour l'équipe : elle explique **ce qui tourne où, pourquoi, et comment on s'en sert**.

---

## 1. Vue d'ensemble

Le projet est déployé en **split strict** : l'entraînement (lourd) reste sur un PC de dev,
le serveur ne fait tourner que les services légers (serving + orchestration).

```
PC DE DEV (192.168.1.10)             VM meteo-liora (192.168.1.36)           VM proxy (192.168.1.32)         INTERNET
────────────────────────             ─────────────────────────────           ───────────────────────         ────────
make train-remote                     mlflow    :5000  (tracking+registry)    nginx + certbot (SSL)
  │ entraîne EN LOCAL                 api       :8000  (FastAPI inférence)      │
  │ (conteneur trainer)               streamlit :8501  (démo web)               ├─ streamlit.jordan-s.org ◄── public
  └── pousse le modèle ──────────────▶ airflow  :8080  (orchestration)          ├─ api.meteo.jordan-s.org ◄── public
      @champion au registry           airflow-db       (PostgreSQL interne)     ├─ mlflow.jordan-s.org    ◄── basic-auth
                                                                                └─ airflow.jordan-s.org   ◄── login admin
make deploy-model
  └── déclenche le DAG Airflow ──────▶ reload → check → smoke test
```

**Pourquoi ce découpage ?**
- Le serveur a des specs limitées : entraîner un RandomForest sur 145 000 lignes n'y a pas sa place.
- Le serveur MLflow tourne avec `--serve-artifacts` : le PC peut donc entraîner **à distance**
  (il pointe `MLFLOW_TRACKING_URI` vers la VM) et le modèle atterrit directement dans le registry
  du serveur. L'API n'a plus qu'à faire un `POST /reload` pour charger la nouvelle version.
- Conséquence : **ni le CSV de données, ni l'image `trainer` n'existent sur le serveur.**

---

## 2. L'infrastructure

| Machine | IP | Rôle |
|---|---|---|
| Node **Proxmox** | `192.168.1.21` (UI web :8006) | hyperviseur qui héberge les VMs |
| VM **meteo-liora** | `192.168.1.36` | notre stack MLOps (VM dédiée au projet) |
| VM **proxy** | `192.168.1.32` | reverse proxy nginx + certificats SSL (partagée avec d'autres projets) |
| PC de dev | `192.168.1.10` | entraînement + pilotage des déploiements |

### La VM `meteo-liora` (créée pour ce projet)

- **VMID 106** sur Proxmox, créée en CLI (`qm create` + image cloud Ubuntu 24.04 + cloud-init).
- **2 vCPU, 6 Go RAM** (ballooning : plancher 3 Go — la VM rend de la RAM au node sous pression), **40 Go disque**.
- Ubuntu 24.04, Docker + Docker Compose installés (script officiel `get.docker.com`).
- Accès : SSH par clé, user `jordan` (groupe `docker` → pas besoin de sudo pour Docker).
- La stack vit dans `/home/jordan/meteo-liora/` :
  ```
  meteo-liora/
  ├── docker-compose.yml        # base (versionné dans le repo → App/)
  ├── docker-compose.prod.yml   # override prod (versionné dans le repo → App/)
  ├── .env.prod                 # secrets (JAMAIS commité)
  └── dags/training_pipeline.py # DAG Airflow
  ```

---

## 3. Les services (docker compose)

| Service | Image | Port | Rôle |
|---|---|---|---|
| `mlflow` | build custom (`Dockerfile.mlflow`) | 5000 | tracking + **Model Registry** (SQLite + artefacts servis par le serveur) |
| `api` | build custom (`Dockerfile.api`) | 8000 | FastAPI : `GET /health`, `POST /predict`, `POST /reload`, `GET /metrics` |
| `streamlit` | build custom (`Dockerfile.streamlit`) | 8501 | interface de démo (appelle l'API en interne via `http://api:8000`) |
| `airflow` | `apache/airflow:2.10.4` | 8080 | orchestration du **déploiement** (pas de l'entraînement !) |
| `airflow-db` | `postgres:16-alpine` | interne | métadonnées Airflow |
| `prometheus` | `prom/prometheus:v2.54.1` | 9090 | collecte des métriques (profil `monitoring`) |
| `grafana` | `grafana/grafana:11.2.0` | 3000 | tableaux de bord provisionnés par fichier (profil `monitoring`) |
| `pushgateway` | `prom/pushgateway:v1.9.0` | interne | relais pour les métriques du job de dérive, qui est éphémère |
| `drift` | build custom (`Dockerfile.drift`) | — | job one-shot de détection de dérive (profil `drift`) |
| `trainer` | build custom (`Dockerfile.trainer`) | — | **jamais lancé sur le serveur** (profil `train`, utilisé sur le PC) |

Les interfaces d'écoute sont pilotées par deux variables : `API_BIND` pour l'API, `HOST_BIND`
pour les autres services. Les deux valent `127.0.0.1` par défaut (poste de dev : rien n'est
exposé, même pas au LAN) et l'IP LAN de la VM en production, `192.168.1.36`, qui est la seule
adresse que le reverse proxy a besoin de joindre. **Ne jamais y mettre `0.0.0.0`** : cf. le
piège n° 6 du §7, la VM a une IPv6 publique et le service se retrouverait sur Internet.

Le port de l'API n'est déclaré que dans le fichier de base — compose **concatène** les listes
de ports entre fichiers, donc le redéclarer dans l'override provoquerait un second binding sur
8000 et un échec au démarrage. La CI vérifie ce point, ainsi que l'absence de bind sur
`0.0.0.0` quand aucune variable n'est fournie.

Démarrage sur la VM :

```bash
cd ~/meteo-liora
docker compose --env-file .env.prod \
  -f docker-compose.yml -f docker-compose.prod.yml \
  --profile orchestration up -d mlflow api streamlit airflow airflow-db
```

Le fichier `docker-compose.prod.yml` (override de prod) ajoute :
- `restart: unless-stopped` partout (survie au reboot de la VM) ;
- des limites mémoire par service (airflow 2G, api 1G, les autres 512M) ;
- l'API REST Airflow en basic-auth (pour déclencher le DAG à distance) ;
- le proxy-fix Airflow (fonctionnement correct derrière nginx en HTTPS).

**Transfert des images** : pas de registry Docker — les images sont buildées sur le PC puis
transférées par SSH :

```bash
docker save meteo-liora-mlflow meteo-liora-api meteo-liora-streamlit \
  | gzip -1 | ssh jordan@192.168.1.36 'gunzip | docker load'
```

---

## 4. Le flux MLOps complet

### Entraîner + déployer un nouveau modèle (depuis le PC de dev)

```bash
make deploy-model        # = train-remote + déclenchement du DAG
```

Ce qui se passe, étape par étape :

1. **`make train-remote`** — lance le conteneur `trainer` **en local** avec
   `MLFLOW_TRACKING_URI=http://192.168.1.36:5000`. Le script `src/train.py` :
   - entraîne le pipeline sklearn (préprocesseur + RandomForest) ;
   - loggue params/métriques dans MLflow (expérience `rain-australia`) ;
   - enregistre une **nouvelle version** du modèle au registry ;
   - pose l'alias **`@champion`** si `recall_rain >= MIN_RECALL_FOR_CHAMPION` (gate qualité).
2. **Déclenchement du DAG** `rain_australia_deploy_pipeline` via l'API REST Airflow
   (`POST /api/v1/dags/.../dagRuns`, basic-auth admin). Le DAG (`schedule=None`, déclenché
   à la demande) enchaîne :
   1. `wait_for_api` — l'API répond ? (`/health`)
   2. `reload_api` — `POST /reload` → l'API recharge `models:/rain-australia@champion`
   3. `check_model_loaded` — `/health` confirme `model_loaded: true`
   4. `smoke_test` — une vraie prédiction `POST /predict`
   5. `verify_prediction` — la réponse est cohérente (probabilité ∈ [0,1], champs présents)
3. En cas d'échec du DAG, le Makefile a un **fallback** : `curl -X POST .../reload` en direct.

> ℹ️ L'ancien DAG entraînait le modèle sur le serveur via un `DockerOperator` (montage du socket
> Docker). On l'a réécrit : le split strict supprime l'entraînement serveur **et** le socket
> Docker monté dans Airflow (surface d'attaque en moins).

### Surveiller le modèle une fois déployé

L'API journalise chaque prédiction en JSONL (volume `prediction-logs`) : c'est le jeu de
données « courant » auquel le job de dérive compare la distribution d'entraînement.

```bash
make monitoring-up   # Prometheus :9090, Grafana :3000, pushgateway
make trafic N=500    # envoie 500 prédictions à l'API
make drift           # compare et pousse les métriques
make drift-demo      # variante : injecte du trafic décalé pour déclencher la dérive
```

Le job `drift` produit trois sorties : un rapport HTML Evidently archivé dans le volume
`drift-reports`, un `dernier_resume.json` exploitable par Airflow, et quatre métriques poussées
au pushgateway (`rain_dataset_drift`, `rain_n_drifted_columns`, `rain_share_drifted_columns`,
`rain_drift_samples`).

Le DAG `rain_australia_drift_check` (tous les jours à 6 h) relit ce résumé et signale s'il faut
réentraîner. Il ne lance pas le job lui-même : le socket Docker a été volontairement retiré du
conteneur Airflow (§4), et le lui rendre reviendrait à lui donner l'équivalent du root sur
l'hôte. Le job est donc déclenché à côté, et le DAG refuse de conclure sur un résumé de plus
de 36 h plutôt que de raisonner sur des données périmées.

Deux garde-fous valent d'être signalés, parce qu'ils viennent d'erreurs constatées :

- le seuil de déclenchement (0,5) et le minimum de 400 observations ont été **mesurés**, pas
  choisis. Avec les valeurs initiales (0,3 et 30 observations), le job annonçait 78 % de dérive
  sur du trafic parfaitement normal. Le détail est dans `Docs/CALIBRATION_DRIFT.md` ;
- la référence est un échantillon **figé et versionné** (`App/reference/reference.csv`), pas un
  tirage refait à chaque exécution : une référence mouvante rend les comparaisons dans le temps
  inexploitables. Elle est embarquée dans l'image, ce qui permet au job de tourner sur le
  serveur alors que le CSV complet n'y est pas déployé.

### Modèle actuellement en prod

- `rain-australia` **v1**, alias `@champion` — RandomForest, accuracy 0.856, ROC-AUC 0.889.
- Seuil de décision : `0.5` (variable `DECISION_THRESHOLD` dans `.env.prod`).
  Le training recommande `0.34` pour viser un recall de 0.70 sur la classe « pluie ».

---

## 5. Les URLs publiques

Le domaine `jordan-s.org` pointe vers l'IP publique de la box, qui NAT les ports 80/443
vers la VM proxy (`192.168.1.32`). nginx y route chaque sous-domaine vers la VM meteo-liora.

| URL | Service | Accès |
|---|---|---|
| https://streamlit.jordan-s.org | démo Streamlit | 🌐 public |
| https://api.meteo.jordan-s.org | API FastAPI (`/docs` pour Swagger) | 🌐 public |
| https://mlflow.jordan-s.org | MLflow UI | 🔒 basic-auth nginx, utilisateur `jordan` |
| https://airflow.jordan-s.org | Airflow UI | 🔒 login Airflow `admin` |

Certificats **Let's Encrypt** (certbot, renouvellement automatique). Les sous-domaines
sont des enregistrements DNS de type A. MLflow est derrière une basic-auth nginx car il
n'a **aucune authentification native** (n'importe qui pourrait écrire dans le registry).

> `api.mlflow.jordan-s.org` était l'ancien nom du vhost MLflow. Retiré le 28/07/2026
> (vhost + certificat) : le nom canonique est `mlflow.jordan-s.org`.

**SNI par défaut fermé.** Un vhost `default_server` sur le 443 (`00-default-deny`, certificat
auto-signé, `return 444`) intercepte tout nom qui n'a pas son propre vhost. Sans lui, nginx
sert le **premier vhost 443 par ordre alphabétique** — n'importe quel nom DNS pointant vers
l'IP publique exposait donc l'UI Airflow. Le `default_server` du port 80 existait déjà ;
celui du 443 manquait.

Contrairement à MLflow, le vhost Airflow ne pose **pas** de basic-auth : l'UI est protégée
par le seul login Airflow. C'est acceptable parce qu'Airflow a une vraie authentification,
là où MLflow n'en a aucune.

> 🔑 Les mots de passe (admin Airflow, basic-auth MLflow) ne sont **pas dans le repo** :
> demandez-les à Jordan. Ils vivent dans `App/.env.prod` (gitignoré) et
> `/etc/nginx/.htpasswd-mlflow` (sur la VM proxy). Ce dernier est un hachage `$apr1$`,
> donc irréversible : un mot de passe perdu se remplace (`htpasswd`), il ne se retrouve pas.
> Celui de MLflow a été régénéré le 28/07/2026 pour cette raison.

---

## 6. Opérations courantes

Depuis le PC de dev (racine du repo) :

```bash
make deploy-model    # entraîner + déployer (le flux complet du §4)
make train-remote    # entraîner seulement (nouveau modèle au registry, sans reload)
```

Sur la VM (`ssh jordan@192.168.1.36`, ou votre user — cf. §8) :

```bash
cd ~/meteo-liora
alias dc='docker compose --env-file .env.prod -f docker-compose.yml -f docker-compose.prod.yml --profile orchestration'
dc ps                      # état des services
dc logs -f --tail=100 api  # logs d'un service
dc restart api             # redémarrer un service
dc up -d                   # (re)démarrer la stack
```

Vérifications rapides :

```bash
curl http://192.168.1.36:8000/health     # model_loaded doit être true
curl -X POST http://192.168.1.36:8000/predict -H 'Content-Type: application/json' \
  -d '{"Location":"Sydney","Month":7,"RainToday":"Yes","Humidity3pm":80,"Sunshine":3.5,
       "Pressure3pm":1008,"Rainfall":12,"WindGustSpeed":56,"Cloud3pm":8,"Temp3pm":16}'
```

### Mettre à jour le code de l'API / Streamlit

Le code est **embarqué dans les images Docker** (pas de volume monté) :

```bash
# 1. sur le PC : rebuild
docker compose -f App/docker-compose.yml build api
# 2. transfert
docker save meteo-liora-api | gzip -1 | ssh jordan@192.168.1.36 'gunzip | docker load'
# 3. sur la VM : recréer le conteneur
ssh jordan@192.168.1.36 'cd ~/meteo-liora && docker compose --env-file .env.prod \
  -f docker-compose.yml -f docker-compose.prod.yml up -d api'
```

---

## 7. Pièges rencontrés (retours d'expérience)

Instructifs pour comprendre pourquoi certains choix ont été faits :

1. **Port déjà pris** — le premier essai de déploiement visait une VM existante où le port
   8000 était occupé. Résolu proprement en créant une **VM dédiée** au projet (isolation :
   nos services ne cohabitent pas avec 26 autres conteneurs).
2. **`.htpasswd` illisible → HTTP 500** — le fichier basic-auth créé en `root:root 640`
   n'était pas lisible par les workers nginx (`www-data`). Symptôme trompeur : 401 sans
   credentials (OK) mais 500 avec. Fix : `chown root:www-data`.
3. **Sous-domaine DNS sans vhost → mauvais site servi** — `mlflow.jordan-s.org` existait en
   DNS mais pas en vhost nginx : nginx servait alors le **vhost par défaut** du port 443
   (Airflow, premier alphabétiquement) avec un avertissement de certificat. Règle : chaque
   record DNS doit avoir son vhost.
   *Corrigé le 28/07/2026* — deux fixes complémentaires : le vhost `mlflow.jordan-s.org`
   a été créé (basic-auth + certbot), **et** un `default_server` a été ajouté sur le 443
   pour que le problème ne puisse plus se reproduire avec un autre nom. La règle « un
   record = un vhost » reste vraie, mais elle ne doit pas être la seule défense : le
   `default_server` est le filet de sécurité.
4. **Guillemets dans `command:` de compose** — `--password "${VAR}"` à l'intérieur d'un
   `bash -c "…"` casse le parsing (guillemets imbriqués) → l'utilisateur admin Airflow
   n'était jamais créé. Fix : pas de guillemets internes.
5. **RAM du node Proxmox** — le node n'avait que ~5 Go réellement libres. La VM utilise le
   **ballooning** (6 Go max, 3 Go garantis) pour ne pas asphyxier les autres VMs.
6. **L'IPv6 contournait tout le reverse proxy** — le plus sérieux des six, découvert le
   28/07/2026. On raisonnait comme si la VM n'était joignable que par les ports 80/443
   redirigés vers nginx. Faux : la box délègue un préfixe IPv6 public (`/64`) à tout le LAN
   **sans filtrer les connexions entrantes**. Chaque port publié sur `0.0.0.0` l'était donc
   aussi sur `::`, et l'adresse IPv6 de la VM est routée depuis Internet. Vérifié depuis
   l'extérieur : MLflow répondait `200` — c'est-à-dire un Model Registry ouvert en écriture
   au monde entier, en clair, en contournant la basic-auth nginx — ainsi que l'UI Airflow,
   l'API et Streamlit.
   Trois corrections, dans cet ordre d'importance :
   - les ports sont désormais publiés sur l'IP LAN (`HOST_BIND` / `API_BIND` = `192.168.1.36`),
     donc plus rien n'écoute sur `::`. C'est la correction de fond : `ufw` seul n'aurait pas
     suffi, car Docker insère ses règles **avant** la chaîne `INPUT` que ufw contrôle ;
   - `ufw` activé sur la VM (`deny incoming`, LAN et réseau Docker autorisés), ce qui ferme
     le port 22 côté Internet — il restait, lui, joignable en IPv6 ;
   - valeur par défaut des deux variables passée à `127.0.0.1`, pour que l'erreur ne
     puisse pas se reproduire par simple oubli. La CI le vérifie.

   Leçon transposable : tester l'exposition d'un service **depuis l'extérieur et en IPv6**
   (`curl "http://[<ipv6>]:<port>/"`), pas seulement en IPv4. Une redirection de port
   absente ne prouve rien.

---

## 8. Accès des membres de l'équipe

- **Le code** passe par GitHub : branche par personne (`prenom_dev`), PR vers `master` —
  comme d'habitude, rien ne se code directement sur le serveur.
- **Les URLs publiques** (§5) suffisent pour démontrer/tester le projet.
- **Accès SSH à la VM** (pour voir les logs, comprendre docker compose…) : chaque membre a
  un compte personnel sur `192.168.1.36`. ⚠️ **Depuis le LAN de Jordan uniquement.** Le
  pare-feu de la VM n'accepte les connexions que depuis `192.168.1.0/24` : il n'y a pas
  d'accès depuis Internet, ni en IPv4 (aucune redirection de port) ni en IPv6 (fermé le
  28/07/2026, cf. piège n° 6). Ouvrir le 22 au monde pour du confort n'a pas de sens ici :
  ce sont des comptes à mot de passe, sans fail2ban, sur une machine qui héberge le registry.
  Pour travailler à distance, tout passe donc par **GitHub** (code + CI) et par les **URLs
  publiques** du §5 ; les logs et l'état des conteneurs se lisent aussi dans l'UI Airflow
  et dans Grafana. Un accès distant reste possible si le besoin s'en fait sentir : ajouter
  la VM au tailnet Tailscale (`tailscale up --ssh` + partage du seul nœud `meteo-liora`),
  ce qui donne un accès nominatif et révocable sans ouvrir le moindre port.

| Membre | User VM | Accès |
|---|---|---|
| Jordan | `jordan` | clé SSH uniquement (admin, sudo) |
| Abdelmalek | `abdelmalek` | mot de passe (demandez-le à Jordan — changement forcé à la 1re connexion) |
| Rodrigue | `rodrigue` | idem |
| Karine | `karine` | idem |

Les mots de passe temporaires ont été régénérés le 28/07/2026 (les précédents n'avaient été
transmis qu'à l'oral et n'étaient consignés nulle part). Première connexion, depuis le LAN :

```bash
ssh abdelmalek@192.168.1.36   # le mot de passe temporaire vous sera demandé,
                              # puis le système vous force à en choisir un nouveau
docker ps                     # vous êtes dans le groupe docker : ps/logs/exec fonctionnent
```

Les 3 comptes sont membres du groupe `docker` : vous pouvez inspecter les conteneurs
(`docker ps`, `docker logs …`, `docker exec …`) sans sudo. Les fichiers compose vivent
dans `/home/jordan/meteo-liora/` (compte admin) — les opérations `up`/`down` passent
par Jordan ou par le Makefile du repo.

---

*Doc générée le 2 juillet 2026 — état : stack déployée et validée de bout en bout
(entraînement distant → registry → DAG de déploiement → smoke test ✅).*
