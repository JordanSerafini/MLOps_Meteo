# Rain in Australia : prédire la pluie du lendemain, et faire tourner le tout en production

Projet MLOps réalisé dans le cadre du parcours Data Scientist (DataScientest / Mines Paris PSL).
L'objectif de départ est simple : à partir des relevés météo d'aujourd'hui sur 49 stations
australiennes, dire s'il pleuvra demain. Oui ou non.

Le modèle n'est qu'une partie du travail. Ce qui nous a occupés le plus longtemps, c'est tout ce
qui vient autour : versionner les données, suivre les entraînements, servir le modèle derrière une
API, découper l'application en services, orchestrer, sécuriser, surveiller la production et savoir
mettre à jour sans casser.

Démonstration en ligne : https://streamlit.jordan-s.org

## Les données

Le jeu de données vient de Kaggle ([weather dataset rattle
package](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)) et couvre dix ans de
relevés quotidiens, de 2007 à 2017.

| | |
|---|---|
| Lignes | 145 460 |
| Colonnes | 23 |
| Stations | 49 |
| Cible | `RainTomorrow`, oui ou non (oui = au moins 1 mm de pluie) |
| Jours de pluie | 22,42 % |

Le déséquilibre de la cible est le vrai problème du projet. Comme il ne pleut qu'un peu plus d'un
jour sur cinq, un modèle qui répondrait toujours "non" obtiendrait déjà 77,58 % de bonnes réponses
tout en étant inutile. C'est pour ça qu'on ne regarde pas l'accuracy, mais le rappel sur
la classe "pluie" : parmi tous les jours où il a effectivement plu, combien avons-nous annoncés ?

Deuxième repère à battre : prédire "demain comme aujourd'hui" donne 76,23 %. Ce n'est pas si mal,
et ça s'explique. Quand il pleut aujourd'hui, il pleut demain dans 46 % des cas, contre 15 %
quand il ne pleut pas.

## Résultats

Découpage 80/20 stratifié, soit 113 754 lignes pour l'entraînement et 28 439 pour le test.
Le préprocessing (imputation, normalisation, encodage) est dans un `Pipeline` scikit-learn ajusté
sur le train uniquement, pour éviter toute fuite de données.

| Modèle | Accuracy | Précision | Rappel | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Toujours "non" (référence) | 0,776 | 0 | 0 | 0 | |
| Régression logistique | 0,849 | 0,734 | 0,515 | 0,605 | 0,873 |
| Régression logistique `balanced` | 0,796 | 0,530 | 0,780 | 0,631 | 0,874 |
| Forêt aléatoire | 0,856 | 0,773 | 0,507 | 0,613 | 0,889 |
| Gradient boosting | 0,858 | 0,757 | 0,537 | 0,628 | 0,891 |

La conclusion qui nous a le plus surpris, c'est que le choix du modèle compte peu. Quatre points
séparent le meilleur du moins bon. Ce qui change vraiment les résultats, c'est le seuil de
décision : le modèle sort une probabilité, et c'est nous qui décidons à partir de quelle valeur on
annonce la pluie. En descendant le seuil de 0,50 à 0,32, le rappel passe d'environ 0,54 à 0,71.
On y gagne dix-sept points là où le changement de modèle en donnait quatre.

La variable la plus utile est l'humidité à 15 h, loin devant les autres, suivie de la pression
atmosphérique et de la vitesse des rafales.

## Les rapports

Les trois documents demandés pour la validation du projet :

1. Exploration, visualisation et préprocessing : [`Notebooks/md/01_exploration.md`](Notebooks/md/01_exploration.md)
   et [`Notebooks/md/02_preprocessing.md`](Notebooks/md/02_preprocessing.md), complétés par
   l'analyse rédigée dans [`Analyses/EDA_Analyse_Donnees.md`](Analyses/EDA_Analyse_Donnees.md)
2. Modélisation : [`Notebooks/md/03_modelisation.md`](Notebooks/md/03_modelisation.md)
3. Rapport final : ce README, plus les documents techniques cités plus bas

Les notebooks sont aussi disponibles au format `.ipynb` dans [`Notebooks/`](Notebooks/), mais le
viewer de GitHub échoue régulièrement sur ce format. Les versions Markdown sont là pour ça.

## Comment c'est construit

Onze services décrits dans un seul fichier Docker Compose. Le principe qu'on a suivi pour le
découpage : deux choses qui n'ont pas le même cycle de vie ne vivent pas dans le même conteneur.
L'entraînement est lourd, occasionnel, et a besoin du jeu de données complet. L'inférence est
légère, permanente, et n'a besoin que du modèle.

```
    poste de développement                serveur (VM dédiée)
    ----------------------                -------------------
    weatherAUS.csv (DVC)
           |
           v
      conteneur trainer  --- log + register --->  MLflow (tracking + registry)
                                                        |
                                                   alias champion
                                                        |
                                                        v
                                        API FastAPI  <---- Streamlit
                                             |
                                        journal JSONL
                                             |
                                             v
                                     job de dérive (Evidently)
                                             |
                                   Prometheus / Grafana / Alertmanager

    Airflow orchestre le déploiement et la vérification quotidienne de la dérive.
```

Le point de contact entre l'entraînement et le service, c'est le Model Registry de MLflow.
L'entraînement pousse une nouvelle version et pose l'alias `champion` dessus, l'API va chercher
cet alias. Aucun fichier de modèle ne circule à la main, aucun volume partagé entre les deux.

Sur le serveur, seuls cinq conteneurs tournent en permanence : MLflow, l'API, le Streamlit,
Airflow et sa base. Le service d'entraînement n'y est jamais lancé, il n'y a de toute façon ni le
CSV ni l'image correspondante. C'est un choix assumé, détaillé dans
[`deployement/README.md`](deployement/README.md).

### Les outils et leur rôle ici

| Outil | Ce qu'il fait dans le projet |
|---|---|
| Docker / Compose | isole chaque service, décrit toute la stack dans un fichier |
| DVC | versionne le CSV de 14 Mo hors de git, avec un remote sur DagsHub |
| MLflow | suit les entraînements et stocke les modèles versionnés |
| FastAPI | expose le modèle, gère l'authentification par jeton |
| Streamlit | interface de démonstration et support de notre soutenance |
| Airflow | déclenche le déploiement et vérifie la dérive chaque matin |
| Prometheus | collecte les métriques de l'API toutes les 15 secondes |
| Grafana | affiche ces métriques dans deux tableaux de bord versionnés |
| Alertmanager | route les alertes déclenchées par Prometheus |
| Evidently | compare les données reçues à celles de l'entraînement |
| nginx | reverse proxy et HTTPS devant les services exposés |

## Démarrer en local

Il faut Docker, Docker Compose et Python 3.12.

```bash
git clone git@github.com:JordanSerafini/MLOps_Meteo.git
cd MLOps_Meteo

python3 -m venv .venv
make install          # dépendances de test et de développement

dvc pull              # récupère le CSV depuis le remote

make secrets          # génère App/.env et App/.env.api avec des secrets neufs
make up               # MLflow, API et Streamlit
```

`make up` reste au premier plan et affiche les logs, donc il faut ouvrir un second terminal pour
la suite. Le Streamlit répond sur http://localhost:8501 et l'API sur http://localhost:8000
(documentation interactive sur `/docs`).

Au premier lancement, le registry est vide et l'API n'a donc aucun modèle à servir. Il faut en
entraîner un :

```bash
make train            # forêt aléatoire par défaut, enregistrée avec l'alias champion
make health           # doit afficher model_loaded: true
make predict          # une prédiction de démonstration
```

`make train` enchaîne l'entraînement puis un rechargement de l'API, ce qui prend une minute ou
deux selon la machine.

Autres commandes utiles :

```bash
make test             # 74 tests unitaires
make lint             # ruff
make airflow-up       # ajoute Airflow (http://localhost:8080)
make monitoring-up    # ajoute Prometheus, Grafana, Alertmanager et le pushgateway
make down             # arrête tout
```

`make help` liste le reste.

## L'API

| Méthode et route | Authentification | Rôle |
|---|---|---|
| `POST /token` | aucune | échange identifiant et mot de passe contre un jeton JWT |
| `GET /health` | ouverte | état du service et version du modèle chargé |
| `POST /predict` | jeton, portée `predict` | la prédiction |
| `POST /reload` | jeton, portée `admin` | recharge le modèle `champion` depuis MLflow |
| `GET /metrics` | ouverte | métriques au format Prometheus |

Deux comptes avec des portées différentes, générés par `make secrets`. Le compte `client` peut
prédire mais pas recharger le modèle : dans ce cas l'API répond 403 et non 401, parce que la
requête est authentifiée mais pas autorisée. C'est la distinction qu'on voulait montrer.

## Surveillance et maintenance

L'API expose son débit, sa latence par centile, ses codes de réponse, un compteur de décisions et
la distribution des probabilités qu'elle renvoie. Prometheus vient lire tout ça, deux tableaux de
bord Grafana l'affichent, et six règles d'alerte surveillent les cas anormaux.

Le point le moins évident de cette phase, c'est qu'on ne peut pas mesurer si le modèle a raison :
la réponse n'arrive que le lendemain. On surveille donc ce qui est observable tout de suite, dont
la distribution des données reçues. Un job Evidently la compare chaque matin à un échantillon figé
des données d'entraînement.

Sa calibration nous a pris du temps et c'est documenté dans
[`Docs/CALIBRATION_DRIFT.md`](Docs/CALIBRATION_DRIFT.md). En résumé : avec les réglages par
défaut, le détecteur annonçait de la dérive sur des données strictement normales. Il a fallu
mesurer le seuil au lieu de le deviner, et imposer un minimum de 400 observations en dessous
duquel le job refuse de conclure.

Le réentraînement n'est pas déclenché automatiquement, et c'est volontaire. Pour qu'Airflow lance
lui-même un entraînement, il faudrait lui redonner l'accès au démon Docker qu'on lui a retiré en
production, ce qui revient à donner les droits administrateur de la machine à un service exposé
sur Internet. Le pipeline signale, un humain lance.

## Intégration continue

À chaque push sur une branche de travail, GitHub Actions exécute le linter, les 74 tests, valide
les deux fichiers Compose et construit les cinq images. Deux vérifications supplémentaires nous
ont paru utiles après une mauvaise surprise : la CI contrôle qu'un seul port est publié pour l'API
en production, et qu'aucun service n'écoute sur `0.0.0.0` par défaut. Nos machines ont une IPv6
publique non filtrée, un service exposé sur toutes les interfaces devient joignable depuis
Internet sans passer par le proxy.

## Organisation du dépôt

| Chemin | Contenu |
|---|---|
| `App/src/` | le code : données, entraînement, dérive, API, application Streamlit |
| `App/src/tests/` | les tests unitaires |
| `App/dags/` | les deux DAG Airflow |
| `App/monitoring/` | configuration Prometheus, règles d'alerte, Alertmanager, dashboards Grafana |
| `App/docker/` | les Dockerfiles |
| `App/docker-compose.yml` | la stack complète, avec `docker-compose.prod.yml` en surcharge |
| `Notebooks/` | les trois notebooks, plus le notebook Colab d'origine laissé intact |
| `Analyses/` | l'analyse exploratoire rédigée |
| `Docs/` | sujet, feuille de route, calibration de la détection de dérive |
| `deployement/` | documentation de déploiement et scripts de configuration nginx |
| `k8s/` | manifests Kubernetes pour la mise à l'échelle de l'API |
| `Data/` | le CSV, suivi par DVC |

## Ce qui n'est pas fait

Autant le dire, plutôt que de laisser croire le contraire.

Le rappel sur la classe "pluie" plafonne autour de 0,54 au seuil servi en production. C'est le
principal chantier restant, et la piste la plus rentable est de descendre ce seuil.

Les manifests Kubernetes sont écrits et cohérents mais n'ont jamais été appliqués sur un vrai
cluster, faute d'en avoir un. La détection de dérive est testée sur une dérive géographique, donc
brutale ; une dérive saisonnière progressive demanderait de suivre une tendance plutôt qu'un seuil
instantané. Les dépendances sont épinglées mais pas mises à jour automatiquement. Enfin, la route
`/metrics` reste accessible depuis l'extérieur alors qu'elle devrait être fermée au niveau du
reverse proxy.

## L'équipe

Jordan, Abdelmalek, Rodrigue et Karine. Chacun a travaillé sur une phase du projet et la présente
lors de la soutenance. Le développement s'est fait sur des branches séparées, une par personne,
fusionnées dans `master` par pull request.
