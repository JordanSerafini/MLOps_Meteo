# Le Streamlit de soutenance

Support de l'oral **et** démonstration du produit. Cinq onglets : une démonstration commune,
puis une partie par personne, calée sur les quatre phases de la feuille de route.

| Onglet | Fichier | Contenu |
|---|---|---|
| Démonstration | `onglets/prediction.py` | prédiction en direct via l'API |
| 1 · Données | `onglets/donnees.py` | exploration, cible déséquilibrée, préprocessing |
| 2 · Modélisation | `onglets/modelisation.py` | comparatif, **curseur de seuil**, MLflow, DVC |
| 3 · Industrialisation | `onglets/industrialisation.py` | architecture, **démo 403 sur `/reload`**, CI |
| 4 · Monitoring | `onglets/monitoring.py` | métriques, **calibration de la dérive**, alertes |

Chaque module expose une fonction `afficher()` et rien d'autre. Quatre personnes peuvent donc
travailler leur partie en parallèle sans conflit de fusion : un fichier par personne.

## Avant l'oral

1. Mettre vos prénoms et le minutage dans `PARTIES`, en tête de `app.py`. Ils s'affichent en
   tête de chaque partie et dans le déroulé affiché en barre latérale.
2. Régénérer les artefacts si les notebooks ou le jeu de données ont changé : `make artefacts`
   (~1 min, entraîne les quatre modèles et écrit `artefacts/*.json`).
3. Vérifier que tout se rend : `make test` inclut `test_app.py`, qui exécute les cinq onglets
   et échoue si l'un lève une exception.

## Pourquoi tout est figé dans `artefacts/`

Le conteneur n'a ni scikit-learn, ni le CSV de 145 460 lignes, ni les notebooks. Les chiffres
affichés viennent de fichiers JSON versionnés, produits par `App/scripts/exporter_artefacts.py`.
La seule dépendance vivante est l'API d'inférence, et seul l'onglet de démonstration en a besoin.

Une démonstration qui entraîne un modèle à chaud est une démonstration qui tombe le jour de
l'oral. Les onglets 1, 2 et 4 fonctionnent même API éteinte ; l'onglet 3 affiche alors
« injoignable », ce qui est une information et pas une panne.

| Artefact | Source |
|---|---|
| `eda.json` | copie de `Data/eda_stats.json` (`python Data/eda_explore.py`) |
| `modeles.json`, `courbes.json`, `seuil.json`, `importances.json` | `make artefacts` |
| `calibration_drift.json` | mesures de `Docs/CALIBRATION_DRIFT.md`, écrites à la main |

## Lancer

```bash
make up                # API + MLflow + Streamlit dans Docker → http://localhost:8501
make streamlit-local   # hors Docker, contre une API déjà démarrée
```

Le compte utilisé par l'application est le compte `client` (`API_USER` / `API_PASSWORD`), qui a
la portée `predict` et pas la portée `admin` : c'est ce qui permet de montrer un vrai 403 sur
`/reload` depuis l'onglet 3.
