# Notebooks

Projet découpé en 3 notebooks indépendants (chacun s'exécute seul) :

1. **Exploration (EDA)** — qualité des données, cible & déséquilibre, baselines, corrélations, géo/saison
2. **Préparation** — feature `Month`, `ColumnTransformer` anti-fuite, encodage
3. **Modélisation** — Régression Logistique, Random Forest (+ `balanced`), comparatif, conclusions

## ⚠️ Lire les notebooks (le rendu `.ipynb` de GitHub est défaillant)

Le viewer Jupyter de GitHub échoue régulièrement (« An error occurred ») même sur des notebooks valides et légers — c'est un **bug de leur moteur de rendu**, pas des fichiers (vérifié : ils se rendent correctement avec les versions exactes de nbformat/nbconvert de GitHub). Pour lire le contenu **avec les résultats**, au choix :

| Méthode | Lien / chemin | Fiabilité |
|---|---|---|
| **Markdown** (rendu GitHub natif) | [`md/01_exploration.md`](md/01_exploration.md) · [`md/02_preprocessing.md`](md/02_preprocessing.md) · [`md/03_modelisation.md`](md/03_modelisation.md) | ✅ **infaillible** (GitHub rend toujours le `.md`) |
| **Google Colab** (lecture + exécution) | `https://colab.research.google.com/github/JordanSerafini/MlOps_Meteo/blob/master/Notebooks/01_exploration.ipynb` | ✅ très fiable |
| **nbviewer** | `https://nbviewer.org/github/JordanSerafini/MlOps_Meteo/blob/master/Notebooks/01_exploration.ipynb` | ⚠️ parfois surchargé |

(remplacer `01_exploration` par `02_preprocessing` / `03_modelisation`)

## Fichiers

| Chemin | Contenu |
|---|---|
| `01_*.ipynb` `02_*.ipynb` `03_*.ipynb` | notebooks **exécutés** (graphiques + scores inline) |
| `md/*.md` (+ `*_files/`) | export **Markdown** — **à privilégier pour lire sur GitHub** |
| `clean/*.ipynb` | notebooks **sans sorties** (légers) |
| `ORIGINAL_…Australie.ipynb` | notebook Colab d'origine (référence, intact) |

Les analyses rédigées (chiffres + interprétations) sont aussi dans `../Analyses/EDA_Analyse_Donnees.md`.
