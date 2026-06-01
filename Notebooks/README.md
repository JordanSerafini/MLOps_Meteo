# Notebooks

Le projet est découpé en 3 notebooks indépendants (chacun s'exécute seul) :

1. **`01_exploration.ipynb`** — chargement + EDA (qualité des données, cible & déséquilibre, baselines, corrélations, géo/saison)
2. **`02_preprocessing.ipynb`** — préparation des données (feature `Month`, `ColumnTransformer` anti-fuite, encodage)
3. **`03_modelisation.ipynb`** — Régression Logistique, Random Forest (+ `balanced`), comparatif, conclusions

## Deux variantes de chaque notebook

| Dossier | Sorties | Usage |
|---|---|---|
| **`clean/`** | **purgées** (code + markdown) | ✅ s'ouvrent de façon fiable **sur GitHub** (légers) |
| racine `Notebooks/` | **exécutées** (graphiques + scores inline) | résultats visibles — voir en local ou via **nbviewer** |

> GitHub échoue parfois à rendre les notebooks lourds (« An error occurred ») — c'est un souci de leur moteur de rendu, pas du fichier. Pour voir les **résultats** sans lancer le code :
>
> **nbviewer** (rendu fiable depuis ce repo) :
> `https://nbviewer.org/github/JordanSerafini/MlOps_Meteo/blob/master/Notebooks/01_exploration.ipynb`
> (idem `02_preprocessing.ipynb`, `03_modelisation.ipynb`)

Les analyses rédigées (chiffres + interprétations) sont aussi dans `Analyses/EDA_Analyse_Donnees.md`.

Le notebook Colab d'origine est conservé tel quel comme référence : `ORIGINAL_Analyse_et_Modélisation_du_Précipitations_en_Australie.ipynb`.
