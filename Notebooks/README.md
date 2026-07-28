# Notebooks

Trois notebooks, chacun exécutable indépendamment (chacun recharge le CSV depuis zéro).

1. `01_exploration.ipynb` — qualité des données, déséquilibre de la cible, baselines à battre,
   corrélations, géographie et saisonnalité
2. `02_preprocessing.ipynb` — nettoyage, feature `Month`, préprocesseur ajusté sur le train seul
3. `03_modelisation.ipynb` — régression logistique, forêt aléatoire, gradient boosting, courbes
   ROC et précision-rappel, choix du seuil de décision

## Lire les notebooks

Le viewer de GitHub échoue régulièrement sur les `.ipynb` (« An error occurred »), y compris sur
des fichiers valides. Trois solutions, par ordre de fiabilité :

| Format | Chemin | Remarque |
|---|---|---|
| Markdown | [`md/01_exploration.md`](md/01_exploration.md) · [`md/02_preprocessing.md`](md/02_preprocessing.md) · [`md/03_modelisation.md`](md/03_modelisation.md) | rendu natif GitHub, marche toujours |
| Colab | `https://colab.research.google.com/github/JordanSerafini/MLOps_Meteo/blob/master/Notebooks/01_exploration.ipynb` | lecture et exécution |
| nbviewer | `https://nbviewer.org/github/JordanSerafini/MLOps_Meteo/blob/master/Notebooks/01_exploration.ipynb` | parfois lent |

(remplacer `01_exploration` par le notebook voulu)

## Organisation des fichiers

| Chemin | Contenu |
|---|---|
| `0*.ipynb` | notebooks exécutés, avec les graphiques et les scores |
| `clean/` | mêmes notebooks sans les sorties — c'est la version qu'on édite |
| `md/` | export Markdown, à privilégier pour lire sur GitHub |
| `ORIGINAL_…Australie.ipynb` | notebook Colab de départ, laissé intact pour comparaison |

Les bugs du notebook d'origine (imputation en chained assignment, fuite de données, chemins
Colab, appels seaborn dépréciés) sont décrits dans `Perso/bugs.txt`. L'analyse rédigée avec les
chiffres détaillés est dans `../Analyses/EDA_Analyse_Donnees.md`.
