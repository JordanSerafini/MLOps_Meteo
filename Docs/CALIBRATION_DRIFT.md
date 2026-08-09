# Calibration du seuil de dérive

Le job `App/src/drift.py` compare les observations reçues par l'API à un échantillon du jeu
d'entraînement, et recommande un réentraînement quand trop de colonnes ont dérivé. Reste à
définir « trop ».

## Le problème

Evidently teste chaque colonne séparément (distance de Wasserstein normalisée pour les
numériques, Jensen-Shannon pour `Location`) avec un seuil par défaut de 0,1. Une première
exécution sur du trafic *normal* — 400 observations tirées du dataset d'entraînement lui-même —
donnait déjà trois colonnes en dérive sur neuf :

| Colonne | Distance | Seuil | Verdict |
|---|---|---|---|
| Pressure3pm | 0,120 | 0,1 | dérive |
| WindGustSpeed | 0,105 | 0,1 | dérive |
| Location | 0,137 | 0,1 | dérive |
| les six autres | 0,044 à 0,090 | 0,1 | — |

Les trois valeurs dépassent le seuil de peu. Ce n'est pas une dérive, c'est du bruit
d'échantillonnage : comparer 400 observations à 5 000 produit mécaniquement des écarts.

Avec un seuil de déclenchement à 0,30, le job recommandait donc un réentraînement sur des
données strictement identiques à celles de l'entraînement.

## La mesure

Quinze tirages de trafic normal (400 observations chacun) et quinze tirages de trafic
volontairement décalé (station de Portland, 37 % de jours pluvieux contre 22 % en moyenne) :

| Trafic | Minimum | Médiane | Maximum |
|---|---|---|---|
| normal | 0,111 | 0,111 | **0,333** |
| décalé (Portland) | **0,889** | 1,000 | 1,000 |

Les deux distributions ne se recouvrent pas. N'importe quel seuil entre 0,34 et 0,88 sépare
correctement les deux cas.

## Le choix

`DRIFT_SHARE_THRESHOLD = 0.5`, à peu près au milieu de l'intervalle. Sur ces trente tirages :
aucun faux positif, aucun faux négatif.

## Combien d'observations faut-il ?

La première version du job acceptait de conclure dès 30 observations. Un essai sur 120
prédictions réelles a renvoyé 78 % de colonnes en dérive — sur du trafic parfaitement normal.
Le seuil de 0,5 ne valait donc que pour la taille d'échantillon utilisée pour le calibrer.

Mesure de la part de colonnes en dérive selon la taille du jeu courant (8 tirages normaux,
5 tirages décalés par taille) :

| Observations | Normal (max) | Normal (médiane) | Portland (min) | Exploitable |
|---|---|---|---|---|
| 50 | 1,00 | 0,78 | 1,00 | non |
| 100 | 1,00 | 0,72 | 0,89 | non |
| 200 | 0,67 | 0,44 | 0,89 | marge trop faible |
| 400 | 0,33 | 0,22 | 0,89 | oui |
| 800 | 0,22 | 0,00 | 1,00 | oui |
| 1500 | 0,11 | 0,00 | 1,00 | oui |

En dessous de 200 observations, les deux distributions se confondent : le test répond « dérive »
quoi qu'on lui donne. La séparation devient franche à partir de 400, où le pire cas normal
(0,33) reste loin du meilleur cas décalé (0,89).

D'où `--min-samples 400`. Sous ce volume, le job renvoie `statut: donnees_insuffisantes` et
le DAG s'arrête en `skipped` — mieux vaut une absence de réponse qu'une réponse fausse dans
un sens ou dans l'autre.

## Réserves

- La dérive testée est géographique (une station atypique), donc brutale. Une dérive
  saisonnière, progressive, produirait des valeurs intermédiaires et demanderait de suivre la
  tendance plutôt qu'un seuil instantané.
- La calibration vaut pour ces neuf colonnes et cette référence de 5 000 lignes. Changer l'un
  ou l'autre impose de refaire la mesure.
- Le dataset s'arrête en 2017 : on ne peut pas valider sur une vraie dérive de production.

## Reproduire

```bash
make drift-demo     # génère du trafic décalé puis lance le job
```

Le rapport HTML détaillé est archivé dans le volume `drift-reports`, le résumé exploitable
par Airflow dans `dernier_resume.json`.
