# Où en est le projet — point du 28 juillet 2026

Note pour l'équipe. La soutenance est le 28 août, il reste donc un mois. Ce document fait le
point sur ce qui tourne, ce qui vient d'être ajouté, et ce qu'il reste à faire.

---

## Vue d'ensemble

| Phase | Échéance | État |
|---|---|---|
| 1 — Fondations, conteneurisation | 10 avril | terminée |
| 2 — Microservices, MLflow, DVC | 5 juin | terminée |
| 3 — Orchestration, CI, déploiement | 3 juillet | terminée |
| 4 — Monitoring, dérive, maintenance | 28 août | l'essentiel est en place |

La phase 4 était à zéro il y a quelques jours. Elle est maintenant fonctionnelle et testée de
bout en bout. Ce qui reste tient surtout à la documentation finale et à la préparation de la
soutenance.

---

## 1. La production tourne à nouveau

La VM `meteo-liora` (192.168.1.36) était éteinte, ce qui expliquait les erreurs 502 sur toutes
les adresses publiques. Après redémarrage, les cinq conteneurs sont repartis tout seuls grâce
aux politiques `restart: unless-stopped` posées au moment du déploiement. Rien n'a eu besoin
d'être relancé à la main, ce qui est plutôt une bonne nouvelle sur la robustesse de la stack.

Vérifications faites après redémarrage :

- le modèle est bien rechargé (`model_loaded: true`, version 1) ;
- une prédiction de bout en bout via l'adresse publique répond correctement ;
- le DAG de déploiement Airflow passe en `success` : attente de l'API, rechargement du modèle,
  contrôle, smoke test, validation de la réponse.

Les quatre adresses publiques répondent :

| Adresse | Accès |
|---|---|
| https://streamlit.jordan-s.org | public |
| https://api.meteo.jordan-s.org | public, `/docs` pour Swagger |
| https://mlflow.jordan-s.org | identifiant et mot de passe (demandez-les) |
| https://airflow.jordan-s.org | compte admin |

### Un trou de sécurité corrigé au passage

`mlflow.jordan-s.org` existait en DNS mais n'avait pas de vhost nginx. Résultat : nginx servait
le premier vhost du port 443 dans l'ordre alphabétique, c'est-à-dire l'interface Airflow, avec
un certificat qui ne correspondait pas.

Le piège était déjà décrit dans notre documentation de déploiement, mais seule la moitié du
problème avait été traitée. On a donc fait les deux :

1. créé le vhost `mlflow.jordan-s.org` avec son certificat et son mot de passe. L'ancien nom
   `api.mlflow.jordan-s.org` a été supprimé, vhost et certificat compris ;
2. surtout, ajouté un vhost par défaut sur le port 443. Il n'en existait un que sur le port 80.
   Sans ça, **n'importe quel nom DNS pointant vers notre IP publique** tombait sur Airflow.
   Maintenant la connexion est simplement fermée.

Le second point est le vrai correctif : il vaut pour tous les noms, pas seulement celui qui
posait problème. Les scripts sont versionnés dans `deployement/`, ils sont relançables sans
dommage et se remettent tout seuls en état si la configuration nginx ne passe pas le test.

---

## 2. Les notebooks ont été repris

### Le notebook 2 était pratiquement vide

Il faisait six cellules et s'arrêtait sur la définition du préprocesseur. Le titre annonçait un
préprocessing « sans fuite de données », mais rien ne le démontrait : ni découpage train/test,
ni ajustement, ni vérification. Pour quelqu'un qui corrige, ça revenait à lire une affirmation
sans preuve.

Il fait maintenant vingt cellules, avec le découpage, l'ajustement, et la vérification que les
valeurs d'imputation mémorisées sont bien celles du train.

Un point mérite d'être signalé parce qu'il va à l'encontre de ce qu'on racontait. **La fuite de
données ne coûte rien sur ce dataset.** On a mesuré : l'écart entre la médiane calculée sur tout
le dataset et celle calculée sur le train seul est de 0,0. Refait sur des échantillons de 1 500
puis 300 lignes, l'écart reste de l'ordre de 0,1 sur des variables qui valent 5 à 1 015.

C'est logique avec 113 000 lignes d'entraînement, et il aurait été malhonnête de prétendre le
contraire — n'importe qui refaisant le calcul s'en apercevrait. Le notebook le dit franchement
et bascule sur les deux raisons qui tiennent vraiment :

- en production, l'API reçoit **une seule observation**. Calculer une médiane sur une ligne n'a
  aucun sens : les statistiques doivent avoir été mémorisées à l'entraînement et transportées
  avec le modèle. C'est exactement ce que fait notre API ;
- une station inconnue ne doit pas faire tomber le service. Le notebook envoie volontairement
  `Location = "Marseille"` dans le préprocesseur pour montrer que ça passe.

Cet angle est plus fort que l'argument académique, et il est directement relié à ce qu'on a
déployé.

### Le notebook 3 a été complété

Ajouté : le gradient boosting, les courbes ROC et précision-rappel, et surtout une section sur
le seuil de décision.

Le gradient boosting bat la forêt aléatoire sur presque tout, et il entraîne **2,8 secondes
contre 21,9**. Un point va dans l'autre sens, la précision baisse légèrement (0,76 contre 0,78),
et c'est écrit tel quel — on avait d'abord noté « meilleur sur toutes les métriques », ce qui
était faux.

La section sur le seuil est celle qui a le plus de valeur pour la soutenance. À modèle constant,
descendre le seuil de 0,50 à 0,32 fait passer le rappel de 0,54 à 0,71, en cédant douze points
de précision. Autrement dit : on rattrape un tiers des jours de pluie qu'on manquait. L'écart
entre le meilleur et le pire de nos trois modèles ne vaut, lui, que quatre points. **Le levier
n'est pas le choix du modèle, c'est le seuil**, et c'est un argument qu'on peut défendre.

Une contrainte technique découverte en route : le gradient boosting de scikit-learn refuse les
matrices creuses, alors que notre encodage en produit. Il faut passer `sparse_output=False`.
C'est noté dans le notebook, et ça vaudra aussi si on décide de basculer la production dessus.

### Le style a été revu

Les notebooks avaient des tics d'écriture assez voyants : chaque graphique était suivi d'un
paragraphe ouvert par le même mot en gras, il y avait du gras un peu partout dans les phrases,
et des emojis dans le README. Comptage avant/après : 7 paragraphes formatés à l'identique et 71
passages en gras, contre 0 et 4 maintenant.

Le fond n'a pas changé, il était bon. C'est la régularité mécanique qui donnait une impression
désagréable à la lecture.

Les trois notebooks ont été réexécutés, et les versions Markdown régénérées — c'est celles-là
qu'il faut lire sur GitHub, le viewer de notebooks échoue une fois sur deux.

---

## 3. Le monitoring et la détection de dérive

C'est le gros morceau de la phase 4, et c'était entièrement à faire.

### Ce qui a été mis en place

L'API enregistre désormais chaque prédiction dans un fichier journal (une ligne JSON par appel :
horodatage, identifiant, données reçues, probabilité, décision, version du modèle). Ce fichier
est le jeu de données « courant » auquel on compare les données d'entraînement. Si l'écriture
échoue pour une raison quelconque, la prédiction est quand même servie — un problème de log ne
doit jamais faire tomber le service.

Trois nouveaux services tournent derrière un profil `monitoring` : Prometheus qui collecte,
Grafana qui affiche, et un pushgateway qui sert de relais pour le job de dérive (celui-ci est
éphémère, Prometheus ne peut pas l'interroger directement).

Deux tableaux de bord Grafana sont livrés avec le projet, sous forme de fichiers versionnés :
ils se créent tout seuls au démarrage, personne n'a besoin de les refaire à la main. Le premier
suit la santé de l'API (débit, latence, taux d'erreur), le second suit les prédictions et la
dérive. Les treize requêtes ont été vérifiées une par une comme renvoyant réellement des données.

Le job de dérive utilise Evidently. Il produit un rapport HTML détaillé, un résumé exploitable
par Airflow, et quatre métriques envoyées à Grafana.

Enfin, un second DAG Airflow tourne chaque matin à 6 h : il relit le dernier résumé et signale
s'il faut réentraîner. Ses trois cas de figure ont été testés (dérive détectée, pas de dérive,
résumé trop vieux pour être exploitable).

### Le piège dans lequel on est tombé

C'est la partie la plus instructive, et je pense qu'on devrait en parler en soutenance.

La première version du job annonçait **78 % de colonnes en dérive sur du trafic parfaitement
normal**. Autrement dit, il aurait déclenché un réentraînement en permanence, sur des données
identiques à celles de l'entraînement.

L'explication : Evidently teste chaque variable avec un seuil, et comparer 120 observations à
5 000 produit mécaniquement des écarts. Ce n'était pas de la dérive, c'était du bruit
d'échantillonnage. Nos deux réglages (seuil de déclenchement et nombre minimum d'observations)
avaient été choisis au feeling.

On les a donc mesurés. D'abord la distribution sur trente tirages, quinze de trafic normal et
quinze de trafic volontairement décalé (la station de Portland, 37 % de jours pluvieux contre
22 % en moyenne) :

| Trafic | Minimum | Médiane | Maximum |
|---|---|---|---|
| normal | 0,111 | 0,111 | 0,333 |
| Portland | 0,889 | 1,000 | 1,000 |

Les deux ne se recouvrent pas, n'importe quelle valeur entre 0,34 et 0,88 sépare correctement.
On a retenu 0,5.

Puis la sensibilité au volume, parce que la première calibration ne valait que pour la taille
d'échantillon utilisée :

| Observations | Normal (pire cas) | Portland (meilleur cas) | Utilisable |
|---|---|---|---|
| 100 | 1,00 | 0,89 | non |
| 200 | 0,67 | 0,89 | marge trop faible |
| 400 | 0,33 | 0,89 | oui |
| 800 | 0,22 | 1,00 | oui |

En dessous de 200 observations, le test répond « dérive » quoi qu'on lui donne. On exige donc
400 observations minimum. En dessous, le job répond explicitement « données insuffisantes » et
le DAG s'arrête : mieux vaut pas de réponse qu'une fausse.

Tout est écrit dans `Docs/CALIBRATION_DRIFT.md`, tableaux compris. C'est le genre de démarche
qui distingue un projet où on a réfléchi d'un projet où on a branché une bibliothèque.

Vérification finale après correction : 11 % de dérive sur 500 prédictions normales, 78 % après
injection du trafic de Portland. Le taux de pluie annoncé par le modèle passe au passage de
23 % à 35 %, cohérent avec une station plus humide.

### Un choix à assumer en soutenance

Le réentraînement n'est pas déclenché automatiquement. Deux raisons.

D'abord, on a volontairement retiré l'accès au démon Docker du conteneur Airflow au moment de
la mise en production : le lui rendre reviendrait à lui donner l'équivalent des droits
administrateur sur la machine. Ensuite, l'entraînement est déporté sur un poste de dev par
conception — le serveur n'a ni le fichier de données ni l'image d'entraînement, il n'a pas les
ressources pour ça.

Le DAG signale donc, et le réentraînement se lance avec `make deploy-model`. Si on voulait aller
jusqu'à l'automatisation complète, la solution propre serait un proxy limitant strictement ce
qu'Airflow peut demander à Docker. C'est documenté dans le DAG lui-même.

---

## 4. Divers

Quelques corrections faites en passant :

- l'API n'écoute plus que sur la boucle locale en développement, et sur toutes les interfaces
  uniquement en production. Avant, MLflow et l'API étaient exposés au réseau local sans
  authentification ;
- la CI se déclenche maintenant sur les branches `prenom_dev`. Avant, elle n'était lancée qu'à
  l'ouverture de la pull request, donc trop tard pour être utile. Elle valide aussi les fichiers
  Docker Compose et construit les cinq images ;
- l'échantillon de référence pour la dérive est figé et versionné, au lieu d'être retiré au
  hasard à chaque exécution. Une référence qui bouge rend les comparaisons dans le temps
  inexploitables.

État des tests : 43 tests passent, le linter ne remonte rien.

---

## 5. Comment vous en servir

Depuis la racine du dépôt :

```bash
make up              # MLflow, API, Streamlit
make monitoring-up   # Prometheus (:9090), Grafana (:3000)
make trafic N=500    # envoie 500 prédictions à l'API
make drift           # compare le trafic reçu aux données d'entraînement
make drift-demo      # injecte du trafic décalé puis mesure : la dérive se déclenche
make test            # les 43 tests
```

`make drift-demo` est la commande à montrer en démonstration : elle provoque une dérive à la
demande. Comme nos données s'arrêtent en 2017, on ne peut pas attendre qu'une vraie dérive
survienne, donc on rejoue les relevés d'une station atypique.

Pour lire les notebooks sur GitHub, passez par les fichiers dans `Notebooks/md/`. Le rendu des
`.ipynb` échoue régulièrement, ce n'est pas nos fichiers, c'est leur moteur.

---

## 6. Ce qu'il reste

Par ordre d'importance pour la note :

1. **Le README à la racine du dépôt n'existe toujours pas.** C'est la première chose que voit
   un correcteur. Il faut un démarrage rapide, un schéma d'architecture et l'état par phase.
2. **Nos documents d'analyse ne sont pas dans le dépôt.** Une dizaine de fichiers de réflexion
   (choix de modèle, architecture, revue de projet) sont dans un dossier ignoré par git. Tout
   le raisonnement de conception est donc invisible pour l'évaluation. Il faut en verser une
   version propre dans `Docs/`.
3. **La protection de la route `/reload`.** N'importe qui peut aujourd'hui forcer le
   rechargement du modèle depuis l'adresse publique. Un en-tête secret suffirait.
4. **Le garde-fou de promotion ne sert à rien.** Le seuil minimum de rappel pour promouvoir un
   modèle est à zéro, donc n'importe quelle version devient la version de production, aussi
   mauvaise soit-elle.
5. Le seuil de décision de l'API est resté à 0,5 alors que notre propre analyse recommande 0,34.
6. Les conteneurs tournent en root, sauf celui de la dérive.
7. Basculer la production sur le gradient boosting, qui est meilleur et huit fois plus rapide.

Les points 1 et 2 sont ceux qui coûtent le plus cher en note pour le moins d'effort. Les points
3 et 4 sont rapides et font bonne impression sur la partie industrialisation.

---

*Questions, ou si quelque chose n'est pas clair : demandez. Les scripts de déploiement et de
configuration sont tous dans `deployement/`, ils sont commentés.*
