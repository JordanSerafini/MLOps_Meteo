"""Partie 4 — Monitoring, détection de dérive et maintenance (phase 4).

Support d'un passage de cinq minutes. L'écran porte les schémas, les tableaux et les
chiffres mesurés ; le reste est dit. Les paragraphes sont donc courts par construction :
une page qu'on lit à voix haute est une page que le jury lit à notre place.

Le cœur est la calibration du seuil de dérive : les tableaux affichés sont les mesures de
`Docs/CALIBRATION_DRIFT.md`, figées dans `artefacts/calibration_drift.json`. C'est la partie
du projet où l'on a le plus appris, parce qu'on s'est d'abord trompés.
"""
import altair as alt
import pandas as pd
import streamlit as st

import artefacts
from formats import ent, nb, pct

PLAN = [
    "Surveiller quoi, quand la vérité terrain arrive demain",
    "Le détecteur annonçait 78 % de dérive sur des données normales",
    "Deux réglages, mesurés au lieu d'être devinés",
    "Six alertes, dont une qui surveille la surveillance",
    "La mise à jour : ce qui est automatique et ce qui ne l'est pas",
]

# Le flux d'observabilité, et lui seul : l'architecture générale des neuf services est
# présentée dans la partie 3, la redessiner ici ferait doublon. Disposition verticale pour
# la même raison que là-bas — Streamlit contraint la largeur, pas la hauteur.
FLUX = """
digraph observabilite {
  rankdir=TB;
  nodesep=0.40; ranksep=0.45; fontname="Helvetica";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=14,
        color="#bbbbbb", margin="0.22,0.13"];
  edge [fontname="Helvetica", fontsize=11, color="#777777"];

  api  [label="API FastAPI — POST /predict", fillcolor="#cfe2ff"];

  met  [label="GET /metrics\\ndébit, latence, codes HTTP,\\ncompteur de décisions", fillcolor="#cfe2ff"];
  jrn  [label="journal JSONL\\nune ligne par prédiction", fillcolor="#cfe2ff"];
  ref  [label="reference.csv\\néchantillon figé, versionné", fillcolor="#fff3cd"];

  job  [label="job dérive — Evidently\\nconteneur éphémère", fillcolor="#f8d7da"];
  push [label="Pushgateway", fillcolor="#f8d7da"];
  prom [label="Prometheus\\n+ 6 règles d'alerte", fillcolor="#f8d7da"];

  graf [label="Grafana\\n2 tableaux de bord provisionnés", fillcolor="#d1e7dd"];
  am   [label="Alertmanager → webhook", fillcolor="#d1e7dd"];
  dag  [label="DAG drift_check — chaque matin 6 h", fillcolor="#e2d9f3"];

  // Rangs forcés : les trois sources sur une ligne, les trois collecteurs sur la
  // suivante. Sans ça graphviz empile dix niveaux et le schéma sort de l'écran.
  {rank=same; met; jrn; ref;}
  {rank=same; push; dag;}
  {rank=same; graf; am;}

  api  -> met  [label=" temps réel"];
  api  -> jrn  [label=" features, proba, version"];
  jrn  -> job  [label=" jeu courant"];
  ref  -> job  [label=" jeu de référence"];
  job  -> push [label=" 5 métriques"];
  job  -> dag  [label=" résumé JSON"];
  met  -> prom [label=" scrutin 15 s"];
  push -> prom [label=" scrutin 15 s"];
  prom -> graf;
  prom -> am   [label=" règle franchie"];
}
"""

# La boucle de maintenance. Le nœud gris est le sujet de la section : la décision reste humaine.
BOUCLE = """
digraph maj {
  rankdir=TB;
  nodesep=0.35; ranksep=0.38; fontname="Helvetica";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=14,
        color="#bbbbbb", margin="0.22,0.13"];
  edge [fontname="Helvetica", fontsize=11, color="#777777"];

  sig  [label="DAG drift_check — chaque matin 6 h
signale, et s'arrête là", fillcolor="#f8d7da"];
  dec  [label="décision humaine
le seul maillon non automatisé", fillcolor="#e9ecef",
        style="rounded,filled,bold", color="#666666"];
  mk   [label="make deploy-model — poste de dev
réentraîne sur les 145 460 relevés", fillcolor="#fff3cd"];
  reg  [label="MLflow Registry
nouvelle version + alias champion", fillcolor="#d1e7dd"];
  dp   [label="DAG deploy_pipeline sur la VM
POST /reload, portée admin", fillcolor="#e2d9f3"];
  sm   [label="smoke test + contrôle de model_loaded", fillcolor="#cfe2ff"];

  sig -> dec -> mk -> reg -> dp -> sm;
  sm -> sig [label=" le cycle reprend", style=dashed, constraint=false];
}
"""


def afficher():
    plan()
    st.divider()
    quoi_mesurer()
    st.divider()
    le_faux_positif()
    st.divider()
    calibration()
    st.divider()
    alertes()
    st.divider()
    boucle_de_mise_a_jour()
    st.divider()
    limites()


def plan():
    st.caption(" · ".join(f"**{i}.** {titre}" for i, titre in enumerate(PLAN, start=1)))


def quoi_mesurer():
    st.subheader("Surveiller quoi, quand la vérité terrain arrive demain")
    st.markdown(
        "Un modèle de prédiction météo pose un problème que n'ont pas les tableaux de bord "
        "habituels : **l'exactitude n'est mesurable qu'avec un jour de retard.** Impossible de "
        "surveiller ce qu'on voudrait surveiller. On suit donc trois choses observables "
        "immédiatement — la santé du service, la distribution des décisions rendues, et la "
        "distribution des données reçues."
    )
    schema, texte = st.columns([3, 2])
    schema.graphviz_chart(FLUX)
    texte.markdown(
        "**La santé du service**  \n"
        "Prometheus interroge `/metrics` toutes les quinze secondes : débit, latence par "
        "centile, taux d'erreur.\n\n"
        "**Le comportement du modèle**  \n"
        "Un compteur des décisions et un histogramme des probabilités. Un modèle qui répond "
        "« pluie » deux fois plus souvent qu'hier est un signal, même sans vérité terrain.\n\n"
        "**Les données reçues**  \n"
        "Chaque prédiction est écrite dans un journal JSONL. Ce journal *est* le jeu de données "
        "courant que le job de dérive compare à la référence d'entraînement.\n\n"
        "Deux chemins distincts arrivent dans Prometheus, et ce n'est pas un détail : le job de "
        "dérive est **éphémère**. Prometheus ne peut pas interroger un conteneur déjà terminé, "
        "d'où le pushgateway qui conserve ses valeurs entre deux exécutions."
    )
    texte.info(
        "Une jauge a été ajoutée à l'API, `rain_model_loaded`. `/health` disait déjà si un "
        "modèle était chargé, mais Prometheus ne lit pas `/health` : sans cette jauge, "
        "impossible d'alerter sur une API debout et pourtant incapable de prédire.",
        icon="🔎",
    )
    st.caption(
        "Les deux tableaux de bord Grafana sont des fichiers versionnés, provisionnés au "
        "démarrage : personne n'a à les refaire à la main, et leurs treize requêtes ont été "
        "vérifiées une par une comme renvoyant réellement des données. Si l'écriture du journal "
        "échoue, la prédiction est servie quand même — un problème de journalisation ne doit "
        "jamais faire tomber le service."
    )


def le_faux_positif():
    donnees = artefacts.charger("calibration_drift")
    if not donnees:
        artefacts.signaler_absent("calibration_drift")
        return
    essai = donnees["premier_essai"]
    second = donnees["second_essai"]
    st.subheader("Notre première version annonçait une dérive permanente")
    st.error(
        f"Sur du trafic strictement normal — {essai['n_observations']} observations tirées du jeu "
        "d'entraînement lui-même — le job concluait à la dérive et recommandait un "
        "réentraînement. Autrement dit : un réentraînement en boucle, sur les données d'origine.",
        icon="🚨",
    )
    c1, c2 = st.columns([2, 3])
    df = pd.DataFrame(essai["colonnes"]).rename(columns={
        "colonne": "Colonne", "distance": "Distance", "seuil": "Seuil de la colonne",
        "derive": "Déclarée en dérive",
    })
    c1.dataframe(df.style.format({c: lambda v: nb(v, 3) for c in ["Distance", "Seuil de la colonne"]}),
                 use_container_width=True, hide_index=True)
    c2.markdown(
        "Evidently teste **chaque colonne séparément** — distance de Wasserstein normalisée pour "
        "les numériques, Jensen-Shannon pour `Location` — avec un seuil par défaut de 0,1. Trois "
        "colonnes sur neuf dépassaient ce seuil, de très peu : "
        f"{pct(essai['part_derive'], 0)} des colonnes, quand notre seuil de déclenchement était "
        f"posé à {nb(essai['seuil_initial'])}, choisi au feeling.\n\n"
        f"Ce n'était pas de la dérive mais du **bruit d'échantillonnage** : comparer "
        f"{essai['n_observations']} observations à {ent(donnees['taille_reference'])} produit "
        "mécaniquement des écarts.\n\n"
        f"Une seconde erreur aggravait la première : le job acceptait de conclure dès "
        f"{second['min_samples_initial']} observations. Sur un essai à "
        f"{second['n_observations']} prédictions, il a annoncé "
        f"**{pct(second['part_derive'], 0)} de colonnes en dérive** — toujours sur du trafic "
        "normal.\n\n"
        "**Une bibliothèque de détection de dérive branchée sans calibration ne détecte pas la "
        "dérive : elle produit du bruit.** Et un système d'alerte qui crie en permanence est un "
        "système d'alerte que plus personne ne lit."
    )
    st.caption(
        "Deux erreurs, donc deux réglages à mesurer : la part de colonnes à partir de laquelle on "
        "parle de dérive, et le nombre minimum d'observations en dessous duquel on refuse de "
        "conclure."
    )


def calibration():
    donnees = artefacts.charger("calibration_drift")
    if not donnees:
        return
    st.subheader("Alors on a mesuré, deux fois")

    dist = donnees["distribution"]
    st.markdown("**Premier réglage — à partir de quelle part de colonnes parle-t-on de dérive ?**")
    c1, c2 = st.columns([3, 2])
    df_dist = pd.DataFrame([
        {"Trafic": "normal", "Minimum": dist["normal"]["min"], "Médiane": dist["normal"]["median"],
         "Maximum": dist["normal"]["max"]},
        {"Trafic": f"décalé ({donnees['station_decalee']['nom']})", "Minimum": dist["decale"]["min"],
         "Médiane": dist["decale"]["median"], "Maximum": dist["decale"]["max"]},
    ])
    c1.dataframe(df_dist.style.format({c: lambda v: nb(v, 3)
                                       for c in ["Minimum", "Médiane", "Maximum"]}),
                 use_container_width=True, hide_index=True)
    c1.caption(
        f"{dist['n_tirages_par_cas']} tirages par cas, {dist['n_observations']} observations "
        f"chacun. Part de colonnes déclarées en dérive."
    )
    c2.markdown(
        f"Les deux distributions **ne se recouvrent pas** : pire cas normal à "
        f"{nb(dist['normal']['max'], 3)}, meilleur cas décalé à {nb(dist['decale']['min'], 3)}. "
        f"N'importe quelle valeur entre {nb(dist['intervalle_valide'][0])} et "
        f"{nb(dist['intervalle_valide'][1])} sépare correctement les deux situations.\n\n"
        f"Nous retenons **{nb(donnees['seuil_retenu'])}**, au milieu de l'intervalle. Sur ces "
        "trente tirages : aucun faux positif, aucun faux négatif."
    )

    st.markdown("**Second réglage — un seuil ne vaut que pour une taille d'échantillon**")
    sensibilite = pd.DataFrame(donnees["sensibilite_volume"])
    lignes = []
    for _, r in sensibilite.iterrows():
        lignes.append({"Observations": r["n_observations"], "Part en dérive": r["normal_max"],
                       "Cas": "trafic normal (pire cas)"})
        lignes.append({"Observations": r["n_observations"], "Part en dérive": r["decale_min"],
                       "Cas": "trafic décalé (meilleur cas)"})
    # Domaine et couleurs fixés explicitement : le trafic décalé doit être le rouge (celui
    # qu'on veut voir déclencher), et l'échelle log étendrait sinon l'axe bien au-delà des
    # tailles réellement mesurées.
    tailles = sensibilite["n_observations"]
    graphique = (alt.Chart(pd.DataFrame(lignes)).mark_line(point=True).encode(
        x=alt.X("Observations:Q",
                scale=alt.Scale(type="log", domain=[tailles.min() * 0.8, tailles.max() * 1.25],
                                nice=False),
                title="nombre d'observations comparées (échelle log)"),
        y=alt.Y("Part en dérive:Q", scale=alt.Scale(domain=[0, 1]),
                title="part de colonnes déclarées en dérive"),
        color=alt.Color("Cas:N", legend=alt.Legend(orient="bottom", direction="vertical",
                                                  title=None),
                        scale=alt.Scale(domain=["trafic normal (pire cas)",
                                                "trafic décalé (meilleur cas)"],
                                        range=["#4c78a8", "#d1495b"])),
        tooltip=["Observations", "Cas", "Part en dérive"],
    ).properties(height=300))
    seuil = (alt.Chart(pd.DataFrame({"y": [donnees["seuil_retenu"]]}))
             .mark_rule(color="grey", strokeDash=[4, 4]).encode(y="y:Q"))
    minimum = (alt.Chart(pd.DataFrame({"x": [donnees["min_samples"]]}))
               .mark_rule(color="#2a9d8f", size=2).encode(x="x:Q"))

    gauche, droite = st.columns([3, 2])
    gauche.altair_chart(graphique + seuil + minimum, use_container_width=True)
    gauche.caption(f"Trait horizontal : seuil de déclenchement retenu "
                   f"({nb(donnees['seuil_retenu'])}). Trait vertical vert : minimum "
                   f"d'observations exigé ({donnees['min_samples']}).")
    droite.dataframe(
        sensibilite.rename(columns={
            "n_observations": "Obs.", "normal_max": "Normal (max)",
            "normal_median": "Normal (méd.)", "decale_min": "Décalé (min)",
            "exploitable": "Exploitable",
        }).style.format({c: lambda v: nb(v, 2)
                         for c in ["Normal (max)", "Normal (méd.)", "Décalé (min)"]}),
        use_container_width=True, hide_index=True,
    )
    droite.markdown(
        "**En dessous de 200 observations, le test répond « dérive » quoi qu'on lui donne** : les "
        "deux courbes se confondent. La séparation devient franche à 400, où le pire cas normal "
        "(0,33) reste loin du meilleur cas décalé (0,89)."
    )
    st.success(
        f"D'où `--min-samples {donnees['min_samples']}`. Sous ce volume, le job renvoie "
        "explicitement `statut: donnees_insuffisantes` et le DAG s'arrête en `skipped`. "
        "**Mieux vaut une absence de réponse qu'une réponse fausse** — dans un sens comme dans "
        "l'autre.",
        icon="✅",
    )

    verif = donnees["verification_finale"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dérive — trafic normal", pct(verif["normal"]["part_derive"], 0))
    c2.metric("Dérive — trafic Portland", pct(verif["portland"]["part_derive"], 0))
    c3.metric("Taux de pluie prédit — normal", pct(verif["normal"]["taux_pluie_predit"], 0))
    c4.metric("Taux de pluie prédit — Portland", pct(verif["portland"]["taux_pluie_predit"], 0))
    st.caption(
        f"Vérification après correction, sur {verif['normal']['n']} prédictions de chaque type. "
        "Le taux de pluie annoncé par le modèle monte de 23 % à 35 %, cohérent avec une station à "
        f"{pct(donnees['station_decalee']['taux_pluie_pct'], 1, sur_cent=True)} de jours pluvieux "
        f"contre {pct(donnees['station_decalee']['moyenne_dataset_pct'], 1, sur_cent=True)} en "
        "moyenne."
    )


def alertes():
    donnees = artefacts.charger("calibration_drift")
    if not donnees:
        return
    st.subheader("Six alertes, dont une qui surveille la surveillance")
    st.markdown(
        "Le cadrage demandait un « système d'alertes » et il n'y en avait aucun : le DAG écrivait "
        "dans les logs Airflow, que personne ne lit. Alertmanager a été ajouté, avec six règles "
        "versionnées dans `App/monitoring/rules/alerts.yml`."
    )
    df = pd.DataFrame(donnees["alertes"]).rename(columns={
        "nom": "Règle", "declencheur": "Se déclenche quand", "delai": "Délai",
        "severite": "Sévérité",
    })
    st.dataframe(df[["Règle", "Se déclenche quand", "Délai", "Sévérité"]],
                 use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    c1.markdown(
        "**`JobDeriveMuet` est la règle dont nous sommes le plus contents**\n\n"
        "Un job de dérive planté laisse ses dernières valeurs en place dans Prometheus. Le "
        "tableau de bord reste vert, aucun seuil n'est franchi, et le système **paraît sain "
        "alors que plus personne ne surveille**. Sans cette règle, nous aurions une surveillance "
        "capable de s'arrêter sans prévenir.\n\n"
        "Une règle d'inhibition complète l'ensemble : quand l'API est injoignable, les alertes de "
        "modèle non chargé et de latence sont étouffées. C'est la même panne, une notification "
        "suffit."
    )
    c2.markdown(
        "**Vérifié, pas seulement configuré**\n\n"
        "En arrêtant le conteneur de l'API, l'alerte est passée en `firing` au bout de deux "
        "minutes, a bien été transmise à Alertmanager, et s'est résolue seule au redémarrage. "
        "L'alerte de dérive s'est déclenchée avec son message complet : « 77.78 % des colonnes "
        "suivies ont dérivé, au-delà du seuil de 50 % ».\n\n"
        "La destination est réglée par `ALERT_WEBHOOK_URL` : chacun met la sienne, rien de "
        "personnel n'est versionné. Sans destination configurée, les alertes restent "
        "consultables dans l'interface Alertmanager."
    )


def boucle_de_mise_a_jour():
    st.subheader("La mise à jour : ce qui est automatique, et ce qui ne l'est pas")
    schema, texte = st.columns([3, 2])
    schema.graphviz_chart(BOUCLE)
    texte.markdown(
        "Tous les maillons de cette chaîne sont automatisés **sauf un**, et c'est un choix, pas "
        "un renoncement.\n\n"
        "Le socket Docker a été retiré du conteneur Airflow au moment du passage en production : "
        "le lui rendre pour qu'il lance l'entraînement reviendrait à lui donner l'équivalent des "
        "droits administrateur sur la machine. Un service exposé sur Internet ne doit pas pouvoir "
        "créer des conteneurs sur son hôte.\n\n"
        "Et l'entraînement est **déporté par conception** : le serveur n'a ni le jeu de données "
        "ni l'image d'entraînement, et n'a pas les ressources pour les héberger. Le DAG signale "
        "donc, et `make deploy-model` exécute."
    )
    texte.warning(
        "Pour aller jusqu'au déclenchement automatique, la solution propre serait un "
        "`docker-socket-proxy` limité à la création d'un seul conteneur, ou un worker Airflow "
        "dédié sur le poste de dev. C'est écrit dans le DAG lui-même, hors périmètre du projet.",
        icon="⚠️",
    )
    st.markdown(
        "La feuille de route demandait les mises à jour du modèle **et des composants**. Les "
        "secondes sont traitées autrement, par la chaîne d'intégration continue :"
    )
    st.dataframe(
        [
            {"Ce qui est mis à jour": "le modèle servi",
             "Déclencheur": "dérive détectée, ou nouvel entraînement",
             "Mécanisme": "trainer → registry MLflow → alias champion → POST /reload",
             "Automatique": "sauf la décision"},
            {"Ce qui est mis à jour": "les cinq images Docker",
             "Déclencheur": "chaque poussée sur une branche prenom_dev",
             "Mécanisme": "CI GitHub Actions : lint, 74 tests, validation des compose, build des 5 images",
             "Automatique": "oui"},
            {"Ce qui est mis à jour": "les dépendances Python",
             "Déclencheur": "revue manuelle",
             "Mécanisme": "versions épinglées dans requirements/*.txt",
             "Automatique": "non — voir les limites"},
            {"Ce qui est mis à jour": "les conteneurs après incident",
             "Déclencheur": "arrêt ou redémarrage de la VM",
             "Mécanisme": "restart: unless-stopped (Docker Compose)",
             "Automatique": "oui, vérifié sur un redémarrage réel"},
        ],
        use_container_width=True, hide_index=True,
    )


def limites():
    st.subheader("Ce que cette surveillance ne couvre pas")
    c1, c2 = st.columns(2)
    c1.markdown(
        "**Sur la détection**\n\n"
        "- La dérive testée est **géographique**, donc brutale. Une dérive saisonnière, "
        "progressive, produirait des valeurs intermédiaires et demanderait de suivre une "
        "tendance plutôt qu'un seuil instantané.\n"
        "- La calibration vaut pour ces neuf colonnes et cette référence de 5 000 lignes. "
        "Changer l'une ou l'autre impose de refaire la mesure.\n"
        "- Le jeu de données s'arrête en 2017 : nous ne pouvons pas valider sur une vraie dérive "
        "de production. D'où le choix de rejouer une station atypique plutôt que d'inventer des "
        "données."
    )
    c2.markdown(
        "**Sur la maintenance**\n\n"
        "- Aucune mise à jour automatique des dépendances (ni Dependabot ni Renovate). Les "
        "versions sont épinglées, ce qui protège de la surprise mais pas de la vulnérabilité "
        "connue.\n"
        "- La dérive du modèle **par rapport à la réalité** reste hors d'atteinte : mesurer si "
        "les prédictions étaient justes demanderait de récupérer les relevés du lendemain, ce "
        "que le projet ne fait pas.\n"
        "- Les alertes partent vers un webhook. Aucune astreinte, aucune escalade : c'est un "
        "projet d'école, pas une équipe d'exploitation."
    )
    with st.expander("Reproduire la démonstration en local"):
        st.code(
            "make up              # MLflow, API, Streamlit\n"
            "make monitoring-up   # Prometheus, Grafana, Alertmanager, pushgateway\n"
            "make trafic N=500    # 500 prédictions vers l'API\n"
            "make drift           # compare le trafic reçu à la référence d'entraînement\n"
            "make drift-demo      # injecte du trafic décalé, puis mesure : la dérive se déclenche",
            language="bash",
        )
        st.markdown(
            "`make drift-demo` provoque une dérive à la demande. Le job produit trois sorties : "
            "un rapport HTML détaillé pour l'analyse humaine, un résumé JSON exploitable par le "
            "DAG, et cinq métriques poussées vers Prometheus. Le tableau de bord Grafana passe "
            "au rouge et l'alerte remonte dans Alertmanager au bout de cinq minutes."
        )
