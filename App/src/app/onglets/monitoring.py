"""Partie 4 — Monitoring, détection de dérive et alertes (phase 4).

Le cœur de cet onglet est la calibration du seuil de dérive : les tableaux affichés sont les
mesures de `Docs/CALIBRATION_DRIFT.md`, figées dans `artefacts/calibration_drift.json`.
C'est la partie du projet où l'on a le plus appris, parce qu'on s'est d'abord trompés.
"""
import altair as alt
import pandas as pd
import streamlit as st

import artefacts
from formats import ent, nb, pct


def afficher():
    quoi_mesurer()
    st.divider()
    le_faux_positif()
    st.divider()
    calibration()
    st.divider()
    alertes()
    st.divider()
    maintenance()


def quoi_mesurer():
    st.subheader("Ce qu'on surveille, et pourquoi")
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        "**La santé du service**\n\n"
        "Prometheus interroge `/metrics` sur l'API : débit, latence par centile, taux d'erreur. "
        "Un tableau de bord Grafana les affiche.\n\n"
        "Une jauge maison a été ajoutée, `rain_model_loaded`. `/health` disait déjà si un modèle "
        "était chargé, mais Prometheus ne lit pas `/health` — sans cette jauge, impossible "
        "d'alerter sur une API debout mais incapable de prédire."
    )
    c2.markdown(
        "**Le comportement du modèle**\n\n"
        "Un compteur des décisions (pluie / sec) et un histogramme des probabilités prédites. "
        "Un modèle qui se met à répondre « pluie » deux fois plus souvent qu'hier est un signal, "
        "même sans connaître la vérité terrain.\n\n"
        "Et c'est bien le problème du cas réel : la vérité terrain n'arrive que le lendemain. "
        "On surveille donc les **entrées** et la **distribution des sorties**, pas l'exactitude."
    )
    c3.markdown(
        "**Les données reçues**\n\n"
        "Chaque prédiction est écrite dans un journal JSONL : horodatage, identifiant de requête, "
        "données reçues, probabilité, décision, version du modèle. Ce journal est le jeu de "
        "données « courant » comparé à la référence d'entraînement.\n\n"
        "Si l'écriture échoue, la prédiction est servie quand même : un problème de journal ne "
        "doit jamais faire tomber le service."
    )
    st.caption(
        "Deux tableaux de bord Grafana sont provisionnés au démarrage sous forme de fichiers "
        "versionnés — personne n'a à les refaire à la main, et leurs treize requêtes ont été "
        "vérifiées une par une comme renvoyant réellement des données."
    )


def le_faux_positif():
    donnees = artefacts.charger("calibration_drift")
    if not donnees:
        artefacts.signaler_absent("calibration_drift")
        return
    st.subheader("La première version du job annonçait une dérive permanente")
    essai = donnees["premier_essai"]
    st.error(
        "Sur du trafic strictement normal — des observations tirées du jeu d'entraînement "
        "lui-même — le job concluait à une dérive et recommandait un réentraînement. "
        "Autrement dit : un réentraînement en boucle, sur des données identiques à celles "
        "d'origine."
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
        "colonnes sur neuf dépassaient ce seuil, de très peu.\n\n"
        f"Ce n'était pas de la dérive, c'était du **bruit d'échantillonnage** : comparer "
        f"{essai['n_observations']} observations à {ent(donnees['taille_reference'])} produit "
        "mécaniquement des écarts. Notre seuil de déclenchement, lui, avait été choisi au feeling."
    )
    st.markdown(
        "C'est le passage le plus instructif du projet : une bibliothèque de détection de dérive "
        "branchée sans calibration ne détecte pas la dérive, elle produit du bruit. Et un système "
        "d'alerte qui crie en permanence est un système d'alerte que plus personne ne lit."
    )


def calibration():
    donnees = artefacts.charger("calibration_drift")
    if not donnees:
        return
    st.subheader("Alors on a mesuré")

    dist = donnees["distribution"]
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
        f"Les deux distributions **ne se recouvrent pas** : le pire cas normal est à "
        f"{nb(dist['normal']['max'], 3)}, le meilleur cas décalé à "
        f"{nb(dist['decale']['min'], 3)}. N'importe quelle valeur entre "
        f"{nb(dist['intervalle_valide'][0])} et {nb(dist['intervalle_valide'][1])} sépare "
        "correctement les deux situations.\n\n"
        f"Nous retenons **{nb(donnees['seuil_retenu'])}**, à peu près au milieu de l'intervalle. Sur "
        f"ces trente tirages : aucun faux positif, aucun faux négatif."
    )

    st.markdown("**Mais un seuil ne vaut que pour une taille d'échantillon**")
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
    ).properties(height=320))
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
        "l'autre."
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
    with st.expander("Ce que cette calibration ne couvre pas"):
        st.markdown(
            "- La dérive testée est **géographique**, donc brutale. Une dérive saisonnière, "
            "progressive, produirait des valeurs intermédiaires et demanderait de suivre une "
            "tendance plutôt qu'un seuil instantané.\n"
            "- La calibration vaut pour ces neuf colonnes et cette référence de 5 000 lignes. "
            "Changer l'un ou l'autre impose de refaire la mesure.\n"
            "- Le jeu de données s'arrête en 2017 : nous ne pouvons pas valider sur une vraie "
            "dérive de production. C'est la raison pour laquelle nous rejouons les relevés d'une "
            "station atypique plutôt que d'inventer des données."
        )


def alertes():
    donnees = artefacts.charger("calibration_drift")
    if not donnees:
        return
    st.subheader("Un système d'alertes, pas des lignes dans un journal")
    st.markdown(
        "Le cadrage demandait un « système d'alertes » et il n'y en avait aucun : le DAG écrivait "
        "dans les logs Airflow, que personne ne lit. Alertmanager a été ajouté, avec six règles "
        "versionnées dans le dépôt."
    )
    df = pd.DataFrame(donnees["alertes"]).rename(columns={
        "nom": "Alerte", "declencheur": "Se déclenche quand", "delai": "Délai",
    })
    st.dataframe(df, use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    c1.markdown(
        "**La dernière règle est celle dont nous sommes le plus contents**\n\n"
        "Un job de dérive planté laisse les anciennes valeurs en place dans Prometheus. Le "
        "tableau de bord reste vert, les seuils ne sont pas franchis, et le système **paraît "
        "sain alors que plus personne ne surveille**. Sans `DriftJobStale`, on aurait une "
        "surveillance capable de s'arrêter sans prévenir."
    )
    c2.markdown(
        "**Vérifié, pas seulement configuré**\n\n"
        "En arrêtant le conteneur de l'API, l'alerte est passée en `firing` au bout de deux "
        "minutes, a bien été transmise à Alertmanager, et s'est résolue seule au redémarrage. "
        "L'alerte de dérive s'est déclenchée avec son message complet : « 77.78 % des colonnes "
        "suivies ont dérivé, au-delà du seuil de 50 % ».\n\n"
        "La destination est réglée par `ALERT_WEBHOOK_URL` — chacun met la sienne, rien de "
        "personnel n'est versionné."
    )


def maintenance():
    st.subheader("Reproduire la démonstration")
    st.code(
        "make up              # MLflow, API, Streamlit\n"
        "make monitoring-up   # Prometheus, Grafana, Alertmanager, pushgateway\n"
        "make trafic N=500    # 500 prédictions vers l'API\n"
        "make drift           # compare le trafic reçu à la référence d'entraînement\n"
        "make drift-demo      # injecte du trafic décalé, puis mesure : la dérive se déclenche",
        language="bash",
    )
    st.markdown(
        "`make drift-demo` provoque une dérive à la demande. Comme le jeu de données s'arrête en "
        "2017, on ne peut pas attendre qu'une vraie dérive survienne : on rejoue les relevés "
        "d'une station atypique. Le tableau de bord Grafana passe au rouge et l'alerte remonte "
        "dans Alertmanager au bout de cinq minutes.\n\n"
        "Le job produit trois sorties : un rapport HTML détaillé pour l'analyse humaine, un "
        "résumé JSON exploitable par le DAG, et cinq métriques poussées vers Prometheus via un "
        "pushgateway — nécessaire parce que le job est éphémère et que Prometheus ne peut pas "
        "interroger un conteneur déjà terminé."
    )
