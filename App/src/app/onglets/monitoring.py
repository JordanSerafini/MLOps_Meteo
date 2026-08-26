"""Partie 4 — Monitoring, détection de dérive et maintenance (phase 4).

Support projeté pendant un passage de cinq minutes. L'écran porte les schémas, les tableaux
et les chiffres mesurés ; l'explication est dite, pas écrite. Un paragraphe affiché est un
paragraphe que le jury lit au lieu d'écouter — les commentaires longs vivent donc dans les
notes de l'orateur, pas ici.

Les chiffres affichés sont les mesures de `Docs/CALIBRATION_DRIFT.md`, figées dans
`artefacts/calibration_drift.json`. C'est la partie du projet où l'on a le plus appris,
parce qu'on s'est d'abord trompés.
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
    schema, texte = st.columns([3, 2])
    schema.graphviz_chart(FLUX)
    texte.markdown(
        "**L'exactitude n'est mesurable qu'avec un jour de retard.** On surveille donc trois "
        "choses observables tout de suite :"
    )
    texte.markdown(
        "- la **santé du service** — `/metrics`, scruté toutes les 15 s\n"
        "- les **décisions rendues** — compteur et histogramme des probabilités\n"
        "- les **données reçues** — journal JSONL, une ligne par prédiction"
    )
    texte.caption(
        "Deux chemins arrivent dans Prometheus : le job de dérive est éphémère, d'où le "
        "pushgateway. Jauge `rain_model_loaded` ajoutée à l'API — Prometheus ne lit pas "
        "`/health`."
    )


def le_faux_positif():
    donnees = artefacts.charger("calibration_drift")
    if not donnees:
        artefacts.signaler_absent("calibration_drift")
        return
    essai = donnees["premier_essai"]
    second = donnees["second_essai"]
    st.subheader("Notre première version annonçait une dérive permanente")

    c1, c2 = st.columns([2, 3])
    df = pd.DataFrame(essai["colonnes"]).rename(columns={
        "colonne": "Colonne", "distance": "Distance", "seuil": "Seuil de la colonne",
        "derive": "Déclarée en dérive",
    })
    c1.dataframe(df.style.format({c: lambda v: nb(v, 3) for c in ["Distance", "Seuil de la colonne"]}),
                 use_container_width=True, hide_index=True)
    c1.caption(
        f"Trafic normal : {essai['n_observations']} observations tirées du jeu d'entraînement "
        f"lui-même, comparées à une référence de {ent(donnees['taille_reference'])}."
    )

    c2.markdown("**Deux erreurs, pas une**")
    m1, m2 = c2.columns(2)
    m1.metric("Colonnes en dérive", pct(essai["part_derive"], 0),
              help="3 colonnes sur 9, dépassant le seuil de très peu")
    m1.caption(f"seuil de déclenchement posé à {nb(essai['seuil_initial'])}, au feeling")
    m2.metric("Sur un essai plus court", pct(second["part_derive"], 0),
              help=f"{second['n_observations']} observations")
    m2.caption(f"le job concluait dès {second['min_samples_initial']} observations")
    c2.markdown(
        "Dans les deux cas : **réentraînement recommandé sur les données d'entraînement "
        "elles-mêmes.** Pas de la dérive, du bruit d'échantillonnage."
    )
    c2.info(
        "Une bibliothèque de détection de dérive branchée sans calibration ne détecte pas la "
        "dérive : elle produit du bruit.",
        icon="🚨",
    )


def calibration():
    donnees = artefacts.charger("calibration_drift")
    if not donnees:
        return
    st.subheader("Alors on a mesuré, deux fois")

    dist = donnees["distribution"]
    c1, c2 = st.columns([3, 2])
    c1.markdown("**Premier réglage — à partir de quelle part de colonnes parle-t-on de dérive ?**")
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
        "chacun. Part de colonnes déclarées en dérive."
    )
    c2.markdown("&nbsp;")
    c2.metric("Seuil retenu", nb(donnees["seuil_retenu"], 1),
              help="au milieu de l'intervalle qui sépare les deux cas")
    c2.caption(
        f"Les deux distributions ne se recouvrent pas : tout seuil entre "
        f"{nb(dist['intervalle_valide'][0])} et {nb(dist['intervalle_valide'][1])} sépare les "
        "deux situations. Aucun faux positif, aucun faux négatif sur trente tirages."
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
    gauche.caption(
        f"Trait horizontal : seuil retenu ({nb(donnees['seuil_retenu'], 1)}). Trait vertical "
        f"vert : minimum d'observations exigé ({donnees['min_samples']}). Sous 200 observations "
        "les deux courbes se confondent."
    )
    droite.markdown("&nbsp;")
    droite.metric("Minimum d'observations", donnees["min_samples"])
    droite.success(
        "Sous ce volume, le job renvoie `donnees_insuffisantes` et le DAG s'arrête en "
        "`skipped`. Mieux vaut une absence de réponse qu'une réponse fausse.",
        icon="✅",
    )

    verif = donnees["verification_finale"]
    st.markdown("**Vérification après correction**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dérive — trafic normal", pct(verif["normal"]["part_derive"], 0))
    c2.metric("Dérive — trafic Portland", pct(verif["portland"]["part_derive"], 0))
    c3.metric("Taux de pluie prédit — normal", pct(verif["normal"]["taux_pluie_predit"], 0))
    c4.metric("Taux de pluie prédit — Portland", pct(verif["portland"]["taux_pluie_predit"], 0))
    st.caption(
        f"{verif['normal']['n']} prédictions de chaque type. Portland : "
        f"{pct(donnees['station_decalee']['taux_pluie_pct'], 1, sur_cent=True)} de jours pluvieux "
        f"contre {pct(donnees['station_decalee']['moyenne_dataset_pct'], 1, sur_cent=True)} en "
        "moyenne sur le jeu de données."
    )


def alertes():
    donnees = artefacts.charger("calibration_drift")
    if not donnees:
        return
    st.subheader("Six alertes, dont une qui surveille la surveillance")
    c1, c2 = st.columns([3, 2])
    df = pd.DataFrame(donnees["alertes"]).rename(columns={
        "nom": "Règle", "declencheur": "Se déclenche quand", "delai": "Délai",
        "severite": "Sévérité",
    })
    c1.dataframe(df[["Règle", "Se déclenche quand", "Délai", "Sévérité"]],
                 use_container_width=True, hide_index=True)
    c1.caption(
        "Versionnées dans `App/monitoring/rules/alerts.yml`, plus une règle d'inhibition : "
        "quand l'API est injoignable, les alertes de modèle non chargé et de latence sont "
        "étouffées."
    )
    c2.markdown(
        "**`JobDeriveMuet`**  \n"
        "Un job de dérive planté laisse ses dernières valeurs dans Prometheus. Le tableau de "
        "bord reste vert et le système paraît sain, alors que plus personne ne surveille."
    )
    c2.success(
        "Vérifié, pas seulement configuré : conteneur d'API arrêté, `firing` au bout de deux "
        "minutes, résolu seul au redémarrage.",
        icon="🔎",
    )


def boucle_de_mise_a_jour():
    st.subheader("La mise à jour : ce qui est automatique, et ce qui ne l'est pas")
    schema, texte = st.columns([3, 2])
    schema.graphviz_chart(BOUCLE)
    texte.markdown(
        "Tous les maillons sont automatisés **sauf un**, et c'est un choix : rendre le socket "
        "Docker à Airflow pour qu'il lance l'entraînement donnerait l'équivalent des droits "
        "administrateur sur la machine à un service exposé sur Internet. L'entraînement est "
        "d'ailleurs déporté par conception — le serveur n'a ni le jeu de données ni l'image "
        "d'entraînement."
    )
    texte.dataframe(
        [
            {"Mis à jour": "le modèle servi", "Mécanisme": "registry MLflow → POST /reload",
             "Automatique": "sauf la décision"},
            {"Mis à jour": "les cinq images", "Mécanisme": "CI à chaque poussée",
             "Automatique": "oui"},
            {"Mis à jour": "les dépendances", "Mécanisme": "versions épinglées",
             "Automatique": "non"},
            {"Mis à jour": "après incident", "Mécanisme": "restart: unless-stopped",
             "Automatique": "oui, vérifié"},
        ],
        use_container_width=True, hide_index=True,
    )


def limites():
    st.subheader("Ce que cette surveillance ne couvre pas")
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        "**Dérive géographique, donc brutale**  \n"
        "Une dérive saisonnière demanderait de suivre une tendance, pas un seuil instantané."
    )
    c2.markdown(
        "**Pas de mise à jour des dépendances**  \n"
        "Ni Dependabot ni Renovate : les versions épinglées protègent de la surprise, pas de la "
        "vulnérabilité connue."
    )
    c3.markdown(
        "**La réalité reste hors d'atteinte**  \n"
        "Mesurer si les prédictions étaient justes demanderait les relevés du lendemain."
    )
    with st.expander("Reproduire la démonstration en local"):
        st.code(
            "make up              # MLflow, API, Streamlit\n"
            "make monitoring-up   # Prometheus, Grafana, Alertmanager, pushgateway\n"
            "make trafic N=500    # 500 prédictions vers l'API\n"
            "make drift           # compare le trafic reçu à la référence d'entraînement\n"
            "make drift-demo      # injecte du trafic décalé, puis mesure",
            language="bash",
        )
