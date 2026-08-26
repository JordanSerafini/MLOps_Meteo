"""Partie 4 : monitoring, détection de dérive et maintenance (phase 4).

Support projeté pendant un passage de cinq minutes. Règle de construction : par section, un
titre qui porte le message, un visuel, et au plus une ligne de légende. Rien de ce qui est
dit à l'oral n'est écrit ici : le jury regarde, il ne lit pas.

Le plan suit les trois points d'attention du cadrage de la phase 4 : santé du système,
qualité des données, performance du modèle. Les chiffres viennent de
`Docs/CALIBRATION_DRIFT.md`, figés dans `artefacts/calibration_drift.json` ; les captures
d'interfaces sont dans `artefacts/captures/`.
"""
import altair as alt
import pandas as pd
import streamlit as st

import artefacts
from formats import nb, pct

# Le flux d'observabilité, et lui seul : l'architecture générale des services est présentée
# dans la partie 3. Disposition verticale, car Streamlit contraint la largeur.
FLUX = """
digraph observabilite {
  rankdir=TB;
  nodesep=0.40; ranksep=0.45; fontname="Helvetica";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=14,
        color="#bbbbbb", margin="0.22,0.13"];
  edge [fontname="Helvetica", fontsize=11, color="#777777"];

  api  [label="API FastAPI : POST /predict", fillcolor="#cfe2ff"];

  met  [label="GET /metrics\\ndébit, latence, codes HTTP,\\ncompteur de décisions", fillcolor="#cfe2ff"];
  jrn  [label="journal JSONL\\nune ligne par prédiction", fillcolor="#cfe2ff"];
  ref  [label="reference.csv\\néchantillon figé, versionné", fillcolor="#fff3cd"];

  job  [label="job dérive Evidently\\nconteneur éphémère", fillcolor="#f8d7da"];
  push [label="Pushgateway", fillcolor="#f8d7da"];
  prom [label="Prometheus\\n+ 6 règles d'alerte", fillcolor="#f8d7da"];

  graf [label="Grafana\\n2 tableaux de bord provisionnés", fillcolor="#d1e7dd"];
  am   [label="Alertmanager → webhook", fillcolor="#d1e7dd"];
  dag  [label="DAG drift_check, chaque matin 6 h", fillcolor="#e2d9f3"];

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
  rankdir=LR;
  nodesep=0.30; ranksep=0.34; fontname="Helvetica";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=13,
        color="#bbbbbb", margin="0.18,0.11"];
  edge [fontname="Helvetica", fontsize=10, color="#777777"];

  sig  [label="DAG drift_check\\nchaque matin 6 h", fillcolor="#f8d7da"];
  dec  [label="décision\\nhumaine", fillcolor="#e9ecef",
        style="rounded,filled,bold", color="#666666"];
  mk   [label="make deploy-model\\nposte de dev", fillcolor="#fff3cd"];
  reg  [label="MLflow Registry\\nalias champion", fillcolor="#d1e7dd"];
  dp   [label="DAG deploy\\nPOST /reload", fillcolor="#e2d9f3"];
  sm   [label="smoke test", fillcolor="#cfe2ff"];

  sig -> dec -> mk -> reg -> dp -> sm;
  sm -> sig [style=dashed, constraint=false];
}
"""


def afficher():
    quoi_surveiller()
    st.divider()
    sante_du_systeme()
    st.divider()
    qualite_des_donnees()
    st.divider()
    alertes()
    st.divider()
    reentrainement()
    st.divider()
    limites()


def quoi_surveiller():
    st.subheader("Trois choses à surveiller, une seule impossible en direct")
    schema, texte = st.columns([3, 2])
    schema.graphviz_chart(FLUX)
    texte.markdown(
        "- **santé du système** : `/metrics`, scruté toutes les 15 s\n"
        "- **qualité des données** : le journal des prédictions contre une référence figée\n"
        "- **performance du modèle** : mesurable seulement le lendemain"
    )
    texte.caption("Le job de dérive est éphémère, d'où le pushgateway.")


def sante_du_systeme():
    st.subheader("Santé du système")
    image = artefacts.capture("grafana-api-sante")
    if image:
        st.image(image, use_container_width=True)
    st.caption("Grafana, tableau de bord provisionné au démarrage. Débit, latence par centile, "
               "taux d'erreur, codes de réponse.")


def qualite_des_donnees():
    donnees = artefacts.charger("calibration_drift")
    if not donnees:
        artefacts.signaler_absent("calibration_drift")
        return
    essai = donnees["premier_essai"]
    second = donnees["second_essai"]
    dist = donnees["distribution"]

    st.subheader("Qualité des données : un détecteur qui criait au loup")
    c1, c2 = st.columns([3, 2])
    df = pd.DataFrame(essai["colonnes"]).rename(columns={
        "colonne": "Colonne", "distance": "Distance", "seuil": "Seuil de la colonne",
        "derive": "En dérive",
    })
    c1.dataframe(df.style.format({c: lambda v: nb(v, 3) for c in ["Distance", "Seuil de la colonne"]}),
                 use_container_width=True, hide_index=True)
    c1.caption(f"Trafic normal, {essai['n_observations']} observations tirées du jeu "
               "d'entraînement lui-même.")
    m1, m2 = c2.columns(2)
    m1.metric("Colonnes en dérive", pct(essai["part_derive"], 0),
              help=f"seuil de déclenchement alors posé à {nb(essai['seuil_initial'])}")
    m2.metric("Sur un essai plus court", pct(second["part_derive"], 0),
              help=f"{second['n_observations']} observations, aucun minimum exigé")
    c2.caption("Deux erreurs : le seuil, et l'absence de volume minimum.")

    st.markdown("**Deux réglages, mesurés au lieu d'être devinés**")
    g1, g2, g3 = st.columns([2, 3, 2])
    g1.dataframe(
        pd.DataFrame([
            {"Trafic": "normal", "Min": dist["normal"]["min"], "Méd.": dist["normal"]["median"],
             "Max": dist["normal"]["max"]},
            {"Trafic": donnees["station_decalee"]["nom"], "Min": dist["decale"]["min"],
             "Méd.": dist["decale"]["median"], "Max": dist["decale"]["max"]},
        ]).style.format({c: lambda v: nb(v, 3) for c in ["Min", "Méd.", "Max"]}),
        use_container_width=True, hide_index=True,
    )
    g1.caption(f"{dist['n_tirages_par_cas']} tirages par cas. Les deux distributions ne se "
               "recouvrent pas.")
    g2.altair_chart(_graphique_volume(donnees), use_container_width=True)
    g3.metric("Seuil retenu", nb(donnees["seuil_retenu"], 1))
    g3.metric("Minimum d'observations", donnees["min_samples"])
    g3.caption("En dessous, le job répond `donnees_insuffisantes` et le DAG s'arrête.")

    image = artefacts.capture("grafana-predictions-derive")
    if image:
        st.image(image, use_container_width=True)
        verif = donnees["verification_finale"]
        st.caption(
            f"Après bascule sur {donnees['station_decalee']['nom']} : dérive détectée, "
            f"{pct(verif['portland']['part_derive'], 0)} des colonnes contre "
            f"{pct(verif['normal']['part_derive'], 0)} en trafic normal, et la part de pluie "
            "prédite qui monte."
        )


def _graphique_volume(donnees):
    """Part de colonnes en dérive selon la taille de l'échantillon comparé."""
    sensibilite = pd.DataFrame(donnees["sensibilite_volume"])
    lignes = []
    for _, r in sensibilite.iterrows():
        lignes.append({"Observations": r["n_observations"], "Part en dérive": r["normal_max"],
                       "Cas": "normal (pire cas)"})
        lignes.append({"Observations": r["n_observations"], "Part en dérive": r["decale_min"],
                       "Cas": "décalé (meilleur cas)"})
    # Domaine et couleurs fixés : le trafic décalé doit être le rouge, et l'échelle log
    # étendrait sinon l'axe au-delà des tailles réellement mesurées.
    tailles = sensibilite["n_observations"]
    courbes = alt.Chart(pd.DataFrame(lignes)).mark_line(point=True).encode(
        x=alt.X("Observations:Q",
                scale=alt.Scale(type="log", domain=[tailles.min() * 0.8, tailles.max() * 1.25],
                                nice=False),
                title="observations comparées (échelle log)"),
        y=alt.Y("Part en dérive:Q", scale=alt.Scale(domain=[0, 1]),
                title="part de colonnes en dérive"),
        color=alt.Color("Cas:N", legend=alt.Legend(orient="bottom", direction="vertical",
                                                  title=None),
                        scale=alt.Scale(domain=["normal (pire cas)", "décalé (meilleur cas)"],
                                        range=["#4c78a8", "#d1495b"])),
        tooltip=["Observations", "Cas", "Part en dérive"],
    ).properties(height=260)
    seuil = (alt.Chart(pd.DataFrame({"y": [donnees["seuil_retenu"]]}))
             .mark_rule(color="grey", strokeDash=[4, 4]).encode(y="y:Q"))
    minimum = (alt.Chart(pd.DataFrame({"x": [donnees["min_samples"]]}))
               .mark_rule(color="#2a9d8f", size=2).encode(x="x:Q"))
    return courbes + seuil + minimum


def alertes():
    donnees = artefacts.charger("calibration_drift")
    if not donnees:
        return
    st.subheader("Six alertes, dont une qui surveille la surveillance")
    df = pd.DataFrame(donnees["alertes"]).rename(columns={
        "nom": "Règle", "declencheur": "Se déclenche quand", "delai": "Délai",
        "severite": "Sévérité",
    })
    st.dataframe(df[["Règle", "Se déclenche quand", "Délai", "Sévérité"]],
                 use_container_width=True, hide_index=True)
    st.caption("`JobDeriveMuet` : un job de dérive planté laisse ses dernières valeurs en "
               "place, le tableau de bord reste vert.")
    image = artefacts.capture("alertes-firing")
    if image:
        st.image(image, use_container_width=True)
        st.caption("Relevé réel : la règle de dérive a déclenché après cinq minutes et a été "
                   "routée vers Alertmanager.")


def reentrainement():
    st.subheader("Réentraînement : automatisé, sauf la décision")
    st.graphviz_chart(BOUCLE)
    c1, c2 = st.columns([2, 3])
    c1.caption("Le socket Docker a été retiré à Airflow au passage en production.")
    c2.dataframe(
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
    c1.markdown("**Dérive géographique, donc brutale**")
    c1.caption("Une dérive saisonnière demanderait une tendance, pas un seuil.")
    c2.markdown("**Dépendances non suivies**")
    c2.caption("Versions épinglées, ni Dependabot ni Renovate.")
    c3.markdown("**La réalité hors d'atteinte**")
    c3.caption("Il faudrait les relevés du lendemain.")
    with st.expander("Reproduire en local"):
        st.code(
            "make up              # MLflow, API, Streamlit\n"
            "make monitoring-up   # Prometheus, Grafana, Alertmanager, pushgateway\n"
            "make trafic N=500    # 500 prédictions vers l'API\n"
            "make drift-demo      # injecte du trafic décalé, puis mesure",
            language="bash",
        )
