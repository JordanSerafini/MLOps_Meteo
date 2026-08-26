"""Partie 2 — Modélisation, suivi d'expériences et versionnement (phase 2).

"""
import altair as alt
import pandas as pd
import streamlit as st

import artefacts
from formats import ent, nb, signe

COLONNES = {
    "modele": "Modèle", "accuracy": "Accuracy", "precision": "Précision (pluie)",
    "rappel": "Rappel (pluie)", "f1": "F1 (pluie)", "roc_auc": "ROC-AUC",
    "secondes_entrainement": "Entraînement (s)",
}
SCORES = ["Accuracy", "Précision (pluie)", "Rappel (pluie)", "F1 (pluie)", "ROC-AUC"]
LEGENDE = alt.Legend(orient="bottom", direction="vertical", title=None, labelLimit=300)


def afficher():
    comparatif()
    st.divider()
    courbes()
    st.divider()
    seuil_interactif()
    st.divider()
    suivi_mlflow()
    st.divider()
    versionnement_dvc()


# ── Modélisation (compact) ─────────────────────────────────────────────

def comparatif():
    donnees = artefacts.charger("modeles")
    if not donnees:
        artefacts.signaler_absent("modeles")
        return
    st.subheader("Comparaison des modèles")
    df = pd.DataFrame(donnees["lignes"]).rename(columns=COLONNES)[list(COLONNES.values())]
    for colonne in SCORES:
        df[colonne] = df[colonne].map(lambda v: "—" if pd.isna(v) else nb(v, 4))
    df["Entraînement (s)"] = df["Entraînement (s)"].map(lambda v: nb(v, 1))
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(
        f"Évaluation sur {ent(donnees['n_test'])} observations de test "
        f"({nb(donnees['taux_pluie_test'] * 100, 1)} % de pluie), seuil à "
        f"{nb(donnees['seuil_evalue'])}."
    )


def courbes():
    donnees = artefacts.charger("courbes")
    if not donnees:
        artefacts.signaler_absent("courbes")
        return
    st.subheader("ROC et précision-rappel")
    lignes_roc, lignes_pr = [], []
    for nom, m in donnees["modeles"].items():
        for x, y in zip(m["roc"]["fpr"], m["roc"]["tpr"], strict=True):
            lignes_roc.append({"Modèle": f"{nom} — AUC {nb(m['auc'], 3)}", "x": x, "y": y})
        for x, y in zip(m["pr"]["rappel"], m["pr"]["precision"], strict=True):
            lignes_pr.append({"Modèle": f"{nom} — AP {nb(m['ap'], 3)}", "x": x, "y": y})

    c1, c2 = st.columns(2)
    c1.markdown("**Courbe ROC**")
    diagonale = (alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
                 .mark_line(strokeDash=[4, 4], color="grey", size=1).encode(x="x", y="y"))
    roc = (alt.Chart(pd.DataFrame(lignes_roc)).mark_line().encode(
        x=alt.X("x:Q", title="taux de faux positifs"),
        y=alt.Y("y:Q", title="taux de vrais positifs"),
        color=alt.Color("Modèle:N", legend=LEGENDE, scale=alt.Scale(scheme="dark2")),
    ).properties(height=280))
    c1.altair_chart(roc + diagonale, use_container_width=True)

    c2.markdown("**Courbe précision-rappel**")
    hasard = (alt.Chart(pd.DataFrame({"y": [donnees["taux_pluie_test"]]}))
              .mark_rule(strokeDash=[4, 4], color="grey", size=1).encode(y="y:Q"))
    pr = (alt.Chart(pd.DataFrame(lignes_pr)).mark_line().encode(
        x=alt.X("x:Q", title="rappel"),
        y=alt.Y("y:Q", title="précision", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Modèle:N", legend=LEGENDE, scale=alt.Scale(scheme="dark2")),
    ).properties(height=280))
    c2.altair_chart(pr + hasard, use_container_width=True)
    st.caption(
        "La lecture précision-rappel est plus sévère que la ROC sur une cible déséquilibrée. "
        f"Ligne pointillée : hasard ({nb(donnees['taux_pluie_test'] * 100, 0)} % de pluie)."
    )


def seuil_interactif():
    donnees = artefacts.charger("seuil")
    if not donnees:
        artefacts.signaler_absent("seuil")
        return
    grille = {round(g["seuil"], 2): g for g in donnees["grille"]}
    defaut = donnees.get("seuil_f1_max", 0.5)

    st.subheader("Seuil de décision")
    seuil = st.slider("Seuil", 0.05, 0.95, float(defaut), 0.01,
                      help="La matrice de confusion est recalculée à la volée.")
    ligne = grille[round(seuil, 2)]
    reference = grille[0.50]

    colonnes = st.columns(4)
    for colonne, (libelle, cle) in zip(colonnes, [
        ("Rappel (pluie)", "rappel"), ("Précision (pluie)", "precision"),
        ("F1 (pluie)", "f1"), ("Accuracy", "accuracy"),
    ], strict=True):
        colonne.metric(libelle, nb(ligne[cle], 3),
                       delta=f"{signe(ligne[cle] - reference[cle])} vs 0,50")

    gauche, droite = st.columns([2, 3])
    with gauche:
        st.markdown("**Matrice de confusion**")
        matrice = pd.DataFrame(
            [[ligne["vn"], ligne["fp"]], [ligne["fn"], ligne["vp"]]],
            index=["Réalité : sec", "Réalité : pluie"],
            columns=["Prédit sec", "Prédit pluie"],
        )
        st.dataframe(matrice.style.format(ent), use_container_width=True)
        st.caption(f"{ent(ligne['fn'])} pluies manquées · {ent(ligne['fp'])} fausses alertes")
    with droite:
        courbe = pd.DataFrame([
            {"Seuil": g["seuil"], "Score": g[cle], "Métrique": nom}
            for g in donnees["grille"]
            for cle, nom in [("precision", "précision"), ("rappel", "rappel"), ("f1", "F1")]
        ])
        trace = (alt.Chart(courbe).mark_line().encode(
            x=alt.X("Seuil:Q", title="seuil de décision"),
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Métrique:N", legend=alt.Legend(orient="bottom", title=None)),
        ).properties(height=280))
        curseur = (alt.Chart(pd.DataFrame({"Seuil": [seuil]}))
                   .mark_rule(color="#111111", size=2).encode(x="Seuil:Q"))
        production = (alt.Chart(pd.DataFrame({"Seuil": [donnees["seuil_production"]]}))
                      .mark_rule(color="grey", strokeDash=[4, 4]).encode(x="Seuil:Q"))
        st.altair_chart(trace + production + curseur, use_container_width=True)
        st.caption("Gris pointillé : seuil en production · Noir : curseur")


# ── MLflow — Tracking & Registry ───────────────────────────────────────

FLUX_MLFLOW = """
digraph mlflow_flow {
  rankdir=LR;
  nodesep=0.5; ranksep=0.6; fontname="Helvetica";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=14,
        color="#bbbbbb", margin="0.20,0.12"];
  edge [fontname="Helvetica", fontsize=12, color="#777777"];

  train [label="train.py\\nentraîne le modèle", fillcolor="#fff3cd"];
  mlflow [label="MLflow\\nTracking Server", fillcolor="#d1e7dd"];
  registry [label="Model Registry\\nalias champion", fillcolor="#d1e7dd"];
  api [label="API FastAPI\\ncharge le modèle", fillcolor="#cfe2ff"];

  train -> mlflow [label=" log params\\n+ métriques"];
  train -> registry [label=" log_model +\\n register"];
  registry -> api [label=" models:/rain-australia\\n@champion"];
}
"""


def suivi_mlflow():
    st.subheader("MLflow — Suivi des expériences")

    # Métriques visuelles
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expériences trackées", "4 modèles")
    c2.metric("Métriques par run", "4")
    c3.metric("Registry", "rain-australia")
    c4.metric("Alias de production", "champion")

    st.markdown("")

    # Flux visuel
    g1, g2 = st.columns([3, 2])
    with g1:
        st.markdown("**Flux de bout en bout**")
        st.graphviz_chart(FLUX_MLFLOW)
    with g2:
        st.markdown("**Ce que chaque run enregistre**")
        st.markdown(
            "- **Paramètres** : type de modèle, nombre de features, taille du train, "
            "seuils recommandés\n"
            "- **Métriques** : accuracy, rappel (pluie), F1 (pluie), ROC-AUC\n"
            "- **Artefact** : le `Pipeline` sklearn complet (préprocessing + modèle)\n"
            "- **Signature** : schéma d'entrée/sortie pour validation automatique"
        )
        st.info(
            "Promouvoir un nouveau modèle = **déplacer l'alias `champion`**, pas redéployer "
            "l'image Docker. L'API recharge le modèle via `POST /reload`."
        )


# ── DVC — Versionnement des données ───────────────────────────────────

def versionnement_dvc():
    st.subheader("DVC — Versionnement des données")

    c1, c2, c3 = st.columns(3)
    c1.metric("Fichier versionné", "weatherAUS.csv")
    c2.metric("Taille", "145 460 lignes")
    c3.metric("Remote", "DagsHub")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Pourquoi DVC ?**")
        st.markdown(
            "Le CSV est trop volumineux pour git. DVC versionne son **empreinte MD5** "
            "dans `Data/weatherAUS.csv.dvc` (98 octets dans git), le fichier réel vit "
            "sur le remote DagsHub."
        )
        st.code("dvc pull    # reconstruit le jeu de données exact d'un commit", language="bash")
    with g2:
        st.markdown("**Intégration Docker**")
        st.markdown(
            "Le conteneur `trainer` monte le dossier `Data/` en lecture seule :\n"
        )
        st.code(
            "volumes:\n"
            "  - ../Data:/data:ro     # CSV local → /data/ dans le conteneur",
            language="yaml",
        )
        st.caption(
            "Le serveur de production n'a ni le CSV ni l'image d'entraînement — "
            "séparation stricte entre train et serve."
        )
