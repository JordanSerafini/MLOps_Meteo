"""Partie 2 — Modélisation, suivi d'expériences et versionnement (phase 2).

Les chiffres viennent de `artefacts/{modeles,courbes,seuil,importances}.json`, régénérables
par `make artefacts`. Le curseur de seuil ne réentraîne rien : la matrice de confusion a été
calculée à l'avance pour 91 seuils, on ne fait que lire la ligne correspondante.
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
# Légende sur une colonne : trois noms de modèles côte à côte débordent d'une demi-largeur.
LEGENDE = alt.Legend(orient="bottom", direction="vertical", title=None, labelLimit=300)


def afficher():
    comparatif()
    st.divider()
    courbes()
    st.divider()
    seuil_interactif()
    st.divider()
    importances()
    st.divider()
    suivi()


def comparatif():
    donnees = artefacts.charger("modeles")
    if not donnees:
        artefacts.signaler_absent("modeles")
        return
    st.subheader("Quatre modèles, et le piège de l'accuracy")
    df = pd.DataFrame(donnees["lignes"]).rename(columns=COLONNES)[list(COLONNES.values())]
    # Colonnes converties en texte : st.dataframe rend les nombres avec son propre format
    # (point décimal, six décimales) et ignore le formatage pandas. `roc_auc` vaut null pour
    # la baseline, d'où le tiret.
    for colonne in SCORES:
        df[colonne] = df[colonne].map(lambda v: "—" if pd.isna(v) else nb(v, 4))
    df["Entraînement (s)"] = df["Entraînement (s)"].map(lambda v: nb(v, 1))
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(
        f"Évaluation sur {ent(donnees['n_test'])} observations de test jamais vues "
        f"({nb(donnees['taux_pluie_test'] * 100, 1)} % de jours de pluie), seuil de décision à "
        f"{nb(donnees['seuil_evalue'])}. Découpage stratifié, graine fixée."
    )
    c1, c2 = st.columns(2)
    c1.markdown(
        "**Les trois modèles se tiennent en quatre points**\n\n"
        "Entre 0,85 et 0,86 d'accuracy, entre 0,50 et 0,54 de rappel. Le gradient boosting "
        "termine devant, mais l'écart avec la régression logistique est mince — et la baseline "
        "« toujours Non » affiche déjà 0,776 d'accuracy avec un rappel nul.\n\n"
        "Le vrai argument en faveur du gradient boosting n'est pas sa précision, c'est son coût : "
        "**2,6 secondes d'entraînement contre 20,8** pour la forêt aléatoire, pour un résultat "
        "meilleur. Sur une pipeline qu'on relance à chaque dérive, ça compte."
    )
    c2.markdown(
        "**La variante `balanced` prend le problème à l'envers**\n\n"
        "En pondérant les classes pendant l'apprentissage, le rappel monte à 0,78. Mais la "
        "précision tombe à 0,53 : près d'une alerte pluie sur deux est fausse, et l'accuracy "
        "descend à 0,80.\n\n"
        "Ni cette position ni celle des modèles bruts n'est bonne dans l'absolu — tout dépend du "
        "coût d'une pluie manquée face à celui d'une fausse alerte. Cet arbitrage se règle mieux "
        "**après** l'entraînement, avec le seuil, comme montré plus bas."
    )
    st.warning(
        "Une contrainte technique rencontrée : `HistGradientBoostingClassifier` refuse les "
        "matrices creuses, or `OneHotEncoder` en produit par défaut. Il faut "
        "`sparse_output=False`, donc un préprocesseur dédié — à retenir si on bascule la "
        "production sur ce modèle."
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
    # Les titres sont posés en markdown, pas dans le graphique : altair les tronque quand la
    # largeur est contrainte par la colonne.
    c1.markdown("**Courbe ROC**")
    diagonale = (alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
                 .mark_line(strokeDash=[4, 4], color="grey", size=1).encode(x="x", y="y"))
    roc = (alt.Chart(pd.DataFrame(lignes_roc)).mark_line().encode(
        x=alt.X("x:Q", title="taux de faux positifs"),
        y=alt.Y("y:Q", title="taux de vrais positifs"),
        color=alt.Color("Modèle:N", legend=LEGENDE, scale=alt.Scale(scheme="dark2")),
    ).properties(height=300))
    c1.altair_chart(roc + diagonale, use_container_width=True)

    c2.markdown("**Courbe précision-rappel**")
    hasard = (alt.Chart(pd.DataFrame({"y": [donnees["taux_pluie_test"]]}))
              .mark_rule(strokeDash=[4, 4], color="grey", size=1).encode(y="y:Q"))
    pr = (alt.Chart(pd.DataFrame(lignes_pr)).mark_line().encode(
        x=alt.X("x:Q", title="rappel"),
        y=alt.Y("y:Q", title="précision", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Modèle:N", legend=LEGENDE, scale=alt.Scale(scheme="dark2")),
    ).properties(height=300))
    c2.altair_chart(pr + hasard, use_container_width=True)

    st.markdown(
        "La ROC est flatteuse sur une cible déséquilibrée : les vrais négatifs, très nombreux, "
        "dominent le calcul, et les trois courbes finissent par se ressembler. La lecture "
        "précision-rappel est plus sévère et plus utile — la ligne pointillée horizontale marque "
        f"le hasard ({nb(donnees['taux_pluie_test'] * 100, 0)} % de pluie). On y voit que la précision "
        "**décroche à partir d'un rappel d'environ 0,7** : c'est l'ordre de grandeur du compromis "
        "atteignable, et ça borne ce qu'il est raisonnable de promettre."
    )


def seuil_interactif():
    donnees = artefacts.charger("seuil")
    if not donnees:
        artefacts.signaler_absent("seuil")
        return
    grille = {round(g["seuil"], 2): g for g in donnees["grille"]}
    defaut = donnees.get("seuil_f1_max", 0.5)

    st.subheader("Le vrai levier, c'est le seuil de décision")
    st.markdown(
        "`predict()` compare la probabilité à un seuil. Ce seuil ne fait pas partie du modèle : "
        "le déplacer ne change ni les coefficients, ni les probabilités, seulement le point de "
        "fonctionnement. En production, c'est la variable d'environnement `DECISION_THRESHOLD`."
    )
    seuil = st.slider("Seuil de décision", 0.05, 0.95, float(defaut), 0.01,
                      help="Déplacez-le : la matrice de confusion est recalculée à partir de "
                           "valeurs mesurées à l'avance sur le jeu de test.")
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
        st.caption(f"{ent(ligne['fn'])} jours de pluie manqués, "
                   f"{ent(ligne['fp'])} fausses alertes.")
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
        ).properties(height=300))
        # Noir, pas rouge : le rouge est déjà pris par la courbe de rappel.
        curseur = (alt.Chart(pd.DataFrame({"Seuil": [seuil]}))
                   .mark_rule(color="#111111", size=2).encode(x="Seuil:Q"))
        production = (alt.Chart(pd.DataFrame({"Seuil": [donnees["seuil_production"]]}))
                      .mark_rule(color="grey", strokeDash=[4, 4]).encode(x="Seuil:Q"))
        st.altair_chart(trace + production + curseur, use_container_width=True)
        st.caption("Trait gris pointillé : seuil actuellement servi en production. "
                   "Trait noir : position du curseur.")

    rattrapes = reference["fn"] - ligne["fn"]
    ajoutees = ligne["fp"] - reference["fp"]
    if rattrapes > 0:
        st.success(
            f"À {nb(seuil)} au lieu de 0,50, on rattrape **{ent(rattrapes)} jours de pluie** "
            f"manqués, au prix de **{ent(ajoutees)} fausses alertes** supplémentaires. "
            "Le modèle, lui, est resté identique."
        )
    elif rattrapes < 0:
        st.info(
            f"À {nb(seuil)}, on manque **{ent(-rattrapes)} jours de pluie de plus** qu'à 0,50, "
            f"mais on évite **{ent(-ajoutees)} fausses alertes**."
        )

    st.markdown(
        f"Le F1 est maximal à **{nb(donnees['seuil_f1_max'])}** : le rappel passe de 0,54 à 0,71 "
        "pour douze points de précision cédés. C'est un tiers des jours de pluie manqués qu'on "
        "récupère, alors que l'écart entre notre meilleur et notre pire modèle ne vaut que quatre "
        "points. **Le choix du modèle pèse moins que le réglage du seuil.**\n\n"
        f"À assumer en soutenance : la production sert encore le seuil de "
        f"{nb(donnees['seuil_production'])}, alors que notre propre analyse recommande "
        f"{nb(donnees['seuil_f1_max'])}. C'est une variable d'environnement à changer, pas un "
        "réentraînement — mais ça n'a pas encore été fait."
    )


def importances():
    donnees = artefacts.charger("importances")
    if not donnees:
        artefacts.signaler_absent("importances")
        return
    st.subheader("Ce sur quoi le modèle s'appuie")
    df = pd.DataFrame(donnees["variables"])
    graphique = (alt.Chart(df).mark_bar().encode(
        x=alt.X("importance:Q", title="importance (forêt aléatoire)"),
        y=alt.Y("variable:N", sort="-x", title=None),
        tooltip=["variable", "importance"],
    ).properties(height=380))
    c1, c2 = st.columns([3, 2])
    c1.altair_chart(graphique, use_container_width=True)
    c2.markdown(
        "`Humidity3pm` domine largement, suivie de la pression et de l'ensoleillement de "
        "l'après-midi. C'est cohérent avec les corrélations mesurées à l'exploration, et c'est "
        "rassurant : le modèle s'appuie sur les variables qu'un météorologue regarderait.\n\n"
        "L'importance calculée par la forêt est celle de la variable **après encodage** : les "
        "modalités one-hot de `Location` et des directions de vent se partagent leur poids, ce "
        "qui les fait paraître moins importantes qu'elles ne le sont réellement."
    )


def suivi():
    st.subheader("Suivi des expériences et versionnement")
    c1, c2 = st.columns(2)
    c1.markdown(
        "**MLflow**\n\n"
        "Chaque entraînement journalise ses paramètres, ses métriques et le `Pipeline` complet. "
        "Le modèle retenu est enregistré dans le **Model Registry** avec l'alias `champion` : "
        "l'API charge `models:/rain-australia@champion` et ne connaît rien d'autre. Promouvoir "
        "un nouveau modèle, c'est déplacer un alias — pas redéployer une image.\n\n"
        "Un garde-fou existe (`MIN_RECALL_FOR_CHAMPION`) mais il est **encore à zéro** : "
        "aujourd'hui n'importe quelle version peut devenir la version de production. C'est une "
        "ligne de configuration à corriger, et nous préférons le dire que le cacher."
    )
    c2.markdown(
        "**DVC**\n\n"
        "Le CSV de 145 460 lignes ne va pas dans git. `Data/weatherAUS.csv.dvc` versionne son "
        "empreinte, le fichier lui-même vit dans le stockage DVC. Un `dvc pull` reconstruit "
        "exactement le jeu de données d'un commit donné.\n\n"
        "L'échantillon de référence du monitoring (`App/reference/reference.csv`, 5 000 lignes) "
        "est en revanche **figé et versionné dans git**. C'était un bug au départ : il était "
        "retiré au hasard à chaque exécution, ce qui rendait les comparaisons de dérive dans le "
        "temps inexploitables."
    )
