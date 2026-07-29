"""Partie 1 — Données, exploration et préprocessing (phase 1 de la feuille de route).

Tout vient de `artefacts/eda.json`, produit par `Data/eda_explore.py`. Aucun chiffre n'est
saisi à la main : ce qui est affiché ici est ce que le script a mesuré sur les 145 460 lignes.
"""
import altair as alt
import pandas as pd
import streamlit as st

import artefacts
from formats import ent, pct

MOIS = ["janv.", "févr.", "mars", "avril", "mai", "juin",
        "juil.", "août", "sept.", "oct.", "nov.", "déc."]


def afficher():
    eda = artefacts.charger("eda")
    if not eda:
        artefacts.signaler_absent("eda")
        return

    volumetrie(eda)
    st.divider()
    cible(eda)
    st.divider()
    manquants(eda)
    st.divider()
    signal(eda)
    st.divider()
    preprocessing()


def volumetrie(eda):
    st.subheader("Ce que contient le jeu de données")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", ent(eda["shape"]["n_rows"]))
    c2.metric("Variables", eda["shape"]["n_cols"])
    c3.metric("Stations", eda["locations"]["n_locations"])
    c4.metric("Années couvertes", eda["dates"]["n_years_covered"])
    st.caption(
        f"Relevés quotidiens du Bureau of Meteorology australien, du {eda['dates']['min']} au "
        f"{eda['dates']['max']}. Aucun doublon. La station la mieux fournie "
        f"({eda['locations']['station_max_rows']}) compte "
        f"{ent(eda['locations']['max_rows_per_station'])} lignes, la plus pauvre "
        f"({eda['locations']['station_min_rows']}) {ent(eda['locations']['min_rows_per_station'])}."
    )


def cible(eda):
    st.subheader("La cible est déséquilibrée, et ça change tout")
    t = eda["target"]
    b = eda["baseline_persistence"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Jours de pluie (RainTomorrow = Yes)", pct(t["base_rate_yes_pct"], sur_cent=True))
    c2.metric("Baseline « toujours Non »", pct(t["naive_always_no_pct"], sur_cent=True),
              help="Accuracy d'un modèle qui répond systématiquement « pas de pluie ».")
    c3.metric("Baseline « comme aujourd'hui »", pct(b["accuracy_pct"], sur_cent=True),
              help="Recopier la météo du jour : la persistance, référence classique en météo.")
    st.markdown(
        f"Répondre toujours « non » donne déjà **{pct(t['naive_always_no_pct'], sur_cent=True)} "
        "d'accuracy** sans rien apprendre. C'est la raison pour laquelle nous suivons le rappel de la classe "
        "« pluie » et non l'accuracy : les deux baselines ci-dessus ont un rappel de 0 % et de "
        f"{pct(b['p_rain_tomorrow_given_raintoday_yes_pct'], 0, sur_cent=True)} respectivement.\n\n"
        f"Une information utile au passage : quand il pleut aujourd'hui, il pleut demain dans "
        f"{pct(b['p_rain_tomorrow_given_raintoday_yes_pct'], 0, sur_cent=True)} des cas, contre "
        f"{pct(b['p_rain_tomorrow_given_raintoday_no_pct'], 0, sur_cent=True)} quand il ne pleut "
        "pas. `RainToday` porte donc un vrai signal — mais loin d'être suffisant."
    )
    st.caption(f"{ent(t['counts']['NaN'])} lignes sans cible "
               f"({pct(t['pct_missing'], 2, sur_cent=True)}) sont retirées : "
               "une cible ne s'impute pas.")


def manquants(eda):
    st.subheader("Les valeurs manquantes : imputer plutôt que supprimer")
    df = (pd.DataFrame(list(eda["missing_pct"].items()), columns=["Variable", "Manquant (%)"])
          .query("`Manquant (%)` > 0")
          .sort_values("Manquant (%)", ascending=False))
    graphique = (
        alt.Chart(df).mark_bar().encode(
            x=alt.X("Manquant (%):Q", title="part de valeurs manquantes (%)"),
            y=alt.Y("Variable:N", sort="-x", title=None),
            color=alt.condition(alt.datum["Manquant (%)"] > 30, alt.value("#d1495b"),
                                alt.value("#4c78a8")),
            tooltip=["Variable", "Manquant (%)"],
        ).properties(height=520)  # 21 barres : en dessous, altair décime les étiquettes
    )
    c1, c2 = st.columns([3, 2])
    c1.altair_chart(graphique, use_container_width=True)
    c2.markdown(
        "Quatre variables dépassent 38 % de manquants : `Sunshine`, `Evaporation`, `Cloud3pm`, "
        "`Cloud9am`.\n\n"
        "Les supprimer aurait été le réflexe. Nous les gardons, parce que `Sunshine` et "
        "`Cloud3pm` sont justement **deux des variables les plus corrélées à la pluie du "
        "lendemain** (voir ci-dessous). Supprimer les lignes concernées reviendrait à jeter "
        "près de la moitié du jeu de données ; supprimer les colonnes reviendrait à jeter du "
        "signal.\n\n"
        "L'imputation est donc faite dans le `Pipeline` scikit-learn : médiane pour les "
        "numériques, modalité la plus fréquente pour les catégorielles. Elle voyage avec le "
        "modèle jusqu'à l'API."
    )


def signal(eda):
    st.subheader("Où est le signal ?")
    onglet_corr, onglet_saison, onglet_geo, onglet_vent = st.tabs(
        ["Corrélations", "Saisonnalité", "Géographie", "Direction du vent"]
    )

    with onglet_corr:
        corr = eda["pointbiserial_corr_highlight"]
        df = pd.DataFrame({"Variable": list(corr), "Corrélation": list(corr.values())})
        df["Sens"] = df["Corrélation"].apply(lambda v: "favorise la pluie" if v > 0 else "annonce le sec")
        graphique = (
            alt.Chart(df).mark_bar().encode(
                x=alt.X("Corrélation:Q", title="corrélation point-bisériale avec RainTomorrow"),
                y=alt.Y("Variable:N", sort=alt.EncodingSortField("Corrélation", op="max"), title=None),
                color=alt.Color("Sens:N", legend=alt.Legend(orient="bottom", title=None),
                                scale=alt.Scale(range=["#4c78a8", "#f58518"])),
                tooltip=["Variable", "Corrélation"],
            ).properties(height=260)
        )
        st.altair_chart(graphique, use_container_width=True)
        moyennes = eda["class_means"]
        df_moy = pd.DataFrame([
            {"Variable": var, "Jours sans pluie demain": vals["No"], "Jours de pluie demain": vals["Yes"]}
            for var, vals in moyennes.items()
        ])
        st.dataframe(df_moy, use_container_width=True, hide_index=True)
        st.markdown(
            "`Humidity3pm` (+0,45) et `Sunshine` (−0,45) dominent, et l'écart entre les deux "
            "classes est franc : 69 % d'humidité à 15 h les jours suivis de pluie contre 47 %, "
            "4,5 heures de soleil contre 8,5. Rien de surprenant pour un météorologue, mais ça "
            "confirme que le signal est dans les mesures de fin de journée."
        )

    with onglet_saison:
        s = eda["seasonality"]["rain_rate_by_month_pct"]
        df = pd.DataFrame({"Mois": [MOIS[int(m) - 1] for m in s], "Taux de pluie (%)": list(s.values())})
        graphique = (
            alt.Chart(df).mark_line(point=True).encode(
                x=alt.X("Mois:N", sort=MOIS, title=None),
                y=alt.Y("Taux de pluie (%):Q", scale=alt.Scale(zero=False)),
                tooltip=["Mois", "Taux de pluie (%)"],
            ).properties(height=280)
        )
        st.altair_chart(graphique, use_container_width=True)
        st.markdown(
            "L'hiver austral est plus humide : "
            f"{pct(eda['seasonality']['wettest_month_pct'], 1, sur_cent=True)} de jours de pluie en "
            f"juillet contre {pct(eda['seasonality']['driest_month_pct'], 1, sur_cent=True)} en "
            "janvier. "
            "L'amplitude reste modérée (7 points) mais elle justifie d'avoir extrait `Month` de la "
            "date et de l'avoir traité comme une variable **catégorielle** : le mois 12 est proche "
            "du mois 1, un encodage numérique aurait imposé au modèle un ordre qui n'existe pas."
        )

    with onglet_geo:
        g = eda["geography"]["rain_rate_by_location_pct"]
        df = (pd.DataFrame({"Station": list(g), "Taux de pluie (%)": list(g.values())})
              .sort_values("Taux de pluie (%)", ascending=False))
        graphique = (
            alt.Chart(df).mark_bar().encode(
                x=alt.X("Taux de pluie (%):Q"),
                y=alt.Y("Station:N", sort="-x", title=None),
                tooltip=["Station", "Taux de pluie (%)"],
            ).properties(height=700)
        )
        st.altair_chart(graphique, use_container_width=True)
        st.markdown(
            "L'écart va de **36,6 % à Portland** à **6,8 % à Woomera** : un facteur cinq. La station "
            "est donc une variable de premier plan, et c'est aussi ce qui nous a servi à fabriquer "
            "une dérive artificielle pour tester le monitoring — rejouer les relevés de Portland "
            "décale les distributions sans inventer de données."
        )

    with onglet_vent:
        v = eda["windgustdir"]["rain_rate_pct"]
        df = (pd.DataFrame({"Direction": list(v), "Taux de pluie (%)": list(v.values())})
              .sort_values("Taux de pluie (%)", ascending=False))
        graphique = (
            alt.Chart(df).mark_bar().encode(
                x=alt.X("Direction:N", sort="-y", title="direction de la rafale maximale"),
                y=alt.Y("Taux de pluie (%):Q"),
                tooltip=["Direction", "Taux de pluie (%)"],
            ).properties(height=280)
        )
        st.altair_chart(graphique, use_container_width=True)
        st.markdown(
            "Les rafales de secteur nord-ouest sont suivies de pluie dans 28 % des cas, celles "
            "d'est dans 15 %. L'ordre des seize modalités n'a aucun sens numérique : encodage "
            "*one-hot*, avec `handle_unknown=\"ignore\"` pour qu'une direction jamais vue en "
            "production ne fasse pas tomber l'API."
        )


def preprocessing():
    st.subheader("Le préprocessing vit dans le modèle, pas à côté")
    c1, c2 = st.columns(2)
    c1.markdown(
        "**Un seul objet sérialisé**\n\n"
        "Imputation, standardisation et encodage sont des étapes d'un `Pipeline` scikit-learn. "
        "C'est ce `Pipeline` complet qui est enregistré dans MLflow et chargé par l'API. "
        "Conséquence : il est impossible que le préprocessing de production diverge de celui de "
        "l'entraînement, puisqu'il n'en existe qu'un.\n\n"
        "Le module `src/data.py` est partagé par `train.py` et l'API — même code, mêmes colonnes."
    )
    c2.markdown(
        "**Sur la fuite de données, soyons honnêtes**\n\n"
        "Nous avons mesuré l'écart entre une médiane calculée sur tout le jeu et la même calculée "
        "sur le train seul : **0,0**. Avec 113 754 lignes d'entraînement, la fuite ne coûte "
        "rien ici, et prétendre le contraire serait facile à démonter.\n\n"
        "L'argument qui tient est ailleurs : en production, l'API reçoit **une seule "
        "observation**. Calculer une médiane sur une ligne n'a aucun sens — les statistiques "
        "doivent avoir été mémorisées à l'entraînement et transportées avec le modèle. "
        "C'est exactement ce que fait le `Pipeline`."
    )
    st.info(
        "Vérification faite dans le notebook 2 : on envoie volontairement `Location = \"Marseille\"` "
        "au préprocesseur. Une station inconnue ne doit pas faire tomber le service — elle passe, "
        "encodée en vecteur nul."
    )
