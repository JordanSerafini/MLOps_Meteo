"""Tests du Streamlit de soutenance.

Deux garanties recherchées :
  - les artefacts versionnés existent et ont la forme attendue. Un artefact régénéré avec un
    champ renommé casserait l'affichage le jour de l'oral, pas avant ;
  - l'application se rend sans exception, y compris quand l'API est injoignable — c'est l'état
    dans lequel la CI tourne, et c'est aussi le pire cas d'une démonstration.
"""
import json
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
ARTEFACTS = APP_DIR / "artefacts"


@pytest.fixture()
def chemin_app():
    """`streamlit run` place le dossier du script en tête de sys.path ; AppTest ne le fait
    pas. On reproduit ce comportement, sinon `import api_client` échoue dans le test seul."""
    sys.path.insert(0, str(APP_DIR))
    try:
        yield APP_DIR / "app.py"
    finally:
        sys.path.remove(str(APP_DIR))


def charger(nom):
    chemin = ARTEFACTS / f"{nom}.json"
    assert chemin.exists(), f"{chemin} absent — lancer `make artefacts`"
    return json.loads(chemin.read_text(encoding="utf-8"))


def test_artefacts_presents():
    attendus = {"eda", "modeles", "seuil", "courbes", "importances", "calibration_drift"}
    presents = {p.stem for p in ARTEFACTS.glob("*.json")}
    assert attendus <= presents, f"manquants : {attendus - presents}"


def test_modeles_comparables():
    donnees = charger("modeles")
    assert donnees["n_test"] > 0
    noms = {ligne["modele"] for ligne in donnees["lignes"]}
    assert "Gradient boosting" in noms
    for ligne in donnees["lignes"]:
        for cle in ("accuracy", "precision", "rappel", "f1"):
            assert 0.0 <= ligne[cle] <= 1.0, f"{ligne['modele']}.{cle} = {ligne[cle]}"


def test_grille_de_seuils_complete():
    """L'onglet modélisation indexe la grille au centième : tout trou provoque un KeyError
    quand l'utilisateur déplace le curseur."""
    donnees = charger("seuil")
    seuils = {round(g["seuil"], 2) for g in donnees["grille"]}
    attendus = {round(0.05 + i / 100, 2) for i in range(91)}
    assert attendus <= seuils, f"seuils manquants : sorted(attendus - seuils)[:5] = {sorted(attendus - seuils)[:5]}"
    assert 0.50 in seuils, "le seuil de référence 0,50 doit être présent"


def test_seuil_bascule_bien_le_compromis():
    """Baisser le seuil doit augmenter le rappel et diminuer la précision. Si cette relation
    s'inverse, c'est l'export qui est faux, pas le modèle."""
    donnees = charger("seuil")
    grille = sorted(donnees["grille"], key=lambda g: g["seuil"])
    rappels = [g["rappel"] for g in grille]
    assert rappels == sorted(rappels, reverse=True)
    bas, haut = grille[0], grille[-1]
    assert bas["rappel"] > haut["rappel"]
    assert bas["fn"] < haut["fn"]


def test_courbes_alignees():
    donnees = charger("courbes")
    for nom, m in donnees["modeles"].items():
        assert len(m["roc"]["fpr"]) == len(m["roc"]["tpr"]), nom
        assert len(m["pr"]["rappel"]) == len(m["pr"]["precision"]), nom
        assert 0.5 < m["auc"] <= 1.0, nom


def test_calibration_coherente():
    """Le raisonnement de l'onglet monitoring repose sur la non-recouvrance des deux
    distributions. Si elle disparaît, le discours tenu à l'oral devient faux."""
    donnees = charger("calibration_drift")
    dist = donnees["distribution"]
    assert dist["normal"]["max"] < dist["decale"]["min"]
    assert dist["normal"]["max"] < donnees["seuil_retenu"] < dist["decale"]["min"]
    assert len(donnees["alertes"]) == 6


def test_eda_expose_les_cles_utilisees():
    eda = charger("eda")
    for cle in ("shape", "locations", "dates", "target", "baseline_persistence", "missing_pct",
                "pointbiserial_corr_highlight", "class_means", "geography", "seasonality",
                "windgustdir"):
        assert cle in eda, f"clé `{cle}` absente de eda.json"
    assert len(eda["geography"]["rain_rate_by_location_pct"]) == eda["locations"]["n_locations"]


def test_application_se_rend_sans_api(chemin_app):
    """Rendu complet des cinq onglets avec une API injoignable : aucune exception ne doit
    remonter, seulement des messages d'erreur affichés."""
    module = pytest.importorskip(
        "streamlit.testing.v1",
        reason="streamlit non installé (pip install -r requirements/streamlit.txt)",
    )
    at = module.AppTest.from_file(str(chemin_app), default_timeout=90).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "pleuvoir" in at.title[0].value
    # L'API n'est pas joignable depuis la CI : la barre latérale doit le dire, pas planter.
    assert any("injoignable" in e.value for e in at.error)
