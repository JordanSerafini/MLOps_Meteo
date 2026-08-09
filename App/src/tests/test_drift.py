"""Tests du job de détection de dérive.

On teste la lecture du journal et l'extraction des métriques, pas Evidently lui-même.
"""
import json

import pandas as pd
import pytest

from src.drift import charge_courant, extrait_metriques

COLONNES = ["Humidity3pm", "Sunshine", "Location"]


def ecrit_journal(chemin, entrees):
    with chemin.open("w", encoding="utf-8") as f:
        for e in entrees:
            f.write(json.dumps(e) + "\n")


class TestChargeCourant:
    def test_journal_absent_renvoie_un_dataframe_vide(self, tmp_path):
        df = charge_courant(tmp_path / "rien.jsonl", COLONNES)
        assert df.empty
        assert list(df.columns) == COLONNES

    def test_extrait_les_features(self, tmp_path):
        journal = tmp_path / "predictions.jsonl"
        ecrit_journal(journal, [
            {"proba": 0.7, "features": {"Humidity3pm": 80, "Sunshine": 2.0, "Location": "Sydney"}},
            {"proba": 0.2, "features": {"Humidity3pm": 30, "Sunshine": 9.0, "Location": "Perth"}},
        ])
        df = charge_courant(journal, COLONNES)

        assert len(df) == 2
        assert df["Location"].tolist() == ["Sydney", "Perth"]
        assert df["Humidity3pm"].tolist() == [80, 30]

    def test_ignore_les_lignes_illisibles(self, tmp_path):
        """Un journal tronqué en cours d'écriture ne doit pas faire échouer le job."""
        journal = tmp_path / "predictions.jsonl"
        journal.write_text(
            '{"features": {"Humidity3pm": 80, "Sunshine": 2.0, "Location": "Sydney"}}\n'
            '{"features": {"Humidity3pm": 4\n'          # ligne coupée
            "\n"                                        # ligne vide
            '{"proba": 0.5}\n'                          # pas de champ features
            '{"features": {"Humidity3pm": 30, "Sunshine": 9.0, "Location": "Perth"}}\n',
            encoding="utf-8",
        )
        df = charge_courant(journal, COLONNES)
        assert len(df) == 2

    def test_colonnes_manquantes_deviennent_nan(self, tmp_path):
        journal = tmp_path / "predictions.jsonl"
        ecrit_journal(journal, [{"features": {"Humidity3pm": 80}}])
        df = charge_courant(journal, COLONNES)

        assert list(df.columns) == COLONNES
        assert pd.isna(df["Sunshine"].iloc[0])


class TestExtraitMetriques:
    def test_lit_le_compte_et_la_part(self):
        resultat = {"metrics": [
            {"metric_name": "ValueDrift(column=Sunshine)", "value": 0.05},
            {"metric_name": "DriftedColumnsCount(drift_share=0.5)", "value": {"count": 3.0, "share": 0.3333}},
        ]}
        nb, part = extrait_metriques(resultat)
        assert nb == 3
        assert part == pytest.approx(0.3333)

    def test_leve_si_la_metrique_est_absente(self):
        with pytest.raises(RuntimeError, match="DriftedColumnsCount"):
            extrait_metriques({"metrics": [{"metric_name": "ValueDrift(column=X)", "value": 0.1}]})
