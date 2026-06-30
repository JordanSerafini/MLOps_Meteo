"""Tests unitaires — chargement des données et construction du préprocesseur.

Utilise un petit CSV de test généré en fixture pour éviter toute dépendance
au fichier réel (qui est géré par DVC et peut ne pas être présent en CI).
"""
import numpy as np
import pandas as pd
import pytest

from src.data import build_preprocessor, get_feature_lists, load_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def sample_csv(tmp_path):
    """Crée un mini CSV ressemblant à weatherAUS.csv."""
    data = {
        "Date": ["2010-01-01", "2010-01-02", "2010-01-03", "2010-01-04", "2010-01-05"],
        "Location": ["Sydney", "Melbourne", "Sydney", "Melbourne", "Sydney"],
        "MinTemp": [18.0, 14.0, np.nan, 12.0, 20.0],
        "MaxTemp": [30.0, 25.0, 28.0, 22.0, 31.0],
        "Rainfall": [0.0, 5.2, 0.0, 10.0, 0.0],
        "WindGustDir": ["N", "S", "N", np.nan, "E"],
        "WindGustSpeed": [44.0, 50.0, 35.0, 60.0, np.nan],
        "WindDir9am": ["N", "S", "N", "S", "E"],
        "WindDir3pm": ["NE", "SE", "NW", "SW", "E"],
        "Humidity9am": [60.0, 80.0, 55.0, 90.0, 50.0],
        "Humidity3pm": [40.0, 70.0, 35.0, 85.0, 30.0],
        "Pressure9am": [1015.0, 1010.0, 1018.0, 1005.0, 1020.0],
        "Pressure3pm": [1012.0, 1007.0, 1015.0, 1002.0, 1017.0],
        "Cloud9am": [1, 7, 0, 8, 2],
        "Cloud3pm": [2, 8, 1, 7, 3],
        "Temp9am": [22.0, 16.0, 20.0, 14.0, 24.0],
        "Temp3pm": [28.0, 23.0, 26.0, 20.0, 29.0],
        "RainToday": ["No", "Yes", "No", "Yes", "No"],
        "RainTomorrow": ["No", "Yes", "No", "Yes", "No"],
        "Sunshine": [10.0, 2.0, 8.0, 1.0, 11.0],
        "Evaporation": [5.0, 3.0, np.nan, 2.0, 6.0],
        "WindSpeed9am": [15.0, 20.0, 10.0, 25.0, np.nan],
        "WindSpeed3pm": [20.0, 25.0, 15.0, 30.0, 18.0],
    }
    path = tmp_path / "weather_test.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Tests — load_dataset
# ---------------------------------------------------------------------------
class TestLoadDataset:
    def test_returns_tuple(self, sample_csv):
        X, y = load_dataset(sample_csv)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_target_not_in_features(self, sample_csv):
        X, _ = load_dataset(sample_csv)
        assert "RainTomorrow" not in X.columns

    def test_date_not_in_features(self, sample_csv):
        X, _ = load_dataset(sample_csv)
        assert "Date" not in X.columns

    def test_month_extracted(self, sample_csv):
        X, _ = load_dataset(sample_csv)
        assert "Month" in X.columns

    def test_target_is_binary(self, sample_csv):
        _, y = load_dataset(sample_csv)
        assert set(y.unique()).issubset({0, 1})

    def test_no_rows_lost_when_target_present(self, sample_csv):
        """Toutes les lignes ont une cible → aucune n'est supprimée."""
        X, y = load_dataset(sample_csv)
        assert len(X) == 5
        assert len(y) == 5

    def test_rows_with_missing_target_dropped(self, tmp_path):
        """Lignes sans cible supprimées."""
        df = pd.DataFrame({
            "Date": ["2010-01-01", "2010-01-02"],
            "Location": ["Sydney", "Sydney"],
            "MinTemp": [18.0, 14.0],
            "MaxTemp": [30.0, 25.0],
            "Rainfall": [0.0, 5.0],
            "RainToday": ["No", "Yes"],
            "RainTomorrow": ["Yes", np.nan],
        })
        path = tmp_path / "missing_target.csv"
        df.to_csv(path, index=False)
        X, y = load_dataset(path)
        assert len(X) == 1
        assert len(y) == 1


# ---------------------------------------------------------------------------
# Tests — get_feature_lists
# ---------------------------------------------------------------------------
class TestGetFeatureLists:
    def test_separates_numeric_and_categorical(self, sample_csv):
        X, _ = load_dataset(sample_csv)
        numeric, categorical = get_feature_lists(X)
        assert "Location" in categorical
        assert "RainToday" in categorical
        assert "Humidity3pm" in numeric
        assert "Pressure3pm" in numeric

    def test_no_overlap(self, sample_csv):
        X, _ = load_dataset(sample_csv)
        numeric, categorical = get_feature_lists(X)
        assert not set(numeric) & set(categorical)

    def test_union_covers_all_columns(self, sample_csv):
        X, _ = load_dataset(sample_csv)
        numeric, categorical = get_feature_lists(X)
        assert set(numeric) | set(categorical) == set(X.columns)


# ---------------------------------------------------------------------------
# Tests — build_preprocessor
# ---------------------------------------------------------------------------
class TestBuildPreprocessor:
    def test_preprocessor_fits_and_transforms(self, sample_csv):
        X, _ = load_dataset(sample_csv)
        numeric, categorical = get_feature_lists(X)
        prep = build_preprocessor(numeric, categorical)
        result = prep.fit_transform(X)
        assert result.shape[0] == len(X)

    def test_output_has_no_nans(self, sample_csv):
        """Le préprocesseur doit imputer les valeurs manquantes."""
        X, _ = load_dataset(sample_csv)
        numeric, categorical = get_feature_lists(X)
        prep = build_preprocessor(numeric, categorical)
        result = prep.fit_transform(X)
        # sparse → dense if needed
        if hasattr(result, "toarray"):
            result = result.toarray()
        assert not np.isnan(result).any()
