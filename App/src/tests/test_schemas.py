"""Tests unitaires — validation des schémas Pydantic (entrée/sortie de l'API)."""
import pytest
from pydantic import ValidationError

from src.api.schemas import PredictionOut, WeatherFeatures


# ---------------------------------------------------------------------------
# Tests — WeatherFeatures (entrée)
# ---------------------------------------------------------------------------
class TestWeatherFeatures:
    def test_all_fields_optional(self):
        """Un payload vide est valide (tous les champs seront NaN → imputés)."""
        wf = WeatherFeatures()
        assert wf.Location is None
        assert wf.Humidity3pm is None

    def test_full_payload(self):
        wf = WeatherFeatures(
            Location="Sydney", Month=7, RainToday="Yes",
            Humidity3pm=80, Sunshine=3.5, Pressure3pm=1008.0,
            Rainfall=12.0, WindGustSpeed=56, Cloud3pm=8, Temp3pm=16.0,
        )
        assert wf.Location == "Sydney"
        assert wf.Month == 7

    def test_month_must_be_between_1_and_12(self):
        with pytest.raises(ValidationError):
            WeatherFeatures(Month=0)
        with pytest.raises(ValidationError):
            WeatherFeatures(Month=13)

    def test_month_boundaries_valid(self):
        assert WeatherFeatures(Month=1).Month == 1
        assert WeatherFeatures(Month=12).Month == 12

    def test_partial_payload(self):
        """Quelques champs renseignés, le reste None."""
        wf = WeatherFeatures(Humidity3pm=65, RainToday="No")
        assert wf.Humidity3pm == 65
        assert wf.RainToday == "No"
        assert wf.Location is None
        assert wf.Sunshine is None

    def test_model_dump_returns_dict(self):
        wf = WeatherFeatures(Location="Perth")
        d = wf.model_dump()
        assert isinstance(d, dict)
        assert d["Location"] == "Perth"
        assert "Humidity3pm" in d  # même s'il est None, il est présent


# ---------------------------------------------------------------------------
# Tests — PredictionOut (sortie)
# ---------------------------------------------------------------------------
class TestPredictionOut:
    def test_valid_output(self):
        out = PredictionOut(rain_tomorrow=True, probability=0.82, threshold=0.5, model_version="3")
        assert out.rain_tomorrow is True
        assert out.probability == 0.82

    def test_default_threshold(self):
        out = PredictionOut(rain_tomorrow=False, probability=0.1)
        assert out.threshold == 0.5

    def test_model_version_optional(self):
        out = PredictionOut(rain_tomorrow=False, probability=0.3)
        assert out.model_version is None

    def test_missing_required_fields_raises(self):
        with pytest.raises(ValidationError):
            PredictionOut()  # rain_tomorrow et probability manquants
