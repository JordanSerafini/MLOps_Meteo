"""DTO d'entrée/sortie de l'API d'inférence (Pydantic v2).

Tous les champs météo sont optionnels : un champ absent devient NaN côté modèle
et est imputé par le Pipeline (median / most_frequent). Les noms correspondent
EXACTEMENT aux colonnes d'entraînement.
"""

from pydantic import BaseModel, Field


class WeatherFeatures(BaseModel):
    # Catégorielles
    Location: str | None = None
    WindGustDir: str | None = None
    WindDir9am: str | None = None
    WindDir3pm: str | None = None
    RainToday: str | None = None
    Month: int | None = Field(default=None, ge=1, le=12)
    # Numériques
    MinTemp: float | None = None
    MaxTemp: float | None = None
    Rainfall: float | None = None
    Evaporation: float | None = None
    Sunshine: float | None = None
    WindGustSpeed: float | None = None
    WindSpeed9am: float | None = None
    WindSpeed3pm: float | None = None
    Humidity9am: float | None = None
    Humidity3pm: float | None = None
    Pressure9am: float | None = None
    Pressure3pm: float | None = None
    Cloud9am: float | None = None
    Cloud3pm: float | None = None
    Temp9am: float | None = None
    Temp3pm: float | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "Location": "Sydney", "Month": 7, "RainToday": "Yes",
                "Humidity3pm": 80, "Sunshine": 3.5, "Pressure3pm": 1008.0,
                "Rainfall": 12.0, "WindGustSpeed": 56, "Cloud3pm": 8, "Temp3pm": 16.0,
            }
        }
    }


class PredictionOut(BaseModel):
    rain_tomorrow: bool
    probability: float = Field(description="probabilité de pluie demain [0-1]")
    threshold: float = Field(default=0.5, description="seuil de décision appliqué")
    model_version: str | None = None
