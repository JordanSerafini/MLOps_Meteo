"""DTO d'entrée/sortie de l'API d'inférence (Pydantic v2).

Tous les champs météo sont optionnels : un champ absent devient NaN côté modèle
et est imputé par le Pipeline (median / most_frequent). Les noms correspondent
EXACTEMENT aux colonnes d'entraînement.
"""
from typing import Optional

from pydantic import BaseModel, Field


class WeatherFeatures(BaseModel):
    # Catégorielles
    Location: Optional[str] = None
    WindGustDir: Optional[str] = None
    WindDir9am: Optional[str] = None
    WindDir3pm: Optional[str] = None
    RainToday: Optional[str] = None
    Month: Optional[int] = Field(default=None, ge=1, le=12)
    # Numériques
    MinTemp: Optional[float] = None
    MaxTemp: Optional[float] = None
    Rainfall: Optional[float] = None
    Evaporation: Optional[float] = None
    Sunshine: Optional[float] = None
    WindGustSpeed: Optional[float] = None
    WindSpeed9am: Optional[float] = None
    WindSpeed3pm: Optional[float] = None
    Humidity9am: Optional[float] = None
    Humidity3pm: Optional[float] = None
    Pressure9am: Optional[float] = None
    Pressure3pm: Optional[float] = None
    Cloud9am: Optional[float] = None
    Cloud3pm: Optional[float] = None
    Temp9am: Optional[float] = None
    Temp3pm: Optional[float] = None

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
    model_version: Optional[str] = None
