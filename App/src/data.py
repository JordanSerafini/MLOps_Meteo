"""Chargement des données et construction du préprocesseur.

Module UNIQUE partagé par l'entraînement et l'inférence -> zéro divergence de schéma
entre train et serve (le préprocessing vit dans le Pipeline sklearn).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from . import config


def load_dataset(path=None):
    """Retourne (X, y) prêts pour l'entraînement.

    - lignes sans cible retirées (non imputables)
    - feature temporelle Month extraite de Date
    - cible encodée 0/1
    """
    path = path or config.DATA_PATH
    df = pd.read_csv(path, na_values=["NA"])
    df = df.dropna(subset=[config.TARGET]).copy()
    df["Month"] = pd.to_datetime(df["Date"], errors="coerce").dt.month.astype(float)
    y = (df[config.TARGET] == "Yes").astype(int)
    X = df.drop(columns=[c for c in config.DROP_COLS if c in df.columns])
    return X, y


def get_feature_lists(X):
    """Sépare colonnes catégorielles / numériques en fonction de la config."""
    categorical = [c for c in config.CATEGORICAL_FEATURES if c in X.columns]
    numeric = [c for c in X.columns if c not in categorical]
    return numeric, categorical


def build_preprocessor(numeric, categorical):
    """ColumnTransformer SANS fuite : imputeurs/scaler ajustés sur le train uniquement."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])
