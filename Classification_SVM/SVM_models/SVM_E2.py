# E2 — Baseline immuable pour référence
# - Prétraitement : imputation (num = médiane, cat = mode), OHE(handle_unknown="ignore"),
#                   StandardScaler(with_mean=False) sur numériques
# - Modèle : LinearSVC(C=1.0, loss="squared_hinge", class_weight="balanced", max_iter=2000, random_state=42)

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC

_pipe = None  # pipeline globale 

def _prepare_xy_from_csv(path: str,
                         target_col: str = "reservation_status",
                         id_col: str = "row_id"):
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Colonne cible '{target_col}' absente de {path}.")

    y = df[target_col]
    # Si la cible est textuelle, la mapper vers {0,1,2}
    if not np.issubdtype(y.dtype, np.number):
        mapping = {
            "Check-out": 0, "Check-Out": 0, "check-out": 0, "Checkout": 0,
            "Cancel": 1, "cancel": 1,
            "No-Show": 2, "no-show": 2, "No Show": 2, "no show": 2
        }
        y = y.map(mapping)
        if y.isna().any():
            raise ValueError("Valeurs cibles non reconnues pour (0/1/2).")

    drop_cols = [target_col]
    if id_col in df.columns:
        drop_cols.append(id_col)
    X = df.drop(columns=drop_cols, errors="ignore")
    return X, y.astype(int)

def load(train_csv_path: str) -> None:
    """Entraîne la baseline E2 directement depuis le CSV d'entraînement."""
    global _pipe
    X, y = _prepare_xy_from_csv(train_csv_path)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]), selector(dtype_include=np.number)),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), selector(dtype_exclude=np.number)),
        ],
        sparse_threshold=1.0  
    )

    clf = LinearSVC(
        C=1.0,
        loss="squared_hinge",
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )

    _pipe = Pipeline([
        ("preprocess", preprocess),
        ("clf", clf),
    ])
    _pipe.fit(X, y)

def predict(X: pd.DataFrame):
    """Prédit 0/1/2 à partir d'un DataFrame de features (sans row_id ni target)."""
    if _pipe is None:
        raise RuntimeError("Modèle non chargé")
    return _pipe.predict(X)
