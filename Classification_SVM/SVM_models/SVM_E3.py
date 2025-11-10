# E3 — Balayage de C (sélection sur split stratifié 80/20), puis refit sur tout le train.
# CSV-only : load() entraîne depuis un CSV d'entraînement ; predict() infère 0/1/2.
# Prétraitement : imputation (num=median, cat=most_frequent) + OHE(handle_unknown) + StandardScaler(with_mean=False)
# Modèle : LinearSVC ; on sélectionne C sur un split 80/20 (rapide et simple).

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

_pipe = None  # pipeline finale (prétraitement + LinearSVC) entraînée avec le meilleur C

# --- Helpers ---

def _prepare_xy_from_csv(path: str,
                         target_col: str = "reservation_status",
                         id_col: str = "row_id"):
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Colonne cible '{target_col}' absente de {path}.")

    y = df[target_col]
    # Mapper si la cible est textuelle
    if not np.issubdtype(y.dtype, np.number):
        mapping = {
            "Check-out": 0, "Check-Out": 0, "check-out": 0, "Checkout": 0,
            "Cancel": 1, "cancel": 1,
            "No-Show": 2, "no-show": 2, "No Show": 2, "no show": 2
        }
        y = y.map(mapping)
        if y.isna().any():
            raise ValueError("Valeurs cibles non reconnues : impossible de les mapper en (0/1/2).")

    drop_cols = [target_col]
    if id_col in df.columns:
        drop_cols.append(id_col)

    X = df.drop(columns=drop_cols, errors="ignore")
    return X, y.astype(int)


def _make_preprocess():
    return ColumnTransformer(
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


def _build_pipeline(C: float, random_state: int = 42):
    preprocess = _make_preprocess()
    clf = LinearSVC(
        C=C,
        loss="squared_hinge",
        class_weight="balanced",
        max_iter=2000,
        random_state=random_state,
    )
    return Pipeline([
        ("preprocess", preprocess),
        ("clf", clf),
    ])


# --- API attendue par ton runner ---

def load(train_csv_path: str) -> None:
    """
    Entraîne en choisissant le meilleur C sur split stratifié 80/20,
    puis réentraine la pipeline complète sur l'ensemble du train avec le meilleur C.
    """
    global _pipe

    # 1) Données
    X, y = _prepare_xy_from_csv(train_csv_path)

    # 2) Split 80/20 stratifié
    RS = 42
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=RS, stratify=y
    )

    # 3) Grid de C (rapide)
    C_GRID = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    best_c = None
    best_f1 = -np.inf

    for C in C_GRID:
        pipe = _build_pipeline(C, random_state=RS)
        pipe.fit(X_tr, y_tr)
        y_hat = pipe.predict(X_va)
        f1w = f1_score(y_va, y_hat, average="weighted")
        # print(f"[E3] C={C} -> F1_w={f1w:.4f}") 

        if f1w > best_f1:
            best_f1 = f1w
            best_c = C

    if best_c is None:
        # sécurité
        best_c = 1.0

    # 4) Refit final sur 100% du train avec le meilleur C
    _pipe = _build_pipeline(best_c, random_state=RS)
    _pipe.fit(X, y)
    # print(f"[E3] Best C={best_c} (valid F1_w={best_f1:.4f}). Modèle final entraîné sur tout le train.")


def predict(X: pd.DataFrame):
    """Prédit 0/1/2 à partir d'un DataFrame de features (sans row_id ni target)."""
    if _pipe is None:
        raise RuntimeError("Modèle non chargé")
    return _pipe.predict(X)
