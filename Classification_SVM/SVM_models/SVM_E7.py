# E7 — Optimisation de l'encodage catégoriel via OneHotEncoder(min_frequency) (+ drop='if_binary').
# - Split 80/20 stratifié pour choisir (min_freq, drop_if_binary), puis refit sur tout le train.
# - Correctifs LinearSVC : tol=1e-3, max_iter=10000, dual dynamique (liblinear).

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC


_pipe = None  # pipeline finale entraînée avec le meilleur encodage


# --------- Helpers ---------

def _prepare_xy_from_csv(path: str,
                         target_col: str = "reservation_status",
                         id_col: str = "row_id"):
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Colonne cible '{target_col}' absente de {path}.")

    y = df[target_col]
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


def _make_ohe(min_frequency, drop_if_binary: bool):
    """
    Construit un OneHotEncoder en tolérant les versions de scikit-learn
    où certains arguments ne seraient pas disponibles.
    """
    kwargs = {"handle_unknown": "ignore"}
    if min_frequency is not None:
        kwargs["min_frequency"] = min_frequency
    if drop_if_binary:
        kwargs["drop"] = "if_binary"
    try:
        return OneHotEncoder(**kwargs)
    except TypeError:
        # Fallback si votre version de sklearn ne supporte pas un argument
        kwargs.pop("min_frequency", None)
        kwargs.pop("drop", None)
        return OneHotEncoder(**kwargs)


def _make_preprocess(min_frequency, drop_if_binary: bool):
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]), selector(dtype_include=np.number)),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _make_ohe(min_frequency, drop_if_binary)),
            ]), selector(dtype_exclude=np.number)),
        ],
        sparse_threshold=1.0
    )


def _build_pipeline(min_frequency, drop_if_binary: bool, dual: bool, C: float = 1.0, random_state: int = 42):
    preprocess = _make_preprocess(min_frequency, drop_if_binary)
    clf = LinearSVC(
        C=C,
        loss="squared_hinge",
        penalty="l2",
        dual=dual,               # dual dynamique selon n_samples vs n_features
        class_weight="balanced",
        tol=1e-3,
        max_iter=10000,
        random_state=random_state,
    )
    return Pipeline([
        ("preprocess", preprocess),
        ("clf", clf),
    ])


# --------- API runner ---------

def load(train_csv_path: str) -> None:
    """
    Sélectionne (min_frequency, drop_if_binary) sur split 80/20 (F1 pondéré),
    puis refit sur 100% du train avec les meilleurs réglages.
    """
    global _pipe

    # 1) Données
    X, y = _prepare_xy_from_csv(train_csv_path)

    # 2) Split 80/20 stratifié
    RS = 42
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=RS, stratify=y
    )

    # 3) dual dynamique pour L2 (liblinear)
    n_samples_tr, n_features_tr = X_tr.shape
    dual_flag_tr = (n_samples_tr < n_features_tr)

    # 4) Grille E7 : valeurs sobres 
    MIN_FREQS = [None, 5, 10, 20]     # None = OHE standard ; 5/10/20 = regrouper modalités rares
    DROP_BIN = [False, True]          # drop='if_binary' ou non

    best = {"min_freq": None, "drop_bin": False, "f1": -np.inf}

    for mf in MIN_FREQS:
        for db in DROP_BIN:
            pipe = _build_pipeline(min_frequency=mf, drop_if_binary=db, dual=dual_flag_tr, C=1.0, random_state=RS)
            pipe.fit(X_tr, y_tr)
            y_hat = pipe.predict(X_va)
            f1w = f1_score(y_va, y_hat, average="weighted")
            if f1w > best["f1"]:
                best.update({"min_freq": mf, "drop_bin": db, "f1": f1w})

    # 5) Refit final sur tout le train avec les meilleurs réglages
    n_samples_all, n_features_all = X.shape
    dual_flag_all = (n_samples_all < n_features_all)

    _pipe = _build_pipeline(
        min_frequency=best["min_freq"],
        drop_if_binary=best["drop_bin"],
        dual=dual_flag_all,
        C=1.0,
        random_state=RS
    )
    _pipe.fit(X, y)
    # print(f"[E7] Best min_freq={best['min_freq']}, drop_if_binary={best['drop_bin']} (valid F1_w={best['f1']:.4f}). Refit terminé.")


def predict(X: pd.DataFrame):
    if _pipe is None:
        raise RuntimeError("Modèle non chargé")
    return _pipe.predict(X)
