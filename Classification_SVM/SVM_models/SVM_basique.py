import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC

_pipe = None  # pipeline global 

def _prepare_xy_from_csv(path: str,
                         target_col: str = "reservation_status",
                         id_col: str = "row_id"):
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"Colonne cible '{target_col}' absente de {path}.")

    y = df[target_col]
    # Mappe si la cible est textuelle
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

def load(train_csv_path: str) -> None:
    """
    Entraîne le modèle directement à partir d'un CSV d'entraînement.
    """
    global _pipe
    X, y = _prepare_xy_from_csv(train_csv_path)

    num_sel = selector(dtype_include=np.number)
    cat_sel = selector(dtype_exclude=np.number)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]), num_sel),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_sel),
        ],
        sparse_threshold=1.0
    )

    clf = LinearSVC(
        class_weight="balanced",
        C=1.0,
        max_iter=2000,
        random_state=42
    )

    _pipe = Pipeline([
        ("preprocess", preprocess),
        ("clf", clf)
    ])
    _pipe.fit(X, y)

def predict(X: pd.DataFrame):
    """
    X : DataFrame de features (pas de row_id ni target).
    Retourne un array-like d'entiers {0,1,2}.
    """
    if _pipe is None:
        raise RuntimeError("Modèle non chargé")
    return _pipe.predict(X)
