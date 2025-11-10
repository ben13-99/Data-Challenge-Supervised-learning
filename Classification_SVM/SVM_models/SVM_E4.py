# E4 — Variation de la perte et de la pénalité pour LinearSVC, avec correctifs de convergence.
# - Split 80/20 stratifié pour sélectionner la meilleure config (F1 pondéré), puis refit sur tout le train.
# - Changements :
#     * tol=1e-3, max_iter=10000  (convergence facilitée)
#     * dual dynamique pour L2 : dual = (n_samples < n_features)
#     * OneHotEncoder(min_frequency=10) pour réduire le bruit des catégories rares

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC


_pipe = None  # pipeline finale 


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
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
            ]), selector(dtype_exclude=np.number)),
        ],
        sparse_threshold=1.0
    )


def _build_pipeline(penalty: str, loss: str, dual: bool, C: float = 1.0, random_state: int = 42):
    """
    penalty='l2' : dual au choix (dynamique en amont) ; loss ∈ {'hinge', 'squared_hinge'}
    penalty='l1' : loss doit être 'squared_hinge' et dual=False
    """
    preprocess = _make_preprocess()
    clf = LinearSVC(
        C=C,
        loss=loss,
        penalty=penalty,
        dual=dual,
        class_weight="balanced",
        tol=1e-3,
        max_iter=10000,
        random_state=random_state,
    )
    return Pipeline([
        ("preprocess", preprocess),
        ("clf", clf),
    ])


# --- API attendue par le runner ---

def load(train_csv_path: str) -> None:
    """
    Entraîne en testant 3 configs (penalty/loss/dual) sur split 80/20,
    choisit la meilleure selon F1 pondéré, puis refit sur 100% du train.
    """
    global _pipe

    # 1) Données
    X, y = _prepare_xy_from_csv(train_csv_path)

    # 2) Split 80/20 stratifié
    RS = 42
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=RS, stratify=y
    )

    # 3) dual dynamique pour L2 sur le split
    n_samples_tr, n_features_tr = X_tr.shape
    dual_flag_L2_tr = (n_samples_tr < n_features_tr)

    # 4) Configs à tester (on fige C=1.0 pour isoler l’effet loss/penalty)
    configs = [
        # (penalty,       loss,             dual,               label lisible)
        ("l2",           "squared_hinge",  dual_flag_L2_tr,    "L2 + squared_hinge"),
        ("l2",           "hinge",          dual_flag_L2_tr,    "L2 + hinge"),
        ("l1",           "squared_hinge",  False,              "L1 + squared_hinge (dual=False)"),
    ]

    best_cfg = None
    best_f1 = -np.inf

    for penalty, loss, dual, label in configs:
        try:
            pipe = _build_pipeline(penalty=penalty, loss=loss, dual=dual, C=1.0, random_state=RS)
            pipe.fit(X_tr, y_tr)
            y_hat = pipe.predict(X_va)
            f1w = f1_score(y_va, y_hat, average="weighted")
            # print(f"[E4] {label}: F1_w={f1w:.4f}")
            if f1w > best_f1:
                best_f1 = f1w
                best_cfg = (penalty, loss, dual)
        except Exception:
            continue

    if best_cfg is None:
        # Fallback
        best_cfg = ("l2", "squared_hinge", dual_flag_L2_tr)

    # 5) Refit final sur 100% du train avec la meilleure config
    penalty, loss, _dual_unused = best_cfg

    # Recalcule le dual dynamique sur TOUT le train pour L2
    n_samples_all, n_features_all = X.shape
    dual_flag_L2_all = (n_samples_all < n_features_all)

    dual_final = False if penalty == "l1" else dual_flag_L2_all
    _pipe = _build_pipeline(penalty=penalty, loss=loss, dual=dual_final, C=1.0, random_state=RS)
    _pipe.fit(X, y)
    # print(f"[E4] Best: {best_label} (valid F1_w={best_f1:.4f}). Refit sur tout le train terminé.")


def predict(X: pd.DataFrame):
    """Prédit 0/1/2 à partir d’un DataFrame de features (sans row_id ni target)."""
    if _pipe is None:
        raise RuntimeError("Modèle non chargé")
    return _pipe.predict(X)
