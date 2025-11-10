# E5 — Renforcement ciblé de la classe 2 (No-Show) en pondérant les échantillons.
# - Split 80/20 stratifié: on teste une grille d'alphas pour la classe 2
#   , on choisit celle qui maximise F1 pondéré.
# - Puis refit sur 100% du train avec l'alpha retenu.

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

_pipe = None  # pipeline finale entraînée avec le meilleur alpha

# --------- Helpers ---------

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
    # min_frequency réduit le bruit des catégories rares
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


def _build_pipeline(C: float = 1.0, loss: str = "squared_hinge",
                    dual: bool = True, random_state: int = 42):
    preprocess = _make_preprocess()
    clf = LinearSVC(
        C=C,
        loss=loss,
        penalty="l2",           
        dual=dual,              
        class_weight=None,      
        tol=1e-3,
        max_iter=10000,
        random_state=random_state,
    )
    return Pipeline([
        ("preprocess", preprocess),
        ("clf", clf),
    ])


def _balanced_per_class_weights(y: np.ndarray):
    # Règle balanced : n_samples / (n_classes * n_samples_c)
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weights = {c: n_samples / (n_classes * cnt) for c, cnt in zip(classes, counts)}
    return weights


def _make_sample_weight(y: np.ndarray, alpha_for_class2: float = 1.0):
    # Poids "balanced" par classe, puis on multiplie la classe 2 par alpha
    w_class = _balanced_per_class_weights(y)
    w = np.array([w_class[int(c)] for c in y], dtype=float)
    if alpha_for_class2 != 1.0:
        w[y == 2] *= alpha_for_class2
    return w


# --------- API runner ---------

def load(train_csv_path: str) -> None:
    """
    Cherche le meilleur alpha (poids multiplicatif sur la classe 2 appliqué par sample_weight)
    sur un split 80/20, puis refit sur l'ensemble du train avec cet alpha.
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

    # 4) Grille d'alphas 
    ALPHAS = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0]  

    best_alpha = None
    best_f1 = -np.inf

    for a in ALPHAS:
        pipe = _build_pipeline(C=1.0, loss="squared_hinge", dual=dual_flag_tr, random_state=RS)
        w_tr = _make_sample_weight(y_tr.values, alpha_for_class2=a)
        pipe.fit(X_tr, y_tr, clf__sample_weight=w_tr)

        y_hat = pipe.predict(X_va)
        f1w = f1_score(y_va, y_hat, average="weighted")

        if f1w > best_f1:
            best_f1 = f1w
            best_alpha = a

    if best_alpha is None:
        best_alpha = 1.0  # fallback

    # 5) Refit final sur 100% du train avec l'alpha retenu
    n_samples_all, n_features_all = X.shape
    dual_flag_all = (n_samples_all < n_features_all)

    _pipe = _build_pipeline(C=1.0, loss="squared_hinge", dual=dual_flag_all, random_state=RS)
    w_all = _make_sample_weight(y.values, alpha_for_class2=best_alpha)
    _pipe.fit(X, y, clf__sample_weight=w_all)
    # print(f"[E5] Best alpha={best_alpha} (valid F1_w={best_f1:.4f}). Refit terminé.")


def predict(X: pd.DataFrame):
    if _pipe is None:
        raise RuntimeError("Modèle non chargé")
    return _pipe.predict(X)
