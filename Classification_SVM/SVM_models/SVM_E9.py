# E9 — Réduction de l'asymétrie des variables numériques :
#       - Clipping (p_low, p_high) sur chaque numérique
#       - log1p sélectif pour les numériques positives très asymétriques

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC


_pipe = None  # pipeline finale (SkewFix -> preprocess -> LinearSVC)


# ---------- Helpers ----------

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


class SkewClipLog(BaseEstimator, TransformerMixin):
    """
    Clipping quantile par colonne numérique + log1p sélectif pour les colonnes très asymétriques.
    - p_low/p_high: quantiles de clipping (ex: 1 et 99)
    - skew_thresh: seuil d'asymétrie (skewness absolue > seuil) au-dessus duquel on applique log1p,
                   seulement si la colonne est >= 0 (min >= 0).
    """
    def __init__(self, p_low: float = 1.0, p_high: float = 99.0, skew_thresh: float = 1.0):
        assert 0 <= p_low < p_high <= 100
        self.p_low = p_low
        self.p_high = p_high
        self.skew_thresh = skew_thresh
        self._num_cols_: Optional[List[str]] = None
        self._bounds_: Dict[str, Tuple[float, float]] = {}
        self._log_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        num_cols = X.select_dtypes(include=np.number).columns.tolist()
        self._num_cols_ = num_cols

        # bornes de clipping + détection des colonnes à log1p
        self._bounds_.clear()
        self._log_cols_.clear()
        if len(num_cols) == 0:
            return self

        q_low = X[num_cols].quantile(self.p_low / 100.0)
        q_high = X[num_cols].quantile(self.p_high / 100.0)

        
        # NB: robustesse si col constante
        skew = X[num_cols].skew(numeric_only=True)

        for col in num_cols:
            lo = float(q_low.get(col, np.nan))
            hi = float(q_high.get(col, np.nan))

            if np.isnan(lo) or np.isnan(hi) or lo >= hi:
                # bornes dégénérées: ignorer clipping pour cette colonne
                continue
            self._bounds_[col] = (lo, hi)

            # log1p seulement si min >= 0 et asymétrie suffisante
            col_min = float(X[col].min())
            col_skew = float(skew.get(col, 0.0))
            if col_min >= 0.0 and abs(col_skew) > self.skew_thresh:
                self._log_cols_.append(col)

        return self

    def transform(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_out = X.copy()

        # clipping
        for col, (lo, hi) in self._bounds_.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].clip(lower=lo, upper=hi)

        # log1p sélectif
        for col in self._log_cols_:
            if col in X_out.columns:
                # sécurité si des valeurs négatives apparaissaient en inference
                X_out[col] = np.log1p(np.clip(X_out[col], a_min=0, a_max=None))

        return X_out


def _make_preprocess():
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),          # par sécurité
                ("scaler", StandardScaler(with_mean=False)),
            ]), selector(dtype_include=np.number)),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
            ]), selector(dtype_exclude=np.number)),
        ],
        sparse_threshold=1.0
    )


def _build_pipeline(p_low: float, p_high: float, skew_thresh: float, dual: bool,
                    C: float = 1.0, random_state: int = 42):
    skewfix = SkewClipLog(p_low=p_low, p_high=p_high, skew_thresh=skew_thresh)
    preprocess = _make_preprocess()
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
        ("skewfix", skewfix),
        ("preprocess", preprocess),
        ("clf", clf),
    ])


# ---------- API runner ----------

def load(train_csv_path: str) -> None:
    """
    Sélectionne (p_low/p_high, skew_thresh) sur split 80/20 (F1 pondéré),
    puis refit sur 100% du train avec les meilleurs réglages.
    """
    global _pipe

    # Données
    X, y = _prepare_xy_from_csv(train_csv_path)

    # Split
    RS = 42
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=RS, stratify=y
    )

    # dual dynamique pour L2 (liblinear) — calculé sur le split train
    n_samples_tr, n_features_tr = X_tr.shape
    dual_flag_tr = (n_samples_tr < n_features_tr)

    # Grille 
    CLIPS = [(1.0, 99.0), (2.0, 98.0), (5.0, 95.0)]
    SKEWS = [0.8, 1.0, 1.5]

    best = {"clip": None, "skew": None, "f1": -np.inf}

    for (pl, ph) in CLIPS:
        for sk in SKEWS:
            pipe = _build_pipeline(p_low=pl, p_high=ph, skew_thresh=sk, dual=dual_flag_tr, C=1.0, random_state=RS)
            pipe.fit(X_tr, y_tr)
            y_hat = pipe.predict(X_va)
            f1w = f1_score(y_va, y_hat, average="weighted")
            if f1w > best["f1"]:
                best.update({"clip": (pl, ph), "skew": sk, "f1": f1w})

    # Refit final sur tout le train
    if best["clip"] is None:
        best["clip"] = (1.0, 99.0)
        best["skew"] = 1.0

    n_samples_all, n_features_all = X.shape
    dual_flag_all = (n_samples_all < n_features_all)

    _pipe = _build_pipeline(
        p_low=best["clip"][0],
        p_high=best["clip"][1],
        skew_thresh=best["skew"],
        dual=dual_flag_all,
        C=1.0,
        random_state=RS
    )
    _pipe.fit(X, y)
    # print(f"[E9] Best clip={best['clip']}, skew_thresh={best['skew']} (valid F1_w={best['f1']:.4f}). Refit terminé.")


def predict(X: pd.DataFrame):
    if _pipe is None:
        raise RuntimeError("Modèle non chargé")
    return _pipe.predict(X)
