
# E11 — Features métier (hôtellerie) + LinearSVC.
# - Ajoute des colonnes dérivées (nuits/clients/coûts/flags/ratios) via un transformer scikit-learn.
# - Petite grille (use_total_cost, use_prev_ratio) sur split 80/20, puis refit sur 100% du train.
# - Correctifs LinearSVC : tol=1e-3, max_iter=10000, dual dynamique. OHE(min_frequency=10).

from typing import Optional, Dict, Tuple
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

_pipe = None  # pipeline finale

# ---------------- Helpers ----------------

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
            raise ValueError("Valeurs cibles non reconnues (0/1/2).")
    drop_cols = [target_col]
    if id_col in df.columns:
        drop_cols.append(id_col)
    X = df.drop(columns=drop_cols, errors="ignore")
    return X, y.astype(int)

def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _series_or_default(X: pd.DataFrame, col: str, default=np.nan, dtype=None) -> pd.Series:
    """Retourne X[col] si présent, sinon une Series de 'default' indexée comme X."""
    if col in X.columns:
        s = X[col]
    else:
        s = pd.Series(default, index=X.index)
    if dtype is not None:
        try:
            s = s.astype(dtype)
        except Exception:
            pass
    return s


class HotelFeatureMaker(BaseEstimator, TransformerMixin):
    """
    Ajoute des features métier robustes :
      - nights_total = stays_in_weekend_nights + stays_in_week_nights
      - guests_total = adults + children + babies
      - adr_total = adr * nights_total ; adr_per_guest = adr / max(guests_total, 1) (si use_total_cost=True)
      - room_changed = (reserved_room_type != assigned_room_type) (si les deux existent)
      - weekend_stay = (stays_in_weekend_nights > 0) ; long_stay = (nights_total >= 7)
      - has_agent / has_company (présence non nulle)
      - deposit_flag = (deposit_type != 'No Deposit')
      - prev_cancel_ratio = prev_cancel / (prev_cancel + prev_not_cancel + ε) (si use_prev_ratio=True)
      - lead_time_log = log1p(lead_time) si colonne présente
    """
    def __init__(self, use_total_cost: bool = True, use_prev_ratio: bool = True):
        self.use_total_cost = use_total_cost
        self.use_prev_ratio = use_prev_ratio

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        # Bases numériques
        we = _num(X.get("stays_in_weekend_nights", 0))
        wk = _num(X.get("stays_in_week_nights", 0))
        adults = _num(X.get("adults", 0))
        children = _num(X.get("children", 0))
        babies = _num(X.get("babies", 0))
        adr = _num(X.get("adr", np.nan))
        lead_time = _num(X.get("lead_time", np.nan))
        prev_c = _num(X.get("previous_cancellations", 0))
        prev_nc = _num(X.get("previous_bookings_not_canceled", 0))

        nights_total = (we.fillna(0) + wk.fillna(0))
        guests_total = (adults.fillna(0) + children.fillna(0) + babies.fillna(0))

        X["nights_total"] = nights_total
        X["guests_total"] = guests_total
        X["weekend_stay"] = (we.fillna(0) > 0).astype("Int8")
        X["long_stay"] = (nights_total >= 7).astype("Int8")

        # Flags
        s_res = _series_or_default(X, "reserved_room_type", default="")
        s_ass = _series_or_default(X, "assigned_room_type", default="")
        X["room_changed"] = (s_res.astype(str) != s_ass.astype(str)).astype(np.int8)
        s_agent = _series_or_default(X, "agent", default=np.nan)
        s_company = _series_or_default(X, "company", default=np.nan)
        X["has_agent"] = (~s_agent.isna()).astype(np.int8)
        X["has_company"] = (~s_company.isna()).astype(np.int8)

        s_dep = _series_or_default(X, "deposit_type", default="")
        X["deposit_flag"] = (s_dep.astype(str) != "No Deposit").astype(np.int8)

        # Transformations numériques utiles
        if self.use_total_cost and "adr" in X.columns:
            # adr_total ~ estimation du panier (si nights_total > 0)
            X["adr_total"] = (adr.fillna(0) * nights_total.fillna(0))
            # adr_per_guest : normalisé par le nombre de clients
            denom = guests_total.clip(lower=1)  # évite /0
            X["adr_per_guest"] = adr.fillna(0) / denom

        if self.use_prev_ratio:
            denom_prev = (prev_c.fillna(0) + prev_nc.fillna(0) + 1e-6)
            X["prev_cancel_ratio"] = (prev_c.fillna(0) / denom_prev)

        if "lead_time" in X.columns:
            X["lead_time_log"] = np.log1p(lead_time.clip(lower=0))  # log1p seulement si >=0

        return X

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

def _build_pipeline(use_total_cost: bool, use_prev_ratio: bool, dual: bool,
                    C: float = 1.0, random_state: int = 42):
    feats = HotelFeatureMaker(use_total_cost=use_total_cost, use_prev_ratio=use_prev_ratio)
    preprocess = _make_preprocess()
    clf = LinearSVC(
        C=C,
        loss="squared_hinge",
        penalty="l2",
        dual=dual,               # dynamique selon n_samples vs n_features
        class_weight="balanced",
        tol=1e-3,
        max_iter=10000,
        random_state=random_state,
    )
    return Pipeline([
        ("hotel_features", feats),
        ("preprocess", preprocess),
        ("clf", clf),
    ])

# ---------------- API runner ----------------

def load(train_csv_path: str) -> None:
    """
    Sélectionne (use_total_cost, use_prev_ratio) sur split 80/20 (F1 pondéré),
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

    # dual dynamique (liblinear) — calculé sur X_tr
    n_samples_tr, n_features_tr = X_tr.shape
    dual_flag_tr = (n_samples_tr < n_features_tr)

    # Grille
    USE_TOTAL = [False, True]
    USE_RATIO = [False, True]

    best = {"use_total_cost": None, "use_prev_ratio": None, "f1": -np.inf}

    for ut in USE_TOTAL:
        for ur in USE_RATIO:
            pipe = _build_pipeline(use_total_cost=ut, use_prev_ratio=ur, dual=dual_flag_tr, C=1.0, random_state=RS)
            pipe.fit(X_tr, y_tr)
            y_hat = pipe.predict(X_va)
            f1w = f1_score(y_va, y_hat, average="weighted")
            if f1w > best["f1"]:
                best.update({"use_total_cost": ut, "use_prev_ratio": ur, "f1": f1w})

    # Refit final sur 100% du train
    n_samples_all, n_features_all = X.shape
    dual_flag_all = (n_samples_all < n_features_all)

    _pipe = _build_pipeline(
        use_total_cost=best["use_total_cost"],
        use_prev_ratio=best["use_prev_ratio"],
        dual=dual_flag_all,
        C=1.0,
        random_state=RS
    )
    _pipe.fit(X, y)
    # print(f"[E11] Best use_total_cost={best['use_total_cost']}, use_prev_ratio={best['use_prev_ratio']} (valid F1_w={best['f1']:.4f}). Refit terminé.")

def predict(X: pd.DataFrame):
    if _pipe is None:
        raise RuntimeError("Modèle non chargé")
    return _pipe.predict(X)
