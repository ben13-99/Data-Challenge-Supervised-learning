# E10 — Features temporelles : on reconstruit l'arrivée à partir
# des colonnes existantes (arrival_date_year / _month / _day_of_month).

from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import calendar

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

_pipe = None

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

def _month_to_num(s: pd.Series) -> pd.Series:
    """Mappe 'January'→1, 'February'→2, ... ; gère aussi des noms FR ('Janvier', etc.)."""
    eng = {m: i for i, m in enumerate(calendar.month_name) if m}
    eng_abbr = {m: i for i, m in enumerate(calendar.month_abbr) if m}
    fr = {
        "janvier":1,"février":2,"fevrier":2,"mars":3,"avril":4,"mai":5,"juin":6,
        "juillet":7,"août":8,"aout":8,"septembre":9,"octobre":10,"novembre":11,"décembre":12,"decembre":12
    }
    def conv(x):
        if pd.isna(x): return np.nan
        if isinstance(x, (int, np.integer, float, np.floating)):
            v = int(x)
            return v if 1 <= v <= 12 else np.nan
        t = str(x).strip()
        # essais en anglais
        if t in eng: return eng[t]
        if t in eng_abbr: return eng_abbr[t]
        # essais en français (lower)
        tl = t.lower()
        if tl in fr: return fr[tl]
        # numeric string ?
        if t.isdigit():
            v = int(t)
            return v if 1 <= v <= 12 else np.nan
        return np.nan
    return s.map(conv)

def _season_from_month_int(m: pd.Series) -> pd.Series:
    # Hiver:12-1-2(0), Printemps:3-4-5(1), Été:6-7-8(2), Automne:9-10-11(3)
    return m.map(lambda x: 0 if x in (12,1,2) else 1 if x in (3,4,5) else 2 if x in (6,7,8) else 3 if x in (9,10,11) else np.nan)

class TemporalFeatureMaker(BaseEstimator, TransformerMixin):
    """
    Construit un datetime d'arrivée à partir de:
      - arrival_date_year (int)
      - arrival_date_month (nom du mois ou numéro)
      - arrival_date_day_of_month (int)
    Puis dérive: dow, is_weekend, week, season, encodage cyclique.
    """
    def __init__(self, cyclical: bool = False, use_season: bool = True):
        self.cyclical = cyclical
        self.use_season = use_season
        self.available_: bool = False

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        cols = {"arrival_date_year","arrival_date_month","arrival_date_day_of_month"}
        self.available_ = cols.issubset(set(X.columns))
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if not self.available_:
            return X

        year = pd.to_numeric(X["arrival_date_year"], errors="coerce")
        month = _month_to_num(X["arrival_date_month"])
        day = pd.to_numeric(X["arrival_date_day_of_month"], errors="coerce")

        # date d'arrivée (valeurs non plausibles -> NaT)
        arr = pd.to_datetime(
            dict(year=year, month=month, day=day),
            errors="coerce"
        )

        # dérivées
        dow = arr.dt.dayofweek  # 0=Mon..6=Sun
        X["arrival_dow"] = dow.astype("Int64")
        X["arrival_is_weekend"] = dow.isin([5,6]).astype("Int8")
        try:
            X["arrival_week"] = arr.dt.isocalendar().week.astype("Int64")
        except Exception:
            pass

        if self.use_season:
            X["arrival_season"] = _season_from_month_int(month).astype("Int64")

        if self.cyclical:
            m = month.fillna(0).astype(int).clip(0, 12)
            X["arrival_month_sin"] = np.sin(2*np.pi*(m%12)/12.0)
            X["arrival_month_cos"] = np.cos(2*np.pi*(m%12)/12.0)
            d = dow.fillna(0).astype(int).clip(0, 6)
            X["arrival_dow_sin"] = np.sin(2*np.pi*(d%7)/7.0)
            X["arrival_dow_cos"] = np.cos(2*np.pi*(d%7)/7.0)

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

def _build_pipeline(cyclical: bool, use_season: bool, dual: bool,
                    C: float = 1.0, random_state: int = 42):
    timefeats = TemporalFeatureMaker(cyclical=cyclical, use_season=use_season)
    preprocess = _make_preprocess()
    clf = LinearSVC(
        C=C,
        loss="squared_hinge",
        penalty="l2",
        dual=dual,
        class_weight="balanced",
        tol=1e-3,
        max_iter=10000,
        random_state=random_state,
    )
    return Pipeline([
        ("timefeats", timefeats),
        ("preprocess", preprocess),
        ("clf", clf),
    ])

# ---------------- API runner ----------------

def load(train_csv_path: str) -> None:
    global _pipe
    X, y = _prepare_xy_from_csv(train_csv_path)

    RS = 42
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=RS, stratify=y
    )

    # dual dynamique (liblinear)
    n_samples_tr, n_features_tr = X_tr.shape
    dual_flag_tr = (n_samples_tr < n_features_tr)

    # petite grille
    CYCL = [False, True]
    SEAS = [False, True]

    best = {"cyc": None, "sea": None, "f1": -np.inf}
    for cyc in CYCL:
        for sea in SEAS:
            pipe = _build_pipeline(cyclical=cyc, use_season=sea, dual=dual_flag_tr, C=1.0, random_state=RS)
            pipe.fit(X_tr, y_tr)
            y_hat = pipe.predict(X_va)
            f1w = f1_score(y_va, y_hat, average="weighted")
            if f1w > best["f1"]:
                best.update({"cyc": cyc, "sea": sea, "f1": f1w})

    # refit final
    n_samples_all, n_features_all = X.shape
    dual_flag_all = (n_samples_all < n_features_all)
    _pipe = _build_pipeline(
        cyclical=best["cyc"], use_season=best["sea"],
        dual=dual_flag_all, C=1.0, random_state=RS
    )
    _pipe.fit(X, y)

def predict(X: pd.DataFrame):
    if _pipe is None:
        raise RuntimeError("Modèle non chargé")
    return _pipe.predict(X)
