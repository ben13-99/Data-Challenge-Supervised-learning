# E12 — Combinaison des gains de E9 (clipping + log1p sélectif) et E11 (features métier)
#        + LinearSVC robuste (tol, itérations, dual dynamique).
# - Grid légère sur split 80/20: (use_total_cost, use_prev_ratio) × (clip) × C

from typing import Optional, Dict, Tuple, List
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


# ---------- Helpers communs ----------

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


# ---------- features métier ----------

def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

class HotelFeatureMaker(BaseEstimator, TransformerMixin):
    """
    Ajoute des features hôtellerie :
      - nights_total, guests_total, weekend_stay, long_stay
      - room_changed, has_agent, has_company, deposit_flag
      - adr_total, adr_per_guest (si use_total_cost)
      - prev_cancel_ratio (si use_prev_ratio)
      - lead_time_log (si lead_time existe)
    """
    def __init__(self, use_total_cost: bool = True, use_prev_ratio: bool = True):
        self.use_total_cost = use_total_cost
        self.use_prev_ratio = use_prev_ratio

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        we = _num(_series_or_default(X, "stays_in_weekend_nights", 0))
        wk = _num(_series_or_default(X, "stays_in_week_nights", 0))
        adults = _num(_series_or_default(X, "adults", 0))
        children = _num(_series_or_default(X, "children", 0))
        babies = _num(_series_or_default(X, "babies", 0))
        adr = _num(_series_or_default(X, "adr", np.nan))
        lead_time = _num(_series_or_default(X, "lead_time", np.nan))
        prev_c = _num(_series_or_default(X, "previous_cancellations", 0))
        prev_nc = _num(_series_or_default(X, "previous_bookings_not_canceled", 0))

        nights_total = (we.fillna(0) + wk.fillna(0))
        guests_total = (adults.fillna(0) + children.fillna(0) + babies.fillna(0))

        X["nights_total"] = nights_total
        X["guests_total"] = guests_total
        X["weekend_stay"] = (we.fillna(0) > 0).astype(np.int8)
        X["long_stay"] = (nights_total >= 7).astype(np.int8)

        s_res = _series_or_default(X, "reserved_room_type", "")
        s_ass = _series_or_default(X, "assigned_room_type", "")
        X["room_changed"] = (s_res.astype(str) != s_ass.astype(str)).astype(np.int8)

        s_agent = _series_or_default(X, "agent", np.nan)
        s_company = _series_or_default(X, "company", np.nan)
        X["has_agent"] = (~s_agent.isna()).astype(np.int8)
        X["has_company"] = (~s_company.isna()).astype(np.int8)

        s_dep = _series_or_default(X, "deposit_type", "")
        X["deposit_flag"] = (s_dep.astype(str) != "No Deposit").astype(np.int8)

        if self.use_total_cost:
            X["adr_total"] = (adr.fillna(0) * nights_total.fillna(0))
            denom = guests_total.clip(lower=1)  # évite /0
            X["adr_per_guest"] = adr.fillna(0) / denom

        if self.use_prev_ratio:
            denom_prev = (prev_c.fillna(0) + prev_nc.fillna(0) + 1e-6)
            X["prev_cancel_ratio"] = (prev_c.fillna(0) / denom_prev)

        if "lead_time" in X.columns:
            X["lead_time_log"] = np.log1p(lead_time.clip(lower=0))

        return X


# ---------- clipping + log1p sélectif ----------

class SkewClipLog(BaseEstimator, TransformerMixin):
    """
    Clipping quantile (p_low/p_high) sur numériques + log1p sur numériques positives très asymétriques.
    """
    def __init__(self, p_low: float = 1.0, p_high: float = 99.0, skew_thresh: float = 1.0):
        assert 0 <= p_low < p_high <= 100
        self.p_low = p_low
        self.p_high = p_high
        self.skew_thresh = skew_thresh
        self._num_cols_: Optional[List[str]] = None
        self._bounds_: Dict[str, Tuple[float, float]] = {}
        self._log_cols_: List[str] = []

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        num_cols = X.select_dtypes(include=np.number).columns.tolist()
        self._num_cols_ = num_cols
        self._bounds_.clear()
        self._log_cols_.clear()
        if len(num_cols) == 0:
            return self
        q_low = X[num_cols].quantile(self.p_low / 100.0)
        q_high = X[num_cols].quantile(self.p_high / 100.0)
        skew = X[num_cols].skew(numeric_only=True)
        for col in num_cols:
            lo = float(q_low.get(col, np.nan))
            hi = float(q_high.get(col, np.nan))
            if not np.isnan(lo) and not np.isnan(hi) and lo < hi:
                self._bounds_[col] = (lo, hi)
                col_min = float(X[col].min())
                col_skew = float(skew.get(col, 0.0))
                if col_min >= 0.0 and abs(col_skew) > self.skew_thresh:
                    self._log_cols_.append(col)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col, (lo, hi) in self._bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lo, upper=hi)
        for col in self._log_cols_:
            if col in X.columns:
                X[col] = np.log1p(np.clip(X[col], a_min=0, a_max=None))
        return X


# ---------- pipeline ----------

def _build_pipeline(use_total_cost: bool, use_prev_ratio: bool,
                    p_low: float, p_high: float, skew_thresh: float,
                    dual: bool, C: float = 1.0, random_state: int = 42):
    feats = HotelFeatureMaker(use_total_cost=use_total_cost, use_prev_ratio=use_prev_ratio)
    skewfix = SkewClipLog(p_low=p_low, p_high=p_high, skew_thresh=skew_thresh)
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
        ("skewfix", skewfix),
        ("preprocess", preprocess),
        ("clf", clf),
    ])


# ---------- API runner ----------

def load(train_csv_path: str) -> None:
    """
    Sélectionne (use_total_cost, use_prev_ratio, clip, C) sur split 80/20 (F1 pondéré),
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

    # dual dynamique (liblinear)
    n_samples_tr, n_features_tr = X_tr.shape
    dual_flag_tr = (n_samples_tr < n_features_tr)

    # Grille
    USE_TOTAL = [False, True]
    USE_RATIO = [False, True]
    CLIPS = [(1.0, 99.0), (2.0, 98.0)]   # good defaults issus de E9
    C_GRID = [0.5, 1.0, 2.0]

    best = {"ut": None, "ur": None, "clip": None, "C": None, "f1": -np.inf}

    for ut in USE_TOTAL:
        for ur in USE_RATIO:
            for (pl, ph) in CLIPS:
                for sk in SKEW:
                    for C in C_GRID:
                        pipe = _build_pipeline(
                            use_total_cost=ut, use_prev_ratio=ur,
                            p_low=pl, p_high=ph, skew_thresh=sk,
                            dual=dual_flag_tr, C=C, random_state=RS
                        )
                        pipe.fit(X_tr, y_tr)
                        y_hat = pipe.predict(X_va)
                        f1w = f1_score(y_va, y_hat, average="weighted")
                        if f1w > best["f1"]:
                            best.update({"ut": ut, "ur": ur, "clip": (pl, ph), "C": C, "f1": f1w})

    # Refit final sur tout le train
    n_samples_all, n_features_all = X.shape
    dual_flag_all = (n_samples_all < n_features_all)
    _pipe = _build_pipeline(
        use_total_cost=best["ut"],
        use_prev_ratio=best["ur"],
        p_low=best["clip"][0],
        p_high=best["clip"][1],
        skew_thresh=1.0,
        dual=dual_flag_all,
        C=best["C"],
        random_state=RS
    )
    _pipe.fit(X, y)
    # print(f"[E12] Best ut={best['ut']}, ur={best['ur']}, clip={best['clip']}, C={best['C']} (valid F1_w={best['f1']:.4f}).")

def predict(X: pd.DataFrame):
    if _pipe is None:
        raise RuntimeError("Modèle non chargé")
    return _pipe.predict(X)

