# E15 — Non-linéarité RBF scalable avec RBFSampler (numériques) + features métier (E11) + clipping/log (E9)
# Grid: use_cats ∈ {True, False} × n_components × gamma × C


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
from sklearn.kernel_approximation import RBFSampler

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


# ---------- E11 : features métier ----------

def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

class HotelFeatureMaker(BaseEstimator, TransformerMixin):
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

        # coûts/ratios
        X["adr_total"] = (adr.fillna(0) * nights_total.fillna(0))
        denom = guests_total.clip(lower=1)
        X["adr_per_guest"] = adr.fillna(0) / denom

        denom_prev = (prev_c.fillna(0) + prev_nc.fillna(0) + 1e-6)
        X["prev_cancel_ratio"] = (prev_c.fillna(0) / denom_prev)

        # lead_time
        if "lead_time" in X.columns:
            X["lead_time_log"] = np.log1p(lead_time.clip(lower=0))

        return X


# ---------- E9 : clipping + log1p sélectif ----------

class SkewClipLog(BaseEstimator, TransformerMixin):
    def __init__(self, p_low: float = 1.0, p_high: float = 99.0, skew_thresh: float = 1.0):
        assert 0 <= p_low < p_high <= 100
        self.p_low = p_low
        self.p_high = p_high
        self.skew_thresh = skew_thresh
        self._bounds_: Dict[str, Tuple[float, float]] = {}
        self._log_cols_: List[str] = []

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        num_cols = X.select_dtypes(include=np.number).columns.tolist()
        self._bounds_.clear(); self._log_cols_.clear()
        if len(num_cols) == 0:
            return self
        q_low = X[num_cols].quantile(self.p_low / 100.0)
        q_high = X[num_cols].quantile(self.p_high / 100.0)
        skew = X[num_cols].skew(numeric_only=True)
        for col in num_cols:
            lo = float(q_low.get(col, np.nan)); hi = float(q_high.get(col, np.nan))
            if not np.isnan(lo) and not np.isnan(hi) and lo < hi:
                self._bounds_[col] = (lo, hi)
                col_min = float(X[col].min()); col_skew = float(skew.get(col, 0.0))
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

def _build_pipeline(use_cats: bool,
                    n_components: int,
                    gamma: float,
                    C: float,
                    random_state: int = 42):
    """
    - hotel_features -> skewfix
    - numeric branch: impute + scale (dense) + RBFSampler(n_components, gamma)
    - optional categorical branch: impute + OHE(min_frequency=10)
    - classifier: LinearSVC (class_weight='balanced')
    """
    feats = HotelFeatureMaker()
    skewfix = SkewClipLog(p_low=2.0, p_high=98.0, skew_thresh=1.0)

    # Prétraitements séparés
    num_pre = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("rbf", RBFSampler(gamma=gamma, n_components=n_components, random_state=random_state)),
    ])

    transformers = [
        ("num_rbf", num_pre, selector(dtype_include=np.number)),
    ]
    if use_cats:
        transformers.append((
            "cat_ohe",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
            ]),
            selector(dtype_exclude=np.number),
        ))


    pre = ColumnTransformer(transformers=transformers)

    clf = LinearSVC(
        C=C,
        loss="squared_hinge",
        penalty="l2",
        dual=True,                
        class_weight="balanced",
        tol=1e-3,
        max_iter=20000,
        random_state=random_state,
    )

    return Pipeline([
        ("hotel_features", feats),
        ("skewfix", skewfix),
        ("pre", pre),
        ("clf", clf),
    ])


# ---------- API runner ----------

def load(train_csv_path: str) -> None:
    """
    Grid (use_cats, n_components, gamma, C) sur split 80/20 (F1 pondéré),
    puis refit sur 100% du train avec la meilleure config.
    """
    global _pipe

    # Données
    X, y = _prepare_xy_from_csv(train_csv_path)

    # Split
    RS = 42
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=RS, stratify=y
    )

    # Grille
    USE_CATS = [True, False]          # True = on concatène OHE aux features RBF
    NCOMP = [300, 600]                # nombre de composantes RFF
    GAMMA = [0.5, 1.0, 2.0]           # largeur RBF (données standardisées)
    C_GRID = [0.5, 1.0, 2.0]          # régularisation du SVM linéaire en feature-space RBF

    best = {"use_cats": None, "n": None, "g": None, "C": None, "f1": -np.inf}

    for uc in USE_CATS:
        for n in NCOMP:
            for g in GAMMA:
                for C in C_GRID:
                    pipe = _build_pipeline(use_cats=uc, n_components=n, gamma=g, C=C, random_state=RS)
                    pipe.fit(X_tr, y_tr)
                    y_hat = pipe.predict(X_va)
                    f1w = f1_score(y_va, y_hat, average="weighted")
                    if f1w > best["f1"]:
                        best.update({"use_cats": uc, "n": n, "g": g, "C": C, "f1": f1w})

    # Refit final sur 100% du train avec la meilleure configuration
    _pipe = _build_pipeline(
        use_cats=best["use_cats"],
        n_components=best["n"],
        gamma=best["g"],
        C=best["C"],
        random_state=RS
    )
    _pipe.fit(X, y)
    # print(f"[E15] Best use_cats={best['use_cats']}, n={best['n']}, gamma={best['g']}, C={best['C']} (valid F1_w={best['f1']:.4f}).")

def predict(X: pd.DataFrame):
    if _pipe is None:
        raise RuntimeError("Modèle non chargé")
    return _pipe.predict(X)
