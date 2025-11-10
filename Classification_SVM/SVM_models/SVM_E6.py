# E6 — Sur-échantillonnage léger de la classe 2 (No-Show) via RandomOverSampler.
# - Split 80/20 stratifié pour sélectionner (ratio_2_to_0, C), puis refit sur 100% du train.
# - Oversampling dans la pipeline (sampler -> preprocess -> clf) pour éviter toute fuite.
# - Correctifs: tol=1e-3, max_iter=10000, dual dynamique, OHE(min_frequency=10)


import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

_pipe = None  # pipeline finale 


# ---------- Helpers ----------

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
            raise ValueError("Valeurs cibles non reconnues pour (0/1/2).")
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


def _make_sampling_strategy(y: np.ndarray, ratio_2_to_0: float):
    """Cible : n2_cible = max(n2, int(ratio * n0)) — on ne change pas 0/1, on augmente seulement 2."""
    uniq, cnts = np.unique(y, return_counts=True)
    counts = {int(c): int(n) for c, n in zip(uniq, cnts)}
    n0 = counts.get(0, 0)
    n2 = counts.get(2, 0)
    # cible pour la classe 2 
    target_n2 = max(n2, int(round(ratio_2_to_0 * n0)))
    # RandomOverSampler attend un dict des classes à suréchantillonner
    # 
    if target_n2 <= n2:
        # rien à faire 
        return {2: n2}
    return {2: target_n2}


def _build_pipeline(C: float,
                    dual: bool,
                    sampling_strategy: dict,
                    random_state: int = 42):
    preprocess = _make_preprocess()
    sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    clf = LinearSVC(
        C=C,
        loss="squared_hinge",
        penalty="l2",
        dual=dual,               
        class_weight=None,       
        tol=1e-3,
        max_iter=10000,
        random_state=random_state,
    )
    # sampler -> preprocess -> clf 
    return ImbPipeline([
        ("sampler", sampler),
        ("preprocess", preprocess),
        ("clf", clf),
    ])


# ---------- API runner ----------

def load(train_csv_path: str) -> None:
    """
    Sélectionne (ratio_2_to_0, C) sur split 80/20 (F1 pondéré), puis refit sur 100% du train.
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

    # Grilles 
    RATIOS = [0.05, 0.10, 0.20]   # n2 ≈ {5%, 10%, 20%} de n0
    C_GRID = [0.5, 1.0]

    best = {"ratio": None, "C": None, "f1": -np.inf}

    for r in RATIOS:
        strat = _make_sampling_strategy(y_tr.values, r)
        for C in C_GRID:
            pipe = _build_pipeline(C=C, dual=dual_flag_tr, sampling_strategy=strat, random_state=RS)
            pipe.fit(X_tr, y_tr)
            y_hat = pipe.predict(X_va)
            f1w = f1_score(y_va, y_hat, average="weighted")
            if f1w > best["f1"]:
                best.update({"ratio": r, "C": C, "f1": f1w})

    # Refit final sur tout le train 
    if best["ratio"] is None:
        best["ratio"], best["C"] = 0.10, 1.0 
    n_samples_all, n_features_all = X.shape
    dual_flag_all = (n_samples_all < n_features_all)

    strat_all = _make_sampling_strategy(y.values, best["ratio"])
    _pipe = _build_pipeline(C=best["C"], dual=dual_flag_all, sampling_strategy=strat_all, random_state=RS)
    _pipe.fit(X, y)
    # print(f"[E6] Best ratio={best['ratio']}, C={best['C']} (valid F1_w={best['f1']:.4f}). Refit terminé.")


def predict(X: pd.DataFrame):
    if _pipe is None:
        raise RuntimeError("Modèle non chargé")
    return _pipe.predict(X)
