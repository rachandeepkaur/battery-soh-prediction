"""Shared evaluation protocol for every SOH model (baselines and non-linear models).

Rules (see docs/evaluation_protocol.md):

1. Data: the 306 labelled rows (25 °C reference tests with SOH) from results/rtp_features.csv.
2. Inputs: the 42 numeric main inputs + the categorical ``schedule`` listed in
   results/preprocessed/main_feature_list.csv. Comparison / leaky columns are never model inputs.
3. Splits: leave-one-cell-out (LOCO, primary, 15 folds), leave-one-profile-out (LOPO, 14 folds) and
   random 5-fold (reference only, shows how optimistic a row-level split is). The folds are fixed and
   saved to results/evaluation_protocol/fold_assignment.csv so every teammate uses the same ones.
4. Everything that learns from data - redundancy pruning, clipping, imputation, scaling, one-hot
   encoding and hyperparameter tuning - is fitted on the training part of each outer fold only.
   Tuning uses an inner GroupKFold that groups by the same column as the outer split.
5. Hyperparameter grids are declared before looking at outer-fold scores and are not changed
   afterwards based on them, so the outer score is an honest estimate.
6. Reported metrics come from pooled out-of-fold predictions: MAE (headline), RMSE, R², max error,
   plus MAE per cell, per health class and for interpolation vs extrapolation folds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import max_error, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .modelling import RedundancyPruner

ROOT = Path(__file__).resolve().parents[1]
FEATURE_TABLE = ROOT / "results" / "rtp_features.csv"
PREPROCESSED = ROOT / "results" / "preprocessed"
OUT_DIR = ROOT / "results" / "evaluation_protocol"

TARGET = "SOH"
ID_COLS = ["cell_id", "profile", "idtest", "rtp_index"]
CATEGORICAL = ["schedule"]
HEALTH_BINS = [-np.inf, 80, 90, np.inf]
HEALTH_LABELS = ["EOL risk (<80)", "Degraded (80-90)", "Healthy (>=90)"]
# Profile-design columns used to decide whether a held-out row needs extrapolation.
DESIGN = ["lp_T_set", "lp_CC", "lp_ADC", "lp_PDC", "lp_F", "lp_SoC", "lp_dSoC"]

SCHEMES = {                      # scheme -> grouping column (None = row-level random split)
    "LOCO": "cell_id",
    "LOPO": "profile",
    "Random 5-fold": None,
}
PRIMARY_SCHEME = "LOCO"
SEED = 42


# ----------------------------------------------------------------------------- data
def load_labelled(path: Path = FEATURE_TABLE) -> pd.DataFrame:
    """306 labelled rows: 25 °C reference tests with a valid SOH, sorted by cell and check-up."""
    df = pd.read_csv(path)
    df = df[(df.rtp_T_set == 25) & df[TARGET].notna()]
    df = df.sort_values(["cell_id", "rtp_index"]).reset_index(drop=True)
    df["health_class"] = pd.cut(df[TARGET], HEALTH_BINS, right=False, labels=HEALTH_LABELS)
    return df


def main_features(path: Path = PREPROCESSED / "main_feature_list.csv") -> tuple[list[str], list[str]]:
    """(numeric main inputs, categorical inputs) exactly as defined in notebook 01."""
    feats = pd.read_csv(path)["feature"].tolist()
    numeric = [f for f in feats if f not in CATEGORICAL]
    categorical = [f for f in feats if f in CATEGORICAL]
    return numeric, categorical


def comparison_features(path: Path = PREPROCESSED / "data_dictionary.csv") -> list[str]:
    dd = pd.read_csv(path)
    return dd.loc[dd.role == "comparison", "column"].tolist()


# ----------------------------------------------------------------------------- preprocessing
class PercentileClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Cap each column at percentiles learned on the training rows (same as notebook 01)."""

    def __init__(self, lower: float = 1.0, upper: float = 99.0):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.lo_ = np.nanpercentile(X, self.lower, axis=0)
        self.hi_ = np.nanpercentile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)   # NaN stays NaN


class FoldPruner(RedundancyPruner):
    """RedundancyPruner that also passes the categorical columns through untouched."""

    def __init__(self, features, passthrough=(), group_col="cell_id", threshold=0.95):
        super().__init__(features, group_col=group_col, threshold=threshold)
        self.passthrough = passthrough

    def transform(self, X):
        return X[self.selected_ + list(self.passthrough)]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_ + list(self.passthrough), dtype=object)


def make_preprocessor(numeric, categorical=CATEGORICAL, threshold: float = 0.95,
                      clip: tuple[float, float] = (1.0, 99.0)) -> Pipeline:
    """prune redundant blocks -> [numeric: clip, median-impute, scale | categorical: mode-impute, one-hot]."""
    def is_numeric(X):
        return [c for c in X.columns if c not in categorical]

    num = Pipeline([("clip", PercentileClipper(*clip)),
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                                             sparse_output=False))])
    columns = ColumnTransformer([("num", num, is_numeric), ("cat", cat, list(categorical))],
                                verbose_feature_names_out=False)
    return Pipeline([("prune", FoldPruner(list(numeric), passthrough=list(categorical), threshold=threshold)),
                     ("columns", columns)]).set_output(transform="pandas")


def make_pipeline(model, numeric, categorical=CATEGORICAL, **prep_kw) -> Pipeline:
    """Full fold-safe pipeline. Hyperparameters of the model are addressed as ``model__<name>``."""
    return Pipeline([("prep", make_preprocessor(numeric, categorical, **prep_kw)), ("model", clone(model))])


# ----------------------------------------------------------------------------- splits
def outer_folds(df: pd.DataFrame, scheme: str, seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    col = SCHEMES[scheme]
    if col is None:
        return list(KFold(5, shuffle=True, random_state=seed).split(df))
    return list(LeaveOneGroupOut().split(df, groups=df[col].values))


def inner_cv(scheme: str, seed: int = SEED, n_splits: int = 5):
    """Inner CV for tuning: same kind of split as the outer loop, so tuning optimises the same goal."""
    if SCHEMES[scheme] is None:
        return KFold(n_splits, shuffle=True, random_state=seed)
    return GroupKFold(n_splits=n_splits)


def fold_assignment(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """One row per labelled row, one column per scheme: which outer fold the row is *tested* in."""
    out = df[ID_COLS].copy()
    for scheme in SCHEMES:
        fold = np.full(len(df), -1)
        for k, (_, te) in enumerate(outer_folds(df, scheme, seed)):
            fold[te] = k
        out[f"fold_{scheme}"] = fold
    return out


def regime_flags(df: pd.DataFrame, train_idx, test_idx) -> np.ndarray:
    """'extrapolation' if any profile-design value of a test row lies outside the training range,
    'interpolation' otherwise, 'unclassified' when the design is unknown (V212)."""
    tr = df.iloc[train_idx][DESIGN]
    te = df.iloc[test_idx][DESIGN]
    lo, hi = tr.min(), tr.max()
    outside = ((te < lo) | (te > hi)).any(axis=1)
    flags = np.where(outside, "extrapolation", "interpolation").astype(object)
    flags[te.isna().any(axis=1).to_numpy()] = "unclassified"
    return flags


# ----------------------------------------------------------------------------- models
@dataclass
class ModelSpec:
    """A model and its pre-declared tuning grid (keys without the ``model__`` prefix)."""
    name: str
    estimator: object
    param_grid: dict = field(default_factory=dict)
    family: str = ""                  # "baseline", "linear", "non-linear" - for tables only


@dataclass
class CVResult:
    model: str
    scheme: str
    oof: pd.DataFrame                 # one row per labelled row
    folds: pd.DataFrame               # one row per outer fold

    @property
    def metrics(self) -> dict:
        return pooled_metrics(self.oof) | {"model": self.model, "scheme": self.scheme}


def run_cv(df: pd.DataFrame, spec: ModelSpec, scheme: str = PRIMARY_SCHEME, numeric=None,
           categorical=CATEGORICAL, n_jobs: int = -1, seed: int = SEED, verbose: bool = True,
           prep_kw: dict | None = None) -> CVResult:
    """Nested CV: outer folds give the score, an inner grouped CV on the training rows picks hyperparameters."""
    if numeric is None:
        numeric, categorical = main_features()
    X = df[list(numeric) + list(categorical) + ["cell_id"]]
    y = df[TARGET].to_numpy(float)
    group_col = SCHEMES[scheme]
    grid = {f"model__{k}": v for k, v in spec.param_grid.items()}

    pred = np.full(len(df), np.nan)
    fold_of = np.full(len(df), -1)
    regime = np.empty(len(df), dtype=object)
    fold_rows = []
    for k, (tr, te) in enumerate(outer_folds(df, scheme, seed)):
        pipe = make_pipeline(spec.estimator, numeric, categorical, **(prep_kw or {}))
        if grid:
            search = GridSearchCV(pipe, grid, cv=inner_cv(scheme, seed), scoring="neg_mean_absolute_error",
                                  n_jobs=n_jobs, refit=True, error_score="raise")
            groups = None if group_col is None else df[group_col].to_numpy()[tr]
            search.fit(X.iloc[tr], y[tr], groups=groups)
            fitted, best = search.best_estimator_, {k.removeprefix("model__"): v
                                                    for k, v in search.best_params_.items()}
        else:
            fitted, best = pipe.fit(X.iloc[tr], y[tr]), {}
        pred[te] = fitted.predict(X.iloc[te])
        fold_of[te] = k
        regime[te] = regime_flags(df, tr, te)
        held_out = "random" if group_col is None else ", ".join(map(str, df[group_col].iloc[te].unique()))
        fold_rows.append(dict(model=spec.name, scheme=scheme, fold=k, held_out=held_out, n_test=len(te),
                              MAE=mean_absolute_error(y[te], pred[te]),
                              n_selected=len(fitted.named_steps["prep"].named_steps["prune"].selected_),
                              best_params=best))
        if verbose:
            print(f"\r{spec.name} | {scheme}: fold {k + 1}", end="", flush=True)
    if verbose:
        print()
    oof = df[ID_COLS + ["health_class"]].assign(model=spec.name, scheme=scheme, fold=fold_of,
                                                regime=regime, y_true=y, y_pred=pred)
    assert np.isfinite(pred).all(), "every labelled row must be predicted exactly once"
    return CVResult(spec.name, scheme, oof, pd.DataFrame(fold_rows))


# ----------------------------------------------------------------------------- metrics
def pooled_metrics(oof: pd.DataFrame) -> dict:
    y, p = oof.y_true.to_numpy(), oof.y_pred.to_numpy()
    return dict(n=len(oof), MAE=mean_absolute_error(y, p), RMSE=root_mean_squared_error(y, p),
                R2=r2_score(y, p), MaxAE=max_error(y, p))


def breakdown(oof: pd.DataFrame, by: str) -> pd.DataFrame:
    """MAE / RMSE / n per level of ``by`` (e.g. cell_id, health_class, regime)."""
    rows = []
    for level, g in oof.groupby(by, observed=True):
        rows.append(dict(**{by: level}, n=len(g), MAE=mean_absolute_error(g.y_true, g.y_pred),
                         RMSE=root_mean_squared_error(g.y_true, g.y_pred),
                         bias=float((g.y_pred - g.y_true).mean())))
    return pd.DataFrame(rows)


def summary_table(results: list[CVResult]) -> pd.DataFrame:
    return (pd.DataFrame([r.metrics for r in results])
            .loc[:, ["model", "scheme", "n", "MAE", "RMSE", "R2", "MaxAE"]])


def generalization_gap(results: list[CVResult]) -> pd.DataFrame:
    """MAE under each grouped split minus MAE under the random split (how optimistic random CV is)."""
    t = summary_table(results).pivot(index="model", columns="scheme", values="MAE")
    if "Random 5-fold" in t:
        for s in ("LOCO", "LOPO"):
            if s in t:
                t[f"gap {s} - random"] = t[s] - t["Random 5-fold"]
    return t


def paired_comparison(a: CVResult, b: CVResult) -> dict:
    """Compare two models on the same outer folds (per-fold MAE): mean difference, wins, Wilcoxon p.

    A negative ``mean_diff`` means model ``a`` has the lower error.
    """
    assert a.scheme == b.scheme, "compare models on the same split"
    fa, fb = a.folds.set_index("fold").MAE, b.folds.set_index("fold").MAE
    d = (fa - fb).dropna()
    p = wilcoxon(d).pvalue if (d != 0).any() and len(d) >= 5 else np.nan
    return dict(a=a.model, b=b.model, scheme=a.scheme, n_folds=len(d), mean_diff=d.mean(),
                a_wins=int((d < 0).sum()), b_wins=int((d > 0).sum()), wilcoxon_p=p)


# ----------------------------------------------------------------------------- locked hold-out
def holdout_check(df: pd.DataFrame, spec: ModelSpec, numeric=None, categorical=CATEGORICAL,
                  n_jobs: int = -1, seed: int = SEED) -> dict:
    """Secondary check with the notebook-01 split: tune on train+validation cells (inner GroupKFold by
    cell), refit, score once on the 3 locked test cells. Run once per final model, never for selection."""
    if numeric is None:
        numeric, categorical = main_features()
    split = pd.read_csv(PREPROCESSED / "split_assignment.csv")[["cell_id", "idtest", "split"]]
    d = df.merge(split, on=["cell_id", "idtest"], how="left", validate="one_to_one")
    assert d.split.notna().all()
    dev, test = d[d.split != "test"], d[d.split == "test"]
    X_cols = list(numeric) + list(categorical) + ["cell_id"]
    pipe = make_pipeline(spec.estimator, numeric, categorical)
    grid = {f"model__{k}": v for k, v in spec.param_grid.items()}
    if grid:
        pipe = GridSearchCV(pipe, grid, cv=GroupKFold(5), scoring="neg_mean_absolute_error", n_jobs=n_jobs)
        pipe.fit(dev[X_cols], dev[TARGET], groups=dev.cell_id)
    else:
        pipe.fit(dev[X_cols], dev[TARGET])
    p = pipe.predict(test[X_cols])
    return dict(model=spec.name, n_dev_cells=dev.cell_id.nunique(), n_test=len(test),
                test_cells=", ".join(sorted(test.cell_id.unique())),
                MAE=mean_absolute_error(test[TARGET], p), RMSE=root_mean_squared_error(test[TARGET], p))
