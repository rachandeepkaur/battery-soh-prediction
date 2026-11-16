"""Health-class classification under the shared evaluation protocol (Step 5, Shraddhaben).

Two ways to get a health class are compared on the same 306 rows and the same folds:
  A. Direct classifier  - logistic regression, decision tree, random forest (plus a majority-class floor).
  B. Regress, then threshold - take the out-of-fold SOH predictions of the regression models and cut them at 80 and 90.

Classes (project definition): EOL risk (< 80), Degraded (80-90), Healthy (>= 90).
Everything that learns from data (pruning, clipping, imputation, scaling, tuning) is fitted inside each training fold,
exactly as in ``src/evaluation.py``. Hyperparameter grids are declared in ``SPECS`` before any run.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, make_scorer,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from . import evaluation as ev

warnings.filterwarnings("ignore")

ROOT = ev.ROOT
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "classification"
LABELS = list(ev.HEALTH_LABELS)            # ['EOL risk (<80)', 'Degraded (80-90)', 'Healthy (>=90)']
EOL, DEG, HEALTHY = LABELS


def to_class(soh) -> np.ndarray:
    """SOH (%) -> health class name, using the project cut-offs 80 and 90 (right-open bins)."""
    return pd.cut(pd.Series(np.asarray(soh, float)), ev.HEALTH_BINS, right=False,
                  labels=LABELS).astype(str).to_numpy()


class SOHClassifier(BaseEstimator, ClassifierMixin):
    """Fold-safe classifier. ``fit`` receives the continuous SOH so the shared pruner ranks features with it
    (as in the regression notebooks); the classifier itself is trained on the health class."""

    def __init__(self, prep=None, clf=None):
        self.prep = prep
        self.clf = clf

    def fit(self, X, y):
        self.prep_ = clone(self.prep).fit(X, y)
        self.clf_ = clone(self.clf).fit(self.prep_.transform(X), to_class(y))
        self.classes_ = self.clf_.classes_
        return self

    def predict(self, X):
        return self.clf_.predict(self.prep_.transform(X))


def _balanced_accuracy(y_soh, y_pred):
    return balanced_accuracy_score(to_class(y_soh), y_pred)


SCORER = make_scorer(_balanced_accuracy)       # inner-CV tuning objective: balanced accuracy

# Pre-declared models and grids (not changed after seeing outer-fold scores).
SPECS = {
    "Majority class (floor)": (DummyClassifier(strategy="most_frequent"), {}),
    "Logistic regression": (LogisticRegression(class_weight="balanced", max_iter=5000), {"C": [0.01, 0.1, 1, 10]}),
    "Decision tree": (DecisionTreeClassifier(class_weight="balanced", random_state=42),
                      {"max_depth": [2, 3, 4], "min_samples_leaf": [5, 10]}),
    "Random forest": (RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                             random_state=42, n_jobs=1),
                      {"max_depth": [3, 5], "min_samples_leaf": [3, 5]}),
}


def run_direct(df: pd.DataFrame, name: str, scheme: str, numeric, categorical, n_jobs: int = -1) -> pd.DataFrame:
    """Nested CV for one direct classifier. Returns one row per labelled row (out-of-fold prediction)."""
    est, grid = SPECS[name]
    X = df[list(numeric) + list(categorical) + ["cell_id"]]
    y = df[ev.TARGET].to_numpy(float)
    gcol = ev.SCHEMES[scheme]
    pred = np.empty(len(df), dtype=object)
    fold_of = np.full(len(df), -1)
    regime = np.empty(len(df), dtype=object)
    for k, (tr, te) in enumerate(ev.outer_folds(df, scheme)):
        wrap = SOHClassifier(prep=ev.make_preprocessor(numeric, categorical), clf=est)
        if grid:
            search = GridSearchCV(wrap, {f"clf__{a}": b for a, b in grid.items()}, cv=ev.inner_cv(scheme),
                                  scoring=SCORER, n_jobs=n_jobs, refit=True)
            search.fit(X.iloc[tr], y[tr], groups=None if gcol is None else df[gcol].to_numpy()[tr])
            model = search.best_estimator_
        else:
            model = wrap.fit(X.iloc[tr], y[tr])
        pred[te] = model.predict(X.iloc[te])
        fold_of[te] = k
        regime[te] = ev.regime_flags(df, tr, te)
    return df[ev.ID_COLS].assign(model=name, scheme=scheme, approach="Direct classifier", fold=fold_of,
                                 regime=regime, y_soh=y, y_true=to_class(y), y_pred=pred)


# Out-of-fold SOH predictions already produced by the teammates' notebooks (same folds, same inputs).
REGRESSION_OOF = [
    ("baselines/oof_baselines.csv", {"Ridge (tuned)": "Ridge", "Lasso (tuned)": "Lasso",
                                      "ElasticNet (tuned)": "ElasticNet",
                                      "Mean SOH (training cells)": "Mean SOH (training mean)"}),
    ("nonlinear_svr_gpr/oof_predictions.csv", {"SVR (RBF)": "SVR (RBF)", "Gaussian process": "Gaussian process"}),
    ("modelling_random_forest_hgb/oof_rf_hgb.csv", {"Random forest": "Random forest (regressor)",
                                                     "HistGradientBoosting": "HistGradientBoosting"}),
]


def load_regression_thresholded(results: Path = RESULTS) -> pd.DataFrame:
    """Regression OOF predictions cut at 80 / 90 into health classes."""
    parts = []
    for rel, names in REGRESSION_OOF:
        o = pd.read_csv(results / rel)
        o = o[o.model.isin(names)].copy()
        o["model"] = o.model.map(names)
        parts.append(o)
    reg = pd.concat(parts, ignore_index=True)
    reg["approach"] = "Regression, then threshold"
    reg["y_soh"] = reg["y_true"]
    reg["y_true"] = to_class(reg["y_soh"])
    reg["y_pred"] = to_class(reg["y_pred"])
    return reg


def class_metrics(g: pd.DataFrame) -> dict:
    """Pooled out-of-fold classification metrics for one model/scheme."""
    yt, yp = g["y_true"].to_numpy(), g["y_pred"].to_numpy()
    p, r, f, _ = precision_recall_fscore_support(yt, yp, labels=LABELS, zero_division=0)
    return dict(n=len(g), accuracy=accuracy_score(yt, yp), balanced_accuracy=balanced_accuracy_score(yt, yp),
                macro_F1=f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0),
                EOL_precision=p[0], EOL_recall=r[0], EOL_F1=f[0],
                Degraded_recall=r[1], Healthy_recall=r[2])


def binary_metrics(g: pd.DataFrame, cut: int) -> dict:
    """Two-class view of the same predictions: 'below cut' (positive) vs 'at or above cut'."""
    def below(labels):
        labels = np.asarray(labels)
        return np.isin(labels, [EOL] if cut == 80 else [EOL, DEG])
    yt, yp = below(g["y_true"]), below(g["y_pred"])
    tp, fp, fn, tn = (yt & yp).sum(), (~yt & yp).sum(), (yt & ~yp).sum(), (~yt & ~yp).sum()
    rec = tp / (tp + fn) if tp + fn else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(cut=cut, positives=int(tp + fn), recall=rec, precision=prec, specificity=spec, F1=f1,
                balanced_accuracy=np.nanmean([rec, spec]))


def metric_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = [dict(scheme=s, approach=a, model=m, **class_metrics(g))
            for (s, a, m), g in pred.groupby(["scheme", "approach", "model"])]
    return pd.DataFrame(rows)


def binary_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (s, a, m), g in pred.groupby(["scheme", "approach", "model"]):
        for cut in (80, 90):
            rows.append(dict(scheme=s, approach=a, model=m, **binary_metrics(g, cut)))
    return pd.DataFrame(rows)
