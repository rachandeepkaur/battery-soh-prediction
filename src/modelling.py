"""Modelling utilities: feature groups, fold-safe redundancy pruning and the evaluation grid.

Everything that looks at the target (choosing which member of a redundancy block to keep) or at the
feature distribution (imputation, scaling) lives inside a sklearn Pipeline, so it is re-fitted on the
training part of every CV fold and never sees the held-out cell / profile.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Same grouping as notebooks/02_eda_feature_discovery.ipynb, Section 4.
GROUPS = {
    "LEAKY (exclude)": ["cap_1C_Ah", "dis1C_energy_Wh", "cap_1C_spread_Ah", "cap_ref_Ah"],
    "Partial-capacity HI": ["dis1C_Q_3.8_3.5V", "dis1C_t_to_3.6V_s", "dis_C3_CC_Ah", "chg_CC_Ah",
                            "chg_CC_time_s", "chg_CV_Ah", "ica_area_3.4_3.8V"],
    "Shape / resistance HI": ["R_1p13C_10s_mean", "R_1p13C_inst_mean", "R_1p35C_10s_mean", "R_0p18C_10s_mean",
                              "R_1p13C_10s_soc55", "R_1p13C_10s_lowSOC", "dis1C_IRdrop_ohm", "dis1C_dT",
                              "dis1C_meanV", "relax_dV_after_dis", "ica_p1_height", "ica_p1_V", "ica_p2_height",
                              "ica_p2_V", "ica_p3_height", "ica_p3_V", "chg_CC_fraction", "chg_CV_time_s",
                              "chg_CC_V0"],
    "Usage history": ["days_since_start", "cum_EFC", "cum_dis_Ah", "cum_lp_days", "cum_lp_T_mean", "rtp_index"],
    "Previous load period": ["prev_lp_T_mean", "prev_lp_T_max", "prev_lp_V_mean", "prev_lp_V_max", "prev_lp_V_min",
                             "prev_lp_I_dis_mean", "prev_lp_I_chg_mean", "prev_lp_I_dis_peak", "prev_lp_rest_frac",
                             "prev_lp_dis_Ah"],
    "Profile design": ["lp_T_set", "lp_CC", "lp_ADC", "lp_PDC", "lp_F", "lp_SoC", "lp_dSoC"],
}
LEAKY = GROUPS["LEAKY (exclude)"]
# chg_CC_V0 is the rested voltage right after a full 1C discharge to 2.5 V, i.e. a capacity proxy
# (see the caveat in notebook 02, Section 7b). It is removed together with the partial-capacity HIs.
PARTIAL_CAPACITY = GROUPS["Partial-capacity HI"] + ["chg_CC_V0"]


def within_cell_spearman(X: pd.DataFrame, y: np.ndarray, cells: np.ndarray, min_rows: int = 4) -> pd.DataFrame:
    """Median within-cell Spearman rho of each column with y, and the share of cells agreeing in sign."""
    y = pd.Series(np.asarray(y, float), index=X.index)
    per_cell = {}
    for c, idx in pd.Series(X.index, index=X.index).groupby(np.asarray(cells)).groups.items():
        if len(idx) < min_rows or y[idx].nunique() < 2:
            continue
        sub = X.loc[idx]
        r = sub.rank().corrwith(y[idx].rank())
        r[sub.nunique() < 2] = np.nan            # constant inside the cell -> undefined
        per_cell[c] = r
    rho = pd.DataFrame(per_cell)                 # features x cells
    med = rho.median(axis=1)
    agree = rho.apply(lambda s: np.mean(np.sign(s.dropna()) == np.sign(med[s.name])) if s.notna().any() else np.nan,
                      axis=1)
    return pd.DataFrame({"within_rho": med, "sign_consistency": agree, "n_cells": rho.notna().sum(axis=1)})


def redundancy_blocks(X: pd.DataFrame, threshold: float = 0.95) -> pd.Series:
    """Cluster features so that every pair inside a block has |Spearman rho| >= threshold.

    Complete linkage on distance 1-|rho| cut at 1-threshold guarantees the "every pair" property.
    Returns a block id per feature.
    """
    if X.shape[1] < 2:
        return pd.Series(1, index=X.columns, name="block")
    corr = X.corr(method="spearman").abs().fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(corr, 1.0)
    dist = squareform(1.0 - corr, checks=False)
    Z = linkage(dist, method="complete")
    return pd.Series(fcluster(Z, t=1.0 - threshold, criterion="distance"), index=X.columns, name="block")


def rank_features(X: pd.DataFrame, y, cells) -> pd.DataFrame:
    """Score = |median within-cell rho| x sign consistency. Features that are constant inside every cell
    (profile design) have no within-cell rho; they fall back to |pooled rho| x 0.5 so that a feature
    that actually tracks a cell's ageing wins a tie against one that only separates cells."""
    w = within_cell_spearman(X, y, cells)
    pooled = X.rank().corrwith(pd.Series(np.asarray(y, float), index=X.index).rank()).abs()
    score = (w.within_rho.abs() * w.sign_consistency).fillna(0.5 * pooled).fillna(0.0)
    return w.assign(pooled_rho=pooled, score=score)


class RedundancyPruner(BaseEstimator, TransformerMixin):
    """Keep one representative per |rho| >= threshold block, chosen by within-cell rho on the training fold.

    Expects a DataFrame holding ``features`` plus the ``group_col`` column (cell id); outputs only the
    selected feature columns.
    """

    def __init__(self, features, group_col="cell_id", threshold=0.95):
        self.features = features
        self.group_col = group_col
        self.threshold = threshold

    def fit(self, X, y):
        Xf = X[list(self.features)].astype(float)
        Xf = Xf.loc[:, Xf.notna().any() & (Xf.nunique() > 1)]       # unusable in this fold
        self.ranking_ = rank_features(Xf, y, X[self.group_col].values)
        self.blocks_ = redundancy_blocks(Xf, self.threshold)
        best = (self.ranking_.join(self.blocks_)
                .sort_values(["score", "pooled_rho"], ascending=False)
                .groupby("block", sort=False).head(1))
        self.selected_ = [c for c in Xf.columns if c in best.index]   # keep original column order
        return self

    def transform(self, X):
        return X[self.selected_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_, dtype=object)


# ----------------------------------------------------------------------------- models
def make_models(random_state: int = 0) -> dict:
    scaled_y = lambda est: TransformedTargetRegressor(est, transformer=StandardScaler())
    return {
        "Mean baseline": (DummyRegressor(), False),
        "Ridge": (Ridge(), True),
        "SVR (RBF)": (scaled_y(SVR(C=10.0, epsilon=0.05, gamma="scale")), True),
        "Random forest": (RandomForestRegressor(400, min_samples_leaf=2, max_features=0.5,
                                                random_state=random_state, n_jobs=-1), False),
        "Gradient boosting": (HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                            min_samples_leaf=10, random_state=random_state), False),
    }


def build_pipeline(features, model, needs_scaling: bool, threshold: float = 0.95):
    steps = [RedundancyPruner(features, threshold=threshold), SimpleImputer(strategy="median")]
    if needs_scaling:
        steps.append(StandardScaler())
    steps.append(clone(model))
    return make_pipeline(*steps)


def splitters(df: pd.DataFrame, n_random: int = 5, random_state: int = 0) -> dict:
    """name -> (cv object, groups array or None)."""
    return {
        "Random split": (KFold(n_random, shuffle=True, random_state=random_state), None),
        "Leave-cell-out": (LeaveOneGroupOut(), df["cell_id"].values),
        "Leave-profile-out": (LeaveOneGroupOut(), df["profile"].values),
    }


def grouped_ridge_alpha(X: pd.DataFrame, y, feats, threshold: float = 0.95,
                        alphas=np.logspace(-3, 3, 13), n_inner: int = 5) -> float:
    groups = X["cell_id"].values
    inner = GroupKFold(n_splits=min(n_inner, len(np.unique(groups))))
    scores = {}
    for a in alphas:
        errs = []
        for tr, te in inner.split(X, y, groups):
            pipe = build_pipeline(feats, Ridge(alpha=a), True, threshold).fit(X.iloc[tr], y[tr])
            errs.append(mean_absolute_error(y[te], pipe.predict(X.iloc[te])))
        scores[a] = np.mean(errs)
    return min(scores, key=scores.get)


def evaluate(df: pd.DataFrame, feature_sets: dict, target: str = "SOH", threshold: float = 0.95):
    """Run every model x feature set x split. Returns (metrics, out-of-fold predictions, selections)."""
    y = df[target].values
    metrics, preds, picks = [], [], []
    for fs_name, feats in feature_sets.items():
        X = df[list(feats) + ["cell_id"]]
        for split_name, (cv, groups) in splitters(df).items():
            folds = list(cv.split(X, y, groups))
            for m_name, (model, scale) in make_models().items():
                oof = np.full(len(y), np.nan)
                fold_id = np.full(len(y), -1)
                for k, (tr, te) in enumerate(folds):
                    est = model
                    if m_name == "Ridge":
                        est = Ridge(alpha=grouped_ridge_alpha(X.iloc[tr], y[tr], feats, threshold))
                    pipe = build_pipeline(feats, est, scale, threshold).fit(X.iloc[tr], y[tr])
                    oof[te] = pipe.predict(X.iloc[te])
                    fold_id[te] = k
                    if m_name == "Ridge":     # selection is model-independent; record it once
                        picks.append(dict(feature_set=fs_name, split=split_name, fold=k,
                                          selected=pipe[0].selected_))
                metrics.append(dict(feature_set=fs_name, split=split_name, model=m_name,
                                    n_folds=len(folds), MAE=mean_absolute_error(y, oof),
                                    RMSE=root_mean_squared_error(y, oof), R2=r2_score(y, oof)))
                preds.append(pd.DataFrame({"feature_set": fs_name, "split": split_name, "model": m_name,
                                           "cell_id": df["cell_id"].values, "profile": df["profile"].values,
                                           "idtest": df["idtest"].values, "fold": fold_id,
                                           "y_true": y, "y_pred": oof}))
    return pd.DataFrame(metrics), pd.concat(preds, ignore_index=True), pd.DataFrame(picks)
