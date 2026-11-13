#!/usr/bin/env python
# coding: utf-8

# # 07 · Baseline models for SOH regression
# 
# **Owner:** Samina (Step 2 of the final modelling plan)
# 
# This notebook sets the floor that every stronger model has to beat. It follows the shared **evaluation protocol**
# (`docs/evaluation_protocol.md`, `src/evaluation.py`, walkthrough in notebook 05), so its numbers can be compared
# directly with every other model in the project.
# 
# | Baseline | Why we run it |
# |---|---|
# | Mean SOH of the training cells | The floor every model must beat |
# | One feature: `cum_EFC` or `days_since_start` | Shows how much "time used" alone explains |
# | Multiple linear regression (MLR) on the pruned main inputs | Interpretable reference |
# | Ridge, Lasso, ElasticNet | Regularised linear models for the correlated feature blocks |
# 
# At the end we report the linear coefficients and their signs. Higher resistance and more usage should both
# push SOH down, so negative signs are a quick physics sanity check.
# 
# **What the protocol fixes for us (no choices are made in this notebook)**
# * Data: the 306 labelled rows (25 °C reference tests with SOH), 15 cells, 14 profiles — `ev.load_labelled()`.
# * Inputs: the 42 numeric main inputs plus `schedule` — `ev.main_features()`. The comparison and leaky features
#   (including `chg_CC_V0` and `chg_CV_time_s`, which are measured inside the test that defines SOH) are never inputs.
# * Splits: leave-one-cell-out (**LOCO, headline**), leave-one-profile-out (LOPO), random 5-fold (seed 42, reference only).
# * Pruning, 1–99 % clipping, imputation, scaling and one-hot encoding are fitted inside each training fold
#   (`ev.make_pipeline`). Tuning is nested, with an inner grouped CV of the same kind as the outer split.
# * The hyperparameter grids below are declared **before** any run and are not changed after seeing scores.
# * The locked hold-out (`ev.holdout_check`) is **not** used here; it is reserved for final models only.

# In[1]:


from pathlib import Path
import sys, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import explained_variance_score, mean_absolute_percentage_error, median_absolute_error
from sklearn.model_selection import GridSearchCV, GroupKFold

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import src.evaluation as ev
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 50)

OUT = ROOT / "results" / "baselines"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

df = ev.load_labelled()
NUM, CAT = ev.main_features()
print(f"{len(df)} labelled rows | {df.cell_id.nunique()} cells | {df.profile.nunique()} profiles")
print(f"main inputs: {len(NUM)} numeric + {CAT}")
assert "chg_CC_V0" not in NUM and "chg_CV_time_s" not in NUM


# ## 1. Baselines and their pre-declared grids
# 
# The single-feature models use only their one column (no `schedule`). All other models use the full main inputs.

# In[2]:


# (spec, numeric inputs, categorical inputs) — grids fixed here, before any outer-fold score is seen.
BASELINES = [
    (ev.ModelSpec("Mean SOH (training cells)", DummyRegressor(strategy="mean"), family="baseline"), NUM, CAT),
    (ev.ModelSpec("One feature: cum_EFC", LinearRegression(), family="baseline"), ["cum_EFC"], []),
    (ev.ModelSpec("One feature: days_since_start", LinearRegression(), family="baseline"), ["days_since_start"], []),
    (ev.ModelSpec("MLR (pruned main set)", LinearRegression(), family="linear"), NUM, CAT),
    (ev.ModelSpec("Ridge (tuned)", Ridge(), {"alpha": list(np.logspace(-3, 3, 13))}, family="linear"), NUM, CAT),
    (ev.ModelSpec("Lasso (tuned)", Lasso(max_iter=50_000), {"alpha": list(np.logspace(-3, 1, 9))},
                  family="linear"), NUM, CAT),
    (ev.ModelSpec("ElasticNet (tuned)", ElasticNet(max_iter=50_000),
                  {"alpha": list(np.logspace(-3, 1, 9)), "l1_ratio": [0.2, 0.5, 0.8]}, family="linear"), NUM, CAT),
]
SCHEMES = ["LOCO", "LOPO", "Random 5-fold"]
MODELS = [s.name for s, _, _ in BASELINES]
INPUTS = {s.name: (num, cat) for s, num, cat in BASELINES}
SPECS = {s.name: s for s, _, _ in BASELINES}


# ## 2. Run every baseline on the three splits

# In[3]:


results = {}
for spec, num, cat in BASELINES:
    for scheme in SCHEMES:
        results[(spec.name, scheme)] = ev.run_cv(df, spec, scheme, numeric=num, categorical=cat, verbose=False)
    print("done:", spec.name)
R = list(results.values())


# In[4]:


def extra_metrics(oof):
    y, p = oof.y_true, oof.y_pred
    return dict(MedAE=median_absolute_error(y, p), ExplainedVar=explained_variance_score(y, p),
                MAPE_pct=100 * mean_absolute_percentage_error(y, p))

rows = []
for (name, scheme), r in results.items():
    f = r.folds
    rows.append(r.metrics | extra_metrics(r.oof) | dict(
        n_folds=len(f), fold_MAE_mean=f.MAE.mean(), fold_MAE_std=f.MAE.std(),
        worst_fold_MAE=f.MAE.max(), worst_fold=f.loc[f.MAE.idxmax(), "held_out"]))
metrics = pd.DataFrame(rows)
metrics["model"] = pd.Categorical(metrics.model, MODELS)
metrics["scheme"] = pd.Categorical(metrics.scheme, SCHEMES)
metrics = metrics.sort_values(["scheme", "model"]).reset_index(drop=True)
cols = ["model", "scheme", "n", "n_folds", "MAE", "RMSE", "R2", "MaxAE", "MedAE", "ExplainedVar", "MAPE_pct",
        "fold_MAE_mean", "fold_MAE_std", "worst_fold_MAE", "worst_fold"]
metrics = metrics[cols]

per_fold = pd.concat([r.folds for r in R], ignore_index=True)
oof_all = pd.concat([r.oof for r in R], ignore_index=True)
metrics.to_csv(OUT / "metrics_baselines.csv", index=False)
per_fold.to_csv(OUT / "per_fold_baselines.csv", index=False)
oof_all.to_csv(OUT / "oof_baselines.csv", index=False)

print("MAE (SOH percentage points, pooled out-of-fold) — lower is better. LOCO is the headline.")
display(metrics.pivot_table(index="model", columns="scheme", values="MAE", observed=True).round(2))
print("R² — higher is better")
display(metrics.pivot_table(index="model", columns="scheme", values="R2", observed=True).round(3))


# In[5]:


display(metrics.round(3))


# ### Consistency checks against notebook 05
# 
# * The mean baseline must match `results/evaluation_protocol/reference_mean_baseline.csv` (same rows, same folds).
# * Our tuned Ridge should land close to the protocol's Ridge smoke test (LOCO ≈ 4.76, LOPO ≈ 4.89, random ≈ 2.79).
#   It will not be identical because our alpha grid is wider, but a large gap would mean something differs.

# In[6]:


ref = pd.read_csv(ROOT / "results" / "evaluation_protocol" / "reference_mean_baseline.csv").set_index("scheme")
ours = metrics[metrics.model == "Mean SOH (training cells)"].set_index("scheme")
chk = pd.DataFrame({"reference MAE": ref.MAE, "this notebook": ours.MAE.astype(float).round(3)}).loc[SCHEMES]
display(chk)
assert np.allclose(chk["reference MAE"], chk["this notebook"], atol=1e-3), "mean baseline does not match notebook 05"
print("OK: the mean baseline matches the protocol reference on every split.")

ridge = metrics[metrics.model == "Ridge (tuned)"].set_index("scheme").MAE.astype(float)
display(pd.DataFrame({"protocol smoke test": pd.Series({"LOCO": 4.76, "LOPO": 4.89, "Random 5-fold": 2.79}),
                      "this notebook": ridge.round(2)}).loc[SCHEMES])


# ## 3. Are the differences real? Paired fold-by-fold comparison (LOCO)
# 
# Following protocol rule 7, a lower pooled MAE only counts if it holds across folds. Each row compares two models
# on the same 15 LOCO folds with a Wilcoxon signed-rank test. A negative `mean_diff` means model **a** is better.

# In[7]:


pairs = [(m, "Mean SOH (training cells)") for m in MODELS[1:]] + [
    ("Ridge (tuned)", "MLR (pruned main set)"), ("Lasso (tuned)", "Ridge (tuned)"),
    ("ElasticNet (tuned)", "Ridge (tuned)")]
paired = pd.DataFrame([ev.paired_comparison(results[(a, "LOCO")], results[(b, "LOCO")]) for a, b in pairs])
paired.to_csv(OUT / "paired_comparison_loco.csv", index=False)
display(paired.round(4))

gap = ev.generalization_gap(R).reindex(MODELS)
gap.to_csv(OUT / "generalization_gap.csv")
print("Generalization gap (grouped MAE minus random MAE): how optimistic a random split is")
display(gap.round(2))


# ## 4. Where does the error come from? (LOCO breakdowns)
# 
# Per protocol rule 6: MAE per held-out cell, per health class (<80, 80–90, ≥90) and for interpolation vs
# extrapolation cells. Shown for the mean baseline and the best linear model.

# In[8]:


loco = metrics[metrics.scheme == "LOCO"].set_index("model").MAE.astype(float)
best_linear = loco.loc[["MLR (pruned main set)", "Ridge (tuned)", "Lasso (tuned)", "ElasticNet (tuned)"]].idxmin()
print("best linear model on LOCO:", best_linear)

bd = {}
for by in ["cell_id", "health_class", "regime"]:
    t = pd.concat([ev.breakdown(results[(m, "LOCO")].oof, by).assign(model=m)
                   for m in ["Mean SOH (training cells)", best_linear]])
    bd[by] = t
    t.to_csv(OUT / f"breakdown_loco_by_{by}.csv", index=False)
    display(t.pivot_table(index=by, columns="model", values="MAE", observed=True)
             .join(t.groupby(by, observed=True).n.first()).round(2))


# ## 5. Figure: MAE of each baseline on the three splits

# In[9]:


# Categorical slots 1–3 of the shared palette, in fixed order, one per split.
COLORS = {"LOCO": "#eb6834", "LOPO": "#1baf7a", "Random 5-fold": "#2a78d6"}
LABELS = {"LOCO": "Unseen cell (LOCO, headline)", "LOPO": "Unseen profile (LOPO)",
          "Random 5-fold": "Random 5-fold (reference)"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

piv = metrics.pivot_table(index="model", columns="scheme", values="MAE", observed=True).loc[MODELS[::-1]]
fig, ax = plt.subplots(figsize=(8, 5.4))
h = 0.26
for i, s in enumerate(["Random 5-fold", "LOCO", "LOPO"]):
    ax.barh(np.arange(len(piv)) + (1 - i) * h, piv[s], height=h - 0.04, color=COLORS[s], label=LABELS[s])
for i, name in enumerate(piv.index):                      # one direct label per model: the headline LOCO MAE
    v = piv.loc[name, "LOCO"]
    ax.text(v + 0.15, i, f"{v:.2f}", va="center", fontsize=8, color="#555")
ax.set_yticks(np.arange(len(piv)), piv.index)
ax.set_xlabel("MAE (SOH percentage points, lower is better)")
ax.xaxis.grid(True, color="#e5e5e5"); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=3, fontsize=8.5)
ax.set_title("Baseline models: error on random, unseen-cell and unseen-profile splits", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "baseline_mae_by_split.png", dpi=200); plt.show()


# ## 6. Linear coefficients and the physics sanity check
# 
# To read the signs we fit each linear model once on **all** 306 rows, tuning with a by-cell `GroupKFold`.
# This fit is for interpretation only; every score above comes from held-out folds. Inputs are standardised, so a
# coefficient is the change in SOH (percentage points) for a one-standard-deviation increase in that feature.
# 
# Expected sign:
# * **Usage history** (`days_since_start`, `cum_*`, `rtp_index`): more use → lower SOH → **negative**.
# * **Internal resistance** (`R_*`, excluding the pulse-count columns `R_*_n`): higher resistance → lower SOH → **negative**.
# 
# We also check **sign stability** across the 15 LOCO folds, refitting each fold with the alpha its inner CV chose.

# In[10]:


USAGE = [c for c in NUM if c.startswith("cum_") or c in ("days_since_start", "rtp_index")]
RESIST = [c for c in NUM if c.startswith("R_") and not c.endswith("_n")]
KIND = {**{c: "usage" for c in USAGE}, **{c: "resistance" for c in RESIST}}
LINEAR = ["One feature: cum_EFC", "One feature: days_since_start", "MLR (pruned main set)",
          "Ridge (tuned)", "Lasso (tuned)", "ElasticNet (tuned)"]

def coef_table(pipe):
    names = pipe.named_steps["prep"].get_feature_names_out()
    return pd.Series(pipe.named_steps["model"].coef_, index=names)

X_all, y_all = df, df[ev.TARGET].to_numpy(float)
coef_rows = []
for m in LINEAR:
    spec, (num, cat) = SPECS[m], INPUTS[m]
    pipe = ev.make_pipeline(spec.estimator, num, cat)
    Xc = df[list(num) + list(cat) + ["cell_id"]]
    if spec.param_grid:
        gs = GridSearchCV(pipe, {f"model__{k}": v for k, v in spec.param_grid.items()}, cv=GroupKFold(5),
                          scoring="neg_mean_absolute_error", n_jobs=-1).fit(Xc, y_all, groups=df.cell_id)
        pipe, settings = gs.best_estimator_, {k.removeprefix("model__"): v for k, v in gs.best_params_.items()}
    else:
        pipe, settings = pipe.fit(Xc, y_all), {}
    for f, c in coef_table(pipe).items():
        coef_rows.append(dict(model=m, settings=settings, feature=f, kind=KIND.get(f, "other"), std_coef=c))
coefs = pd.DataFrame(coef_rows)
coefs["expected"] = np.where(coefs.kind == "other", "", "negative")
coefs["sign_ok"] = np.select([coefs.kind == "other", coefs.std_coef < 0, coefs.std_coef == 0],
                             ["", "yes", "zeroed"], "NO")
coefs.to_csv(OUT / "coefficients_full_fit.csv", index=False)

print("Single-feature slopes (cleanest check, nothing else competes with the feature):")
display(coefs[coefs.model.str.startswith("One feature")][["model", "feature", "std_coef", "sign_ok"]].round(3))
print("Usage and resistance coefficients kept by the pruner in the multi-feature models:")
mf = coefs[(coefs.kind != "other") & ~coefs.model.str.startswith("One feature")]
display(mf.pivot_table(index=["kind", "feature"], columns="model", values="std_coef", observed=True).round(3))


# In[11]:


# Sign stability across the 15 LOCO folds (refit each fold with its own chosen settings)
stab_rows = []
folds = ev.outer_folds(df, "LOCO")
for m in ["MLR (pruned main set)", "Ridge (tuned)", "Lasso (tuned)", "ElasticNet (tuned)"]:
    spec, (num, cat) = SPECS[m], INPUTS[m]
    Xc = df[list(num) + list(cat) + ["cell_id"]]
    chosen = results[(m, "LOCO")].folds.set_index("fold").best_params
    for k, (tr, te) in enumerate(folds):
        est = spec.estimator.__class__(**{**spec.estimator.get_params(), **chosen[k]})
        pipe = ev.make_pipeline(est, num, cat).fit(Xc.iloc[tr], y_all[tr])
        for f, c in coef_table(pipe).items():
            if f in KIND:
                stab_rows.append(dict(model=m, kind=KIND[f], feature=f, fold=k, coef=c))
stab = pd.DataFrame(stab_rows)
stability = (stab.groupby(["model", "kind", "feature"])
                 .agg(folds_kept=("coef", "size"), share_negative=("coef", lambda s: (s < 0).mean()),
                      share_zero=("coef", lambda s: (s == 0).mean()), median_coef=("coef", "median"))
                 .reset_index())
stability["folds_kept"] = stability.folds_kept.astype(str) + f"/{len(folds)}"
stability.to_csv(OUT / "coefficient_sign_stability_loco.csv", index=False)
display(stability.round(3))


# In[12]:


# Figure: Ridge standardised coefficients (full-data fit); usage and resistance features highlighted
r = coefs[coefs.model == "Ridge (tuned)"].copy()
r = r.reindex(r.std_coef.abs().sort_values().index)
col = {"usage": "#2a78d6", "resistance": "#eb6834", "other": "#b9b8b0"}
fig, ax = plt.subplots(figsize=(8, 0.3 * len(r) + 1.4))
ax.barh(r.feature, r.std_coef, color=[col[k] for k in r.kind], height=0.7)
ax.axvline(0, color="#555", lw=1)
for i, row in enumerate(r.itertuples()):
    if row.sign_ok == "NO":
        ax.text(row.std_coef, i, "  unexpected sign", va="center", ha="left", fontsize=8, color="#333")
ax.legend(handles=[Patch(color=col[k], label=k) for k in ("usage", "resistance", "other")],
          frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.07 - 0.6 / len(r)), ncol=3)
ax.set_xlabel("Standardised coefficient (SOH points per 1 SD increase)")
ax.xaxis.grid(True, color="#e5e5e5"); ax.set_axisbelow(True)
ax.set_title(f"Ridge coefficients, full-data fit (alpha = {r.settings.iloc[0]['alpha']:.3g})", loc="left",
             fontsize=11)
fig.tight_layout(); fig.savefig(FIG / "ridge_coefficients.png", dpi=200); plt.show()


# ## 7. Summary (generated from the numbers above)

# In[13]:


m = metrics.set_index(["model", "scheme"]).MAE.astype(float)
floor = m[("Mean SOH (training cells)", "LOCO")]
pv = paired.set_index("a")
lines = [f"- Floor: mean SOH of the training cells, LOCO MAE {floor:.2f} SOH points."]
for name in MODELS[1:]:
    p = pv[(pv.index == name) & (pv.b == "Mean SOH (training cells)")].iloc[0]
    lines.append(f"- {name}: LOCO {m[(name, 'LOCO')]:.2f} | LOPO {m[(name, 'LOPO')]:.2f} | random "
                 f"{m[(name, 'Random 5-fold')]:.2f}; vs mean baseline: better in {p.a_wins}/{p.n_folds} folds, "
                 f"Wilcoxon p = {p.wilcoxon_p:.3g}.")
single = coefs[coefs.model.str.startswith("One feature")]
lines.append("- Single-feature slopes: " + "; ".join(f"{r.feature} {r.std_coef:+.2f} (negative as expected: {r.sign_ok})"
                                                       for r in single.itertuples()))
bad = mf[mf.sign_ok == "NO"]
lines.append("- Usage/resistance coefficients with an unexpected positive sign in the full-data fits: "
             + (", ".join(f"{r.feature} ({r.model})" for r in bad.itertuples()) if len(bad) else "none"))
print("\n".join(lines))


# **How to read the sign check.** In the single-feature models a positive slope would be a real red flag. In the
# multi-feature models, a usage or resistance feature can still get a positive coefficient when it is correlated with
# other kept features (for example the `cum_*` columns and `days_since_start`); the model then splits one effect
# between them. Lasso tends to set such duplicates to zero. The sign-stability table shows whether an odd sign is
# consistent across folds or only appears in a few.
# 
# **Result files** (all in `results/baselines/`): `metrics_baselines.csv`, `per_fold_baselines.csv`,
# `oof_baselines.csv`, `paired_comparison_loco.csv`, `generalization_gap.csv`, `breakdown_loco_by_*.csv`,
# `coefficients_full_fit.csv`, `coefficient_sign_stability_loco.csv`, and two figures in `figures/`.
