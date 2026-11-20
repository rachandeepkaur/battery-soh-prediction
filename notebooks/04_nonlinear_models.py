#!/usr/bin/env python
# coding: utf-8

# # 04 · Non-linear models: SVR and Gaussian process
# Tests two non-linear models for SOH against Ridge (linear reference)

# In[1]:


import sys, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from src import evaluation as ev

df = ev.load_labelled()
numeric, categorical = ev.main_features()
print(df.shape, len(numeric), categorical)


# ## 1. Models and settings tried
# - **SVR (RBF):** fits a smooth curve and ignores small errors inside a margin (epsilon). Tried C ∈ {1, 10, 30, 100, 300}, gamma ∈ {scale, 0.003, 0.01, 0.03, 0.1}, epsilon ∈ {0.1, 0.5, 1.0}.
# - **Gaussian process (GP):** also fits a smooth curve, and gives an uncertainty for each prediction. Tried 3 kernels: RBF, Matérn 1.5, Matérn 2.5 (each + white noise).
# - **Ridge:** linear reference, alpha from 0.001 to 1000 (same grid as the baselines notebook).

# In[2]:


from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, Matern, WhiteKernel

# Ridge = linear reference (same grid as the baselines notebook)
ridge = ev.ModelSpec("Ridge", Ridge(), {"alpha": list(np.logspace(-3, 3, 13))}, family="linear")

# SVR with RBF kernel: tune C, gamma, epsilon
svr = ev.ModelSpec(
    "SVR (RBF)", SVR(kernel="rbf"),
    {"C": [1, 10, 30, 100, 300],
     "gamma": ["scale", 0.003, 0.01, 0.03, 0.1],
     "epsilon": [0.1, 0.5, 1.0]},
    family="non-linear")

# Gaussian process: the grid only picks the kernel type
def k_rbf():  return C(1.0, (1e-2, 1e3)) * RBF(5.0, (1e-2, 1e3)) + WhiteKernel(1.0, (1e-5, 1e2))
def k_m15():  return C(1.0, (1e-2, 1e3)) * Matern(5.0, (1e-2, 1e3), nu=1.5) + WhiteKernel(1.0, (1e-5, 1e2))
def k_m25():  return C(1.0, (1e-2, 1e3)) * Matern(5.0, (1e-2, 1e3), nu=2.5) + WhiteKernel(1.0, (1e-5, 1e2))

gpr = ev.ModelSpec(
    "Gaussian process",
    GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=2, random_state=0),
    {"kernel": [k_rbf(), k_m15(), k_m25()]},
    family="non-linear")

print("Models ready:", ridge.name, "|", svr.name, "|", gpr.name)


# In[3]:


get_ipython().run_cell_magic('time', '', 'results = []\nfor spec in [ridge, svr, gpr]:\n    results.append(ev.run_cv(df, spec, "LOCO"))\n\nR = {(r.model, r.scheme): r for r in results}\ndisplay(ev.summary_table(results).round(2))\n')


# ## 3. Error per cell, chosen settings, worst predictions

# In[4]:


models = ["Ridge", "SVR (RBF)", "Gaussian process"]

# 1) MAE for each held-out cell, models side by side
folds = pd.concat([R[(m, "LOCO")].folds for m in models])
display(folds.pivot(index="held_out", columns="model", values="MAE")[models].round(2))

# 2) Settings each fold picked
for m in ["SVR (RBF)", "Gaussian process"]:
    print("\n", m)
    display(R[(m, "LOCO")].folds[["held_out", "best_params"]])

# 3) The 10 worst predictions
oof = pd.concat([R[(m, "LOCO")].oof for m in models])
oof["abs_err"] = (oof.y_pred - oof.y_true).abs()
display(oof.sort_values("abs_err", ascending=False)
           [["model", "cell_id", "rtp_index", "y_true", "y_pred", "abs_err", "regime"]].head(10).round(2))


# ## 4. Paired comparison and error breakdown
# Per-fold MAE on the same 15 folds. Negative mean_diff = model "a" is better; the Wilcoxon p-value says whether the difference is consistent across folds.

# In[5]:


# 1) Paired comparison on the same 15 folds
display(pd.DataFrame([
    ev.paired_comparison(R[("SVR (RBF)", "LOCO")],        R[("Ridge", "LOCO")]),
    ev.paired_comparison(R[("Gaussian process", "LOCO")], R[("Ridge", "LOCO")]),
    ev.paired_comparison(R[("Gaussian process", "LOCO")], R[("SVR (RBF)", "LOCO")]),
]).round(3))

# 2) Error by health class and by interpolation vs extrapolation
for m in models:
    oof_m = R[(m, "LOCO")].oof
    print(f"\n===== {m} =====")
    display(ev.breakdown(oof_m, "health_class").round(2))
    display(ev.breakdown(oof_m, "regime").round(2))


# ## 5. Gaussian process uncertainty
# The GP gives a standard deviation for each prediction. Coverage = how often the true SOH falls inside the 95% band (ideal ≈ 95%). Correlation = does the GP know when it is wrong?

# In[6]:


get_ipython().run_cell_magic('time', '', 'gp_res = R[("Gaussian process", "LOCO")]\nX = df[numeric + categorical + ["cell_id"]]\ny = df["SOH"].to_numpy(float)\nmu, sd = np.zeros(len(df)), np.zeros(len(df))\n\nfor k, (tr, te) in enumerate(ev.outer_folds(df, "LOCO")):\n    kern = gp_res.folds.loc[k, "best_params"]["kernel"]\n    pipe = ev.make_pipeline(clone(gpr.estimator).set_params(kernel=kern), numeric, categorical)\n    pipe.fit(X.iloc[tr], y[tr])\n    Z = pipe.named_steps["prep"].transform(X.iloc[te])\n    mu[te], sd[te] = pipe.named_steps["model"].predict(Z, return_std=True)\n\nprint("MAE check (should match GP LOCO above):", np.abs(mu - y).mean().round(2))\n\nlo, hi = mu - 1.96 * sd, mu + 1.96 * sd\nunc = df[["cell_id", "profile", "rtp_index", "health_class"]].assign(\n    y_true=y, y_pred=mu, std=sd, abs_err=np.abs(mu - y), inside_95=(y >= lo) & (y <= hi))\nunc["regime"] = gp_res.oof["regime"].values\n\nprint(f"95% interval coverage: {unc.inside_95.mean():.1%}  (ideal ≈ 95%)")\nprint(f"Mean interval width:   {(hi - lo).mean():.2f} SOH %")\nprint(f"Correlation between std and error: {np.corrcoef(sd, unc.abs_err)[0,1]:.2f}")\ndisplay(unc.groupby("regime").agg(coverage=("inside_95", "mean"), mean_std=("std", "mean"), MAE=("abs_err", "mean")).round(2))\ndisplay(unc.groupby("cell_id").agg(coverage=("inside_95", "mean"), mean_std=("std", "mean"), MAE=("abs_err", "mean")).sort_values("MAE", ascending=False).round(2))\n')


# In[7]:


fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Left: predicted vs true
ok = unc.inside_95.to_numpy()
ax[0].errorbar(y[ok], mu[ok], yerr=1.96 * sd[ok], fmt="o", ms=3, alpha=.4, label="inside 95% band")
ax[0].errorbar(y[~ok], mu[~ok], yerr=1.96 * sd[~ok], fmt="o", ms=3, color="red", alpha=.7, label="outside 95% band")
ax[0].plot([30, 105], [30, 105], "k--", lw=1)
ax[0].set(xlabel="True SOH (%)", ylabel="GP predicted SOH (%)", title="Gaussian process, LOCO, 95% bands")
ax[0].legend()

# Right: one held-out cell over time
cell = "2_092"
c = unc[unc.cell_id == cell].sort_values("rtp_index")
ax[1].plot(c.rtp_index, c.y_true, "ko-", label="true SOH")
ax[1].plot(c.rtp_index, c.y_pred, "b.-", label="GP prediction")
ax[1].fill_between(c.rtp_index, c.y_pred - 1.96 * c["std"], c.y_pred + 1.96 * c["std"], alpha=.25, label="95% band")
ax[1].axhline(80, color="red", ls=":", lw=1, label="80% EOL line")
ax[1].set(xlabel="Check-up number", ylabel="SOH (%)", title=f"Held-out cell {cell}")
ax[1].legend()

plt.tight_layout()
plt.savefig("gp_uncertainty.png", dpi=150)
plt.show()


# ## 6. Other splits: LOPO and random 5-fold
# LOPO holds out a whole usage profile. The random split is shown only to measure how optimistic it is (generalization gap).

# In[8]:


get_ipython().run_cell_magic('time', '', 'for scheme in ["LOPO", "Random 5-fold"]:\n    for spec in [ridge, svr, gpr]:\n        results.append(ev.run_cv(df, spec, scheme))\n\nR = {(r.model, r.scheme): r for r in results}\ndisplay(ev.summary_table(results).round(2))\ndisplay(ev.generalization_gap(results).round(2))\n')


# ## 7. LOPO comparison and locked hold-out check (final model only)
# The locked 3-cell check (2_009, 2_092, 2_098) is run only on the final model from this notebook, SVR, as the protocol requires. 

# In[9]:


get_ipython().run_cell_magic('time', '', '# 1) Is SVR vs Ridge consistent on LOPO too?\ndisplay(pd.DataFrame([\n    ev.paired_comparison(R[("SVR (RBF)", "LOPO")],        R[("Ridge", "LOPO")]),\n    ev.paired_comparison(R[("Gaussian process", "LOPO")], R[("Ridge", "LOPO")]),\n]).round(3))\n\n# 2) Locked hold-out: final model only (SVR)\nholdout = pd.DataFrame([ev.holdout_check(df, svr)])\ndisplay(holdout.round(2))\n')


# ## 8. Post-hoc fix: Gaussian process with a linear kernel part
# 
# **This section was added after seeing the results above, so it is labelled post-hoc.** The original GP stays in every table for comparison.
# 
# **What went wrong.** All three original kernels (RBF, Matérn 1.5, Matérn 2.5) only measure how *similar* a row is to the training rows.
# On a battery unlike any training battery, the GP copies the nearest one. Cell 2_092 (true SOH ≈ 90) was predicted at ≈ 42, most likely because
# its nearest neighbour in feature space is 2_095, the other 45 °C cell, which fell to 33%. The same habit makes the GP look best on a random split (1.30), where
# the nearest neighbour is the battery's own previous check-up.
# 
# **The fix.** Add a linear part to the kernel (`ConstantKernel * DotProduct`). The GP then becomes a Bayesian Ridge regression plus a
# smooth non-linear correction, so on unfamiliar batteries it falls back to the linear trend instead of a neighbour.
# 
# **Rules kept.** The kernel grid below was written before running and is run once. It is not tuned further, and it is not run on the locked hold-out.
# * Linear + RBF + white noise
# * Linear + Matérn 2.5 + white noise

# In[10]:


from sklearn.gaussian_process.kernels import DotProduct

def k_lin_rbf():
    return (C(1.0, (1e-3, 1e3)) * DotProduct(1.0, (1e-3, 1e3))
            + C(1.0, (1e-2, 1e3)) * RBF(5.0, (1e-1, 1e3)) + WhiteKernel(1.0, (1e-5, 1e2)))
def k_lin_m25():
    return (C(1.0, (1e-3, 1e3)) * DotProduct(1.0, (1e-3, 1e3))
            + C(1.0, (1e-2, 1e3)) * Matern(5.0, (1e-1, 1e3), nu=2.5) + WhiteKernel(1.0, (1e-5, 1e2)))

gpr_lin = ev.ModelSpec(
    "GP linear+RBF (post-hoc)",
    GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=2, random_state=0),
    {"kernel": [k_lin_rbf(), k_lin_m25()]},
    family="non-linear")


# In[11]:


get_ipython().run_cell_magic('time', '', 'for scheme in ["LOCO", "LOPO", "Random 5-fold"]:\n    results.append(ev.run_cv(df, gpr_lin, scheme))\nR = {(r.model, r.scheme): r for r in results}\ndisplay(ev.summary_table(results).round(2))\ndisplay(ev.generalization_gap(results).round(2))\n')


# In[12]:


# Same checks as for the original GP: paired tests, breakdowns, per-cell error, chosen kernels
NEW = gpr_lin.name
display(pd.DataFrame([ev.paired_comparison(R[(NEW, s)], R[(b, s)])
                      for s in ["LOCO", "LOPO"] for b in ["Gaussian process", "Ridge", "SVR (RBF)"]]).round(3))
for col in ["health_class", "regime"]:
    display(ev.breakdown(R[(NEW, "LOCO")].oof, col).round(2))
models_fix = ["Ridge", "Gaussian process", NEW]
display(pd.concat([R[(m, "LOCO")].folds for m in models_fix])
          .pivot(index="held_out", columns="model", values="MAE")[models_fix].round(2))
print(R[(NEW, "LOCO")].folds.best_params.astype(str).str.contains("RBF").map({True: "linear+RBF", False: "linear+Matern 2.5"}).value_counts())


# In[13]:


get_ipython().run_cell_magic('time', '', '# Uncertainty of the fixed GP (LOCO): refit each fold with its chosen kernel and read the std\nfix_res = R[(NEW, "LOCO")]\nmu2, sd2 = np.zeros(len(df)), np.zeros(len(df))\nfor k, (tr, te) in enumerate(ev.outer_folds(df, "LOCO")):\n    kern = fix_res.folds.loc[k, "best_params"]["kernel"]\n    pipe = ev.make_pipeline(clone(gpr_lin.estimator).set_params(kernel=kern), numeric, categorical)\n    pipe.fit(X.iloc[tr], y[tr])\n    Z = pipe.named_steps["prep"].transform(X.iloc[te])\n    mu2[te], sd2[te] = pipe.named_steps["model"].predict(Z, return_std=True)\nprint("MAE check (should match LOCO above):", np.abs(mu2 - y).mean().round(2))\n\nunc2 = df[["cell_id", "profile", "rtp_index", "health_class"]].assign(\n    y_true=y, y_pred=mu2, std=sd2, abs_err=np.abs(mu2 - y),\n    inside_95=(y >= mu2 - 1.96 * sd2) & (y <= mu2 + 1.96 * sd2), regime=fix_res.oof["regime"].values)\nprint(f"95% interval coverage: {unc2.inside_95.mean():.1%}  (original GP {unc.inside_95.mean():.1%}, ideal ≈ 95%)")\nprint(f"Mean interval width:   {(3.92 * sd2).mean():.2f} SOH %  (original GP {(hi - lo).mean():.2f})")\nprint(f"Correlation between std and error: {np.corrcoef(sd2, unc2.abs_err)[0,1]:.2f}  (original GP {np.corrcoef(sd, unc.abs_err)[0,1]:.2f})")\ndisplay(unc2.groupby("regime").agg(coverage=("inside_95", "mean"), mean_std=("std", "mean"), MAE=("abs_err", "mean")).round(2))\n')


# **The fix did not work.** The linear part barely changed the GP.
# * **Error:** 7.63 on unseen batteries and 7.69 on unseen profiles, against 7.72 / 8.13 before. Not significantly different
#   from the original GP (7 of 15 battery wins, p = 0.80), and still far behind Ridge (4.76) and SVR (4.81).
# * **Cell 2_092 is still predicted at ≈ 42 instead of ≈ 90** (MAE 30.1 vs 30.2 before). The GP still copies a neighbour there.
# * **Uncertainty is unchanged:** 95% band coverage 88% (71% on extrapolation rows), std–error correlation 0.61.
# * **Random split still flatters it** (1.29), so the gap to the honest split stays at about 6.3.
# 
# **Why the linear part didn't take over.** The kernel settings are chosen by maximising the likelihood of the training rows.
# Rows from the same battery are strongly correlated, so a short-range similarity kernel explains the training data very well,
# and the optimiser keeps giving it most of the weight. The linear part is there, but it is too small to matter on new batteries.
# Making it win would need a different setup (for example a fixed, longer length scale, or choosing kernel settings by
# leave-one-cell-out likelihood). That would be a second round of post-hoc tuning, so we stop here.
# 
# **Conclusion:** report the GP as it is. Its predictions are not competitive on unseen batteries, and the kernel fix confirms
# that the problem is the similarity-based approach, not one bad kernel choice. Its uncertainty is still useful as a warning sign.

# In[14]:


import os
out = ROOT / "results" / "nonlinear_svr_gpr"
os.makedirs(out, exist_ok=True)

ev.summary_table(results).to_csv(f"{out}/metrics_all_splits.csv", index=False)
ev.generalization_gap(results).to_csv(f"{out}/generalization_gap.csv")
pd.concat([r.folds for r in results]).to_csv(f"{out}/per_fold_metrics.csv", index=False)
pd.concat([r.oof for r in results]).to_csv(f"{out}/oof_predictions.csv", index=False)
pd.DataFrame([ev.paired_comparison(R[(a, s)], R[(b, s)])
              for s in ["LOCO", "LOPO"]
              for a, b in [("SVR (RBF)", "Ridge"), ("Gaussian process", "Ridge"),
                           ("Gaussian process", "SVR (RBF)"),
                           (gpr_lin.name, "Gaussian process"), (gpr_lin.name, "Ridge")]]
            ).to_csv(f"{out}/paired_comparison.csv", index=False)
holdout.to_csv(f"{out}/holdout_check.csv", index=False)
unc.to_csv(f"{out}/gpr_uncertainty_loco.csv", index=False)
unc2.to_csv(f"{out}/gpr_linear_uncertainty_loco.csv", index=False)
os.replace("gp_uncertainty.png", f"{out}/gp_uncertainty.png")

print(sorted(os.listdir(out)))


# ## 9. Summary
# | Model | LOCO | LOPO | Random 5-fold |
# |---|---|---|---|
# | Ridge | 4.76 | 4.89 | 2.79 |
# | SVR (RBF) | 4.81 | 5.87 | 1.24 |
# | Gaussian process | 7.72 | 8.13 | 1.30 |
# | GP linear+RBF (post-hoc) | 7.63 | 7.69 | 1.29 |
# 
# - **SVR does not beat Ridge.** On LOCO they are about the same (p = 0.85). On LOPO SVR is worse (10 of 14 folds, p = 0.049).
# - **The Gaussian process is the weakest model** (worse than Ridge and SVR, p < 0.05), but its uncertainty is useful: a bigger std goes with a bigger error (corr = 0.60). The 95% band covers 97% of interpolation rows but only 67% of extrapolation rows, so it is over-confident on unfamiliar batteries (e.g. cell 2_092).
# - **The random split is far too optimistic**, especially for the non-linear models (gap: Ridge ≈ 2, SVR ≈ 3.6–4.6, GP ≈ 6.4–6.8).
# - **All models overestimate SOH for batteries near end of life** (bias +9 to +16 points). The largest errors (≈ 56) come from cell 2_095, which drops to 33–38% SOH.
# - **Post-hoc kernel fix (Section 8) did not help.** Adding a linear part gives 7.63 vs 7.72, not significant (p = 0.80); cell 2_092 is still predicted at ≈ 42 instead of ≈ 90. The GP is reported as is.
# - **Locked hold-out (final model only, SVR):** MAE 6.19 on cells 2_009, 2_092, 2_098.
# - **Conclusion:** non-linear models do not improve on Ridge here; Ridge stays the simpler and equally good choice.
