#!/usr/bin/env python
# coding: utf-8

# # 08 · Model comparison
# 
# This notebook puts every team model side by side and answers three questions:
# 1. **Which model is most accurate on a battery it has never seen?**
# 2. **Is the difference real, or just luck?**
# 3. **Where does each model go wrong?**
# 
# **Nothing is retrained here.** We read the saved out-of-fold predictions from each teammate's notebook and score them all the same way.
# All three notebooks used the shared protocol (`src/evaluation.py`, notebook 05), so the folds, rows and inputs are identical.
# 
# | Source notebook | Models | File |
# |---|---|---|
# | `07_baselines` | Mean, one-feature, MLR, Ridge, Lasso, ElasticNet | `results/baselines/oof_baselines.csv` |
# | `04_nonlinear_models` | SVR, Gaussian process (+ a Ridge copy) | `results/nonlinear_svr_gpr/oof_predictions.csv` |
# | `06_modelling_random_forest_hgb` | Random forest, HistGradientBoosting | `results/modelling_random_forest_hgb/oof_rf_hgb.csv` |
# 
# **How to read the scores:** MAE is in **SOH points**. An MAE of 4 means predictions are off by 4 percentage points of health on average. Lower is better.

# In[1]:


import sys, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (mean_absolute_error as mae, root_mean_squared_error as rmse, r2_score,
                             f1_score, recall_score, confusion_matrix)
from src import evaluation as ev

pd.set_option("display.width", 180)
OUT = ROOT / "results" / "model_comparison"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# One colour per model family (reference palette): grey = baselines, blue = linear, orange = non-linear
FAMILY_COLOR = {"baseline": "#8a8984", "linear": "#2a78d6", "non-linear": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": GRID,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 10})


# ## 1. Load everyone's predictions and check they match

# In[2]:


FILES = {
    "07_baselines":   ROOT / "results/baselines/oof_baselines.csv",
    "04_nonlinear":   ROOT / "results/nonlinear_svr_gpr/oof_predictions.csv",
    "06_rf_hgb":      ROOT / "results/modelling_random_forest_hgb/oof_rf_hgb.csv",
}
raw = pd.concat([pd.read_csv(p).assign(source=k) for k, p in FILES.items()], ignore_index=True)

# Same rows, same true SOH, same folds as the fixed fold file?
folds = pd.read_csv(ev.OUT_DIR / "fold_assignment.csv")
checks = {}
for (src, model, scheme), g in raw.groupby(["source", "model", "scheme"]):
    j = g.merge(folds, on=["cell_id", "idtest"], how="left")
    checks[(src, model, scheme)] = len(g) == 306 and (j.fold == j[f"fold_{scheme}"]).all()
assert all(checks.values()), [k for k, v in checks.items() if not v]
# same truth in every file (rounded: files written on different machines differ by ~1e-14)
assert raw.y_true.round(9).groupby([raw.cell_id, raw.idtest]).nunique().max() == 1
print(f"{raw.groupby(['model', 'scheme']).ngroups} model x split runs, all with 306 rows, identical truth and identical folds")


# In[3]:


# Ridge was run twice (07 'Ridge (tuned)' and 04 'Ridge'). Same predictions? Then keep one.
r = raw[raw.model.isin(["Ridge", "Ridge (tuned)"])].pivot_table(index=["cell_id", "idtest", "scheme"],
                                                                 columns="model", values="y_pred")
print("largest difference between the two Ridge runs:", float((r["Ridge"] - r["Ridge (tuned)"]).abs().max()))

# Keep the baselines notebook's Ridge copy. The post-hoc GP kernel fix from notebook 04 is not part of the
# pre-declared comparison; its result is quoted in Section 7.
oof = raw[~raw.model.isin(["Ridge", "GP linear+RBF (post-hoc)"])].copy()
oof["model"] = oof.model.replace({"Ridge (tuned)": "Ridge", "Lasso (tuned)": "Lasso",
                                  "ElasticNet (tuned)": "ElasticNet", "Mean SOH (training cells)": "Mean baseline",
                                  "MLR (pruned main set)": "MLR", "HistGradientBoosting": "Gradient boosting (HGB)"})
FAMILY = {"Mean baseline": "baseline", "One feature: cum_EFC": "baseline", "One feature: days_since_start": "baseline",
          "MLR": "linear", "Ridge": "linear", "Lasso": "linear", "ElasticNet": "linear",
          "SVR (RBF)": "non-linear", "Gaussian process": "non-linear",
          "Random forest": "non-linear", "Gradient boosting (HGB)": "non-linear"}
assert set(oof.model) == set(FAMILY), set(oof.model) ^ set(FAMILY)
oof["family"] = oof.model.map(FAMILY)
loco = oof[oof.scheme == "LOCO"]
print(len(FAMILY), "models kept:", sorted(FAMILY))


# **Check passed.** All 39 runs (13 models × 3 splits, counting Ridge twice and notebook 04's post-hoc GP) cover the same 306 rows, with the same true SOH and the same folds, so the comparison is fair.
# 
# The two Ridge runs are the same model: their predictions differ by about 0.000000000002 (rounding noise). We keep one.

# ## 2. The main table: accuracy on unseen batteries (leave-one-cell-out)
# 
# * **MAE** = average miss, in SOH points (headline)
# * **RMSE** = like MAE, but punishes big misses more
# * **R²** = share of the variation explained (1 = perfect, 0 = no better than guessing the average)
# * **Max error** = the single worst miss
# * **Within 5 / 10 pts** = share of predictions within 5 or 10 SOH points of the truth
# * **Relative accuracy** = 100 − average % error (same definition as notebook 06)

# In[4]:


def scores(g):
    y, p = g.y_true.to_numpy(), g.y_pred.to_numpy()
    err = np.abs(p - y)
    return pd.Series({"MAE": mae(y, p), "RMSE": rmse(y, p), "R2": r2_score(y, p), "Max error": err.max(),
                      "Within 5 pts %": 100 * (err <= 5).mean(), "Within 10 pts %": 100 * (err <= 10).mean(),
                      "Relative accuracy %": 100 - 100 * (err / y).mean()})

all_scores = oof.groupby(["model", "scheme"]).apply(scores).reset_index()
all_scores["family"] = all_scores.model.map(FAMILY)
main = (all_scores[all_scores.scheme == "LOCO"].drop(columns="scheme")
        .sort_values("MAE").set_index("model"))
main.round(2)


# **What this says**
# * **Top two:** ElasticNet (4.25) and gradient boosting (4.27) are the most accurate on unseen batteries, about half the error of the mean baseline (8.18).
# * **A middle group:** Lasso, random forest, Ridge and SVR sit close behind, between 4.5 and 4.8.
# * **Plain MLR is clearly worse (5.85).** With no regularisation, its correlated inputs push the coefficients around.
# * **The Gaussian process (7.72) barely beats guessing the average.** Section 2a shows why.
# * **About "relative accuracy":** it sounds great (about 94% for the top models), but the mean baseline already scores 89%, because SOH sits between 33 and 100%. Report MAE and "within 5 points" instead. They're easier to read and harder to misread.

# In[5]:


fig, ax = plt.subplots(figsize=(8, 4.6))
m = main.sort_values("MAE", ascending=False)
ax.barh(m.index, m.MAE, color=[FAMILY_COLOR[f] for f in m.family], height=0.62, edgecolor="#fcfcfb", linewidth=2)
for y, v in enumerate(m.MAE):
    ax.text(v + 0.1, y, f"{v:.2f}", va="center", fontsize=9, color=MUTED)
base = main.loc["Mean baseline", "MAE"]
ax.axvline(base, color=MUTED, lw=1, ls="--")
ax.text(base, len(m) - 0.4, " mean baseline", color=MUTED, fontsize=8, va="bottom")
ax.set(xlabel="MAE on unseen batteries (SOH points, lower is better)", title="Leave-one-cell-out MAE by model")
ax.grid(axis="y", visible=False)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLOR.values()],
          labels=list(FAMILY_COLOR), loc="upper right", bbox_to_anchor=(0.98, 0.9), frameon=False)
fig.tight_layout(); fig.savefig(FIG / "loco_mae_by_model.png", dpi=150); plt.show()


# ### 2a. The same models under the other two splits
# 
# * **LOPO** = leave a whole *profile* out (harder: new operating conditions)
# * **Random 5-fold** = rows shuffled at random (too easy: neighbouring check-ups of the same battery leak into training)
# * **Gap** = LOCO MAE − random MAE. A big gap means the model looks great on random splits but doesn't transfer to new batteries.

# In[6]:


splits = all_scores.pivot(index="model", columns="scheme", values="MAE")[["LOCO", "LOPO", "Random 5-fold"]]
splits["Gap (LOCO - random)"] = splits["LOCO"] - splits["Random 5-fold"]
splits = splits.loc[main.index]
splits.round(2)


# In[7]:


fig, ax = plt.subplots(figsize=(8, 4.6))
s = splits.sort_values("LOCO", ascending=False)
yy = np.arange(len(s))
ax.hlines(yy, s["Random 5-fold"], s["LOCO"], color=GRID, lw=3)
ax.scatter(s["Random 5-fold"], yy, s=60, color="#8a8984", zorder=3, label="Random 5-fold (optimistic)",
           edgecolor="#fcfcfb", linewidth=2)
ax.scatter(s["LOCO"], yy, s=60, color="#2a78d6", zorder=3, label="Leave-one-cell-out (honest)",
           edgecolor="#fcfcfb", linewidth=2)
ax.set_yticks(yy, s.index)
ax.set(xlabel="MAE (SOH points)", title="How much a random split flatters each model")
ax.grid(axis="y", visible=False)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(FIG / "random_vs_loco_gap.png", dpi=150); plt.show()


# **Random splits flatter every model that learns something.** The two baselines barely move (gap below 1), because they have nothing to memorise.
# * **Gaussian process:** looks like the best model on a random split (1.30) and is nearly the worst on new batteries (7.72). It memorises each battery's curve and falls back to the average when it meets a new one.
# * **SVR and the tree models:** gaps of 2.6–3.6. They also lean on battery-specific patterns.
# * **ElasticNet:** the smallest gap of the real models (1.45). Its random-split score is closest to what it actually delivers.
# 
# This is the main reliability point for the report: **a random split would have picked the wrong model.**

# ## 3. Is the difference real? Paired tests on the same 15 folds
# 
# Every model was tested on the same 15 held-out batteries, so we compare them **battery by battery**:
# * **Wins** = held-out batteries where the model beats the reference
# * **Wilcoxon p** = how likely a difference this consistent is by chance
# * **Holm p** = the same p-value, adjusted because we run many tests at once. **Use this one.** Below 0.05 counts as a real difference.

# In[8]:


fold_mae = loco.groupby(["model", "fold"]).apply(lambda g: mae(g.y_true, g.y_pred)).unstack(0)
best = main.index[0]
rows = []
for ref in [best, "Ridge"]:
    for m in main.index:
        if m == ref:
            continue
        rows.append(dict(model=m, reference=ref, **ev.paired_test(fold_mae[m], fold_mae[ref])))
paired = pd.DataFrame(rows)
paired["holm_p"] = ev.holm(paired.wilcoxon_p)
paired = paired.rename(columns={"a_wins": "model wins", "b_wins": "reference wins",
                                "mean_diff": "MAE diff (model - ref)"})
print("best model on LOCO:", best)
paired.round(3)


# **After correcting for multiple tests, none of the top six models is significantly better than another.**
# * **ElasticNet vs gradient boosting:** 8 to 7 battery wins. A coin flip.
# * **ElasticNet vs Ridge:** ElasticNet wins 10 of 15 batteries (raw p = 0.026), but that disappears after the Holm correction (0.33).
# * **ElasticNet vs the mean baseline:** the only comparison that holds up (14 of 15 wins, Holm p = 0.04).
# 
# With only 15 test batteries, differences under about 0.5 MAE can't be told apart from luck. **Treat the top group as a tie.** Choose between them on reliability and simplicity, not on the second decimal.

# ## 4. Where do the models go wrong?
# 
# Two splits of the same leave-one-cell-out predictions:
# * **Regime:** is the held-out battery's operating profile *inside* the range the model trained on (interpolation) or *outside* it (extrapolation)?
# * **Health class:** Healthy (≥ 90), Degraded (80–90), EOL risk (< 80)

# In[9]:


def by(col):
    t = loco.groupby(["model", col], observed=True).apply(lambda g: mae(g.y_true, g.y_pred)).unstack(col)
    return t.loc[main.index]

regime = by("regime")[["interpolation", "extrapolation", "unclassified"]]
health = by("health_class")[["Healthy (>=90)", "Degraded (80-90)", "EOL risk (<80)"]]
bias = loco.groupby(["model", "health_class"], observed=True).apply(lambda g: (g.y_pred - g.y_true).mean()) \
           .unstack("health_class").loc[main.index, ["Healthy (>=90)", "Degraded (80-90)", "EOL risk (<80)"]]
print("MAE by regime (rows:", loco[loco.model == best].regime.value_counts().to_dict(), ")")
display(regime.round(2))
print("MAE by health class (rows:", loco[loco.model == best].health_class.value_counts().to_dict(), ")")
display(health.round(2))
print("Bias by health class (+ = predicts too healthy)")
display(bias.round(2))


# In[10]:


FINALISTS = ["ElasticNet", "Ridge", "Gradient boosting (HGB)", "Random forest"]
STYLE = {"ElasticNet": ("#2a78d6", None), "Ridge": ("#2a78d6", "///"),
         "Gradient boosting (HGB)": ("#eb6834", None), "Random forest": ("#eb6834", "///")}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, (tab, title) in zip(axes, [(regime[["interpolation", "extrapolation"]], "By regime"),
                                   (health, "By health class")]):
    t = tab.loc[FINALISTS]
    x = np.arange(t.shape[1]); w = 0.8 / len(FINALISTS)
    for i, mname in enumerate(FINALISTS):
        colour, hatch = STYLE[mname]
        bars = ax.bar(x + i * w - 0.4 + w / 2, t.loc[mname], width=w, color=colour if hatch is None else "#fcfcfb",
                      edgecolor=colour, hatch=hatch, linewidth=1.5, label=mname)
    ax.set_xticks(x, t.columns)
    ax.set(ylabel="MAE (SOH points)", title=title)
    ax.grid(axis="x", visible=False)
axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
fig.suptitle("Where the errors come from (leave-one-cell-out, four finalists)", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "error_by_regime_and_health.png", dpi=150); plt.show()


# **This is where linear and non-linear models really differ.**
# * **Familiar conditions (interpolation):** all models do well, 2–3 points. Random forest is best (2.10).
# * **Unfamiliar conditions (extrapolation):** errors jump to 10–14 points. ElasticNet holds up best (9.66). Random forest is the worst of the top models (13.63), because trees can't predict outside the values they were trained on.
# * **Healthy batteries:** gradient boosting is best (2.02).
# * **Worn-out batteries (EOL < 80%):** every model misses by 10+ points, and the bias table shows they all guess **too healthy** (+8.6 to +12.6). Ridge and ElasticNet miss the least (about 10); the tree models and SVR miss 13–14.
# 
# **In short:** the non-linear models are sharper on familiar, healthy batteries. The linear models are safer on new conditions and worn-out batteries, which are the cases that matter most for safety.

# ### 4a. Error for each held-out battery
# 
# Each cell of the heatmap is one model tested on one unseen battery. Darker means a bigger miss.

# In[11]:


cell_info = (loco.groupby("cell_id").agg(profile=("profile", "first"), regime=("regime", "first"),
                                         min_SOH=("y_true", "min")))
per_cell = loco.groupby(["model", "cell_id"]).apply(lambda g: mae(g.y_true, g.y_pred)).unstack("cell_id")
order = cell_info.sort_values("min_SOH", ascending=False).index
per_cell = per_cell.loc[main.index, order]

fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(per_cell.to_numpy(), cmap=BLUES, aspect="auto", vmin=0, vmax=20)
ax.set_xticks(range(len(order)), [f"{c}\n{cell_info.profile[c]}\nmin {cell_info.min_SOH[c]:.0f}%" for c in order],
              fontsize=7.5)
ax.set_yticks(range(len(per_cell)), per_cell.index)
for i in range(per_cell.shape[0]):
    for j in range(per_cell.shape[1]):
        v = per_cell.iat[i, j]
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white" if v > 11 else INK)
for j, c in enumerate(order):
    if cell_info.regime[c] == "extrapolation":
        ax.get_xticklabels()[j].set_fontweight("bold")
ax.grid(False)
ax.set_title("MAE per held-out battery (sorted from healthiest to most worn; bold = extrapolation)")
fig.colorbar(im, ax=ax, label="MAE (SOH points)", shrink=0.8)
fig.tight_layout(); fig.savefig(FIG / "mae_per_battery_heatmap.png", dpi=150); plt.show()


# In[12]:


hardest = per_cell.loc[[m for m in main.index if FAMILY[m] != "baseline"]].median().sort_values(ascending=False)
pd.DataFrame({"median MAE across models": hardest.round(2)}).join(cell_info).head(6)


# **Most of the error comes from four batteries, all of them extrapolation cases:**
# * **2_095** (L206): stored at 45 °C and 95% charge, higher than any other profile; falls to 33% SOH
# * **2_076** (L223): the only profile charged at 0.5C (all others ≤ 0.2C)
# * **2_092** (L202): stored at 45 °C and only 15% charge, lower than any other profile
# * **2_012** (L35): its `lp_F` setting is 0.5, while every other profile has 0.25 or less
# 
# Each was aged under conditions no other battery in the training set covers, so no model can learn them from the data we have. On the other 11 batteries, the good models are typically within 1–3 points.
# 
# **Worth noting:** gradient boosting and random forest do much worse than ElasticNet on 2_012 (15–16 vs 7), while ElasticNet does worse on 2_092 (10 vs 6). There is no single winner on every battery.

# ### 4b. Predicted vs true SOH (best linear and best non-linear model)
# 
# Points on the diagonal are perfect predictions. Points **above** the line mean the model thinks the battery is healthier than it really is (the risky direction).

# In[13]:


best_lin = next(m for m in main.index if FAMILY[m] == "linear")
best_nl = next(m for m in main.index if FAMILY[m] == "non-linear")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
for ax, mname in zip(axes, [best_lin, best_nl]):
    g = loco[loco.model == mname]
    for reg, mk in [("interpolation", "o"), ("extrapolation", "^"), ("unclassified", "s")]:
        h = g[g.regime == reg]
        ax.scatter(h.y_true, h.y_pred, s=22, marker=mk, color=FAMILY_COLOR[FAMILY[mname]],
                   alpha=0.85 if reg == "interpolation" else 0.45, edgecolor="#fcfcfb", linewidth=0.6, label=reg)
    ax.plot([30, 102], [30, 102], color=MUTED, lw=1, ls="--")
    for t in (80, 90):
        ax.axvline(t, color=GRID, lw=1); ax.axhline(t, color=GRID, lw=1)
    ax.set(title=f"{mname}  (MAE {main.loc[mname, 'MAE']:.2f})", xlabel="true SOH (%)")
axes[0].set_ylabel("predicted SOH (%)")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "predicted_vs_true.png", dpi=150); plt.show()


# **Both models squash the extremes.** Above 85% SOH they track the diagonal closely. Below 70% they flatten out and predict 70–85% for batteries that are really at 33–60% (the triangles, mostly extrapolation cells). That's the "too healthy" bias from Section 4.
# 
# Gradient boosting also has a band of points stuck at about 85%: a tree can't predict beyond the leaves it learned, so it caps out.

# ## 5. Classification view: turn predicted SOH into health classes
# 
# This uses the same predictions, cut at 80 and 90, so Shraddha's classification can be compared on the same footing.
# * **Macro-F1** gives each class equal weight, so the small EOL class counts as much as the big Healthy class.
# * **EOL recall** = the share of truly worn-out check-ups (SOH < 80) the model flags as EOL. Missing them is the costly error.

# In[14]:


labels = ["EOL risk (<80)", "Degraded (80-90)", "Healthy (>=90)"]
to_class = lambda s: pd.cut(s, ev.HEALTH_BINS, right=False, labels=labels)
cls = []
for mname, g in loco.groupby("model"):
    t, p = to_class(g.y_true), to_class(g.y_pred)
    cls.append(dict(model=mname, accuracy=(t == p).mean(), macro_F1=f1_score(t, p, average="macro"),
                    EOL_recall=recall_score(t, p, labels=[labels[0]], average="macro"),
                    EOL_missed=int(((t == labels[0]) & (p != labels[0])).sum())))
cls = pd.DataFrame(cls).set_index("model").loc[main.index]
cls.round(3)


# In[15]:


fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
for ax, mname in zip(axes, [best_lin, best_nl]):
    g = loco[loco.model == mname]
    cm = confusion_matrix(to_class(g.y_true), to_class(g.y_pred), labels=labels)
    ax.imshow(cm, cmap=BLUES)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() * 0.55 else INK)
    short = ["EOL", "Degraded", "Healthy"]
    ax.set_xticks(range(3), short); ax.set_yticks(range(3), short)
    ax.set(xlabel="predicted class", ylabel="true class", title=mname); ax.grid(False)
fig.tight_layout(); fig.savefig(FIG / "confusion_matrices.png", dpi=150); plt.show()


# **For health classes, Ridge and ElasticNet are the safest choice.**
# * **Ridge** catches the most EOL batteries: 40 of 53 (recall 0.76). Its macro-F1 (0.73) is also the highest.
# * **ElasticNet** is close behind (recall 0.72).
# * **The non-linear models miss many more:** random forest misses 25 of 53, SVR misses 27.
# 
# The mean baseline flags none at all. A model that labels a worn-out battery "healthy" is the dangerous mistake, so this table favours the linear models, even though their MAE ties with gradient boosting.
# 
# **For Shraddha:** this is the regression-then-threshold baseline. A direct classifier should beat these numbers to be worth keeping.

# ## 6. Locked hold-out (notebook 01's 3 test batteries)
# 
# This is the course-style "final test set". The protocol says to score each **final** model on it once and never use it to choose models.

# In[16]:


hold = pd.concat([pd.read_csv(ROOT / "results/baselines/holdout_check.csv"),
                  pd.read_csv(ROOT / "results/nonlinear_svr_gpr/holdout_check.csv"),
                  pd.read_csv(ROOT / "results/modelling_random_forest_hgb/holdout_check.csv")], ignore_index=True)
hold["model"] = hold.model.replace({"HistGradientBoosting": "Gradient boosting (HGB)",
                                    "ElasticNet (tuned)": "ElasticNet", "Ridge (tuned)": "Ridge"})
hold = hold.set_index("model")
hold["LOCO MAE on the same 3 batteries"] = [
    mae(*loco[(loco.model == m) & loco.cell_id.isin(["2_009", "2_092", "2_098"])][["y_true", "y_pred"]].T.to_numpy())
    for m in hold.index]
display(hold.round(2))
print("hold-out still missing for:", [m for m in FINALISTS if m not in hold.index])


# **All five final candidates now have a hold-out score.**
# * **Tree models do best here:** gradient boosting 2.94 (LOCO on the same batteries 3.07, consistent) and random forest 3.89.
#   Random forest has the lowest RMSE (5.88), so it makes the fewest large misses.
# * **Linear models:** ElasticNet 5.27 and Ridge 5.40. ElasticNet is a little worse than its LOCO score on the same batteries (4.43).
# * **SVR** scores 6.19, worse than its LOCO 4.16 on the same batteries. It is less stable when trained on fewer batteries.
# * **One battery drives the linear-vs-tree gap.** On 2_092 (stored at 45 °C, outside the training range) ElasticNet's LOCO
#   error is 9.88 against 6.04 for gradient boosting. On the other two batteries the models are close.
# 
# **This does not change the model choice.** Three batteries is too few to overrule a 15-battery comparison, and the protocol
# forbids picking a model on the hold-out. They also contain no EOL rows (lowest SOH 87%), so the hold-out only tests healthy
# batteries. Report it as a secondary check alongside the LOCO results.

# ## 7. Extra (post-hoc): averaging ElasticNet and gradient boosting
# 
# **This was tried after seeing the results above, so it is labelled post-hoc.** The idea comes from Section 4: ElasticNet is
# safer on new conditions and worn-out batteries, gradient boosting is sharper on familiar, healthy ones. Averaging their
# predictions might keep the strengths of both.
# 
# * **Nothing is retrained.** For every row, the averaged prediction is (ElasticNet + gradient boosting) / 2, using the saved
#   out-of-fold predictions. Both models were scored on the same folds, so the average is also a fair out-of-fold prediction.
# * **Not run on the locked hold-out.** The average was designed after seeing scores, so it must not touch the 3 test batteries.
# 
# **The other post-hoc change, the GP kernel fix (notebook 04, Section 8), did not help:** adding a linear part to the kernel
# gives 7.63 on unseen batteries against 7.72 before (p = 0.80), still far behind Ridge (4.76). It is left out of the tables in
# this notebook, and the original GP is reported as is.

# In[17]:


PAIR = ["ElasticNet", "Gradient boosting (HGB)"]
AVG = "Average (ElasticNet + HGB)"
key = ["cell_id", "idtest", "scheme"]
w = oof[oof.model.isin(PAIR)].pivot_table(index=key, columns="model", values="y_pred")
assert len(w) == 3 * 306 and w.notna().all().all()          # every row, every split, both models
avg = (oof[oof.model == PAIR[0]].drop(columns="y_pred").set_index(key)
       .assign(y_pred=w.mean(axis=1), model=AVG).reset_index())
ens = pd.concat([oof[oof.model.isin(PAIR)], avg], ignore_index=True)

ens_scores = ens.groupby(["model", "scheme"]).apply(scores).reset_index()
ens_split = ens_scores.pivot(index="model", columns="scheme", values="MAE")[["LOCO", "LOPO", "Random 5-fold"]]
print("MAE by split")
display(ens_split.round(2))
print("All scores, leave-one-cell-out")
display(ens_scores[ens_scores.scheme == "LOCO"].set_index("model").drop(columns="scheme").round(2))


# In[18]:


# Is the average better battery by battery (LOCO) and profile by profile (LOPO)? Holm-corrected over the 4 tests.
rows = []
for s in ["LOCO", "LOPO"]:
    f = ens[ens.scheme == s].groupby(["model", "fold"]).apply(lambda g: mae(g.y_true, g.y_pred)).unstack(0)
    for ref in PAIR:
        rows.append(dict(split=s, reference=ref, **ev.paired_test(f[AVG], f[ref])))
ens_paired = pd.DataFrame(rows)
ens_paired["holm_p"] = ev.holm(ens_paired.wilcoxon_p)
ens_paired = ens_paired.rename(columns={"a_wins": "average wins", "b_wins": "reference wins",
                                        "mean_diff": "MAE diff (avg - ref)"})
ens_paired.round(3)


# In[19]:


# Where does averaging help? Same breakdowns as Section 4 and 5, leave-one-cell-out only.
L = ens[ens.scheme == "LOCO"]
HC = ["Healthy (>=90)", "Degraded (80-90)", "EOL risk (<80)"]
def by_ens(col, f=lambda g: mae(g.y_true, g.y_pred)):
    return L.groupby(["model", col], observed=True).apply(f).unstack(col)

ens_regime = by_ens("regime")[["interpolation", "extrapolation", "unclassified"]]
ens_health = by_ens("health_class")[HC]
ens_bias = by_ens("health_class", lambda g: (g.y_pred - g.y_true).mean())[HC]
ens_eol = L.groupby("model").apply(lambda g: recall_score(to_class(g.y_true), to_class(g.y_pred),
                                                          labels=[labels[0]], average="macro")).rename("EOL recall")
ens_cells = by_ens("cell_id").T.loc[order]
print("MAE by regime"); display(ens_regime.round(2))
print("MAE by health class"); display(ens_health.round(2))
print("Bias by health class (+ = predicts too healthy)"); display(ens_bias.round(2))
print("EOL recall"); display(ens_eol.round(3).to_frame())
print("MAE per held-out battery (healthiest first)"); display(ens_cells.round(2))


# **Averaging helps a little, but it is not clearly better than either model alone.**
# * **Lowest error of any model:** 3.97 on unseen batteries and 4.20 on unseen profiles, about 0.25–0.3 better than
#   ElasticNet (4.25 / 4.45) and gradient boosting (4.27 / 4.49).
# * **Against ElasticNet the gain is consistent but not proven.** The average wins 11 of 15 batteries and 12 of 14 profiles,
#   but after the Holm correction p = 0.25 and 0.14.
# * **Against gradient boosting it is a coin flip:** 7 of 15 and 5 of 14 wins (Holm p = 1.0).
# * **Where it helps:** familiar conditions (interpolation 2.09, against 2.54 for ElasticNet) and fewer big misses than
#   gradient boosting (RMSE 7.05 against 7.72).
# * **Where it doesn't:** unfamiliar conditions (extrapolation 10.13, against 9.66 for ElasticNet) and worn-out batteries
#   (EOL recall 0.68, against 0.72; still +9 points too healthy). On 2_012 the tree model drags the average from 7.2 to 11.0.
# * **The four hard batteries stay hard** (7–12 points). Averaging can't fix conditions missing from the training data.
# 
# **Verdict:** report the average as a post-hoc extra. Keep ElasticNet as the main model: it is simpler, explainable, and safer
# on new conditions and worn-out batteries. If the only goal is the lowest average error, the average is the better pick.

# ## 8. Save the comparison tables

# In[20]:


main.round(4).to_csv(OUT / "loco_scores.csv")
splits.round(4).to_csv(OUT / "mae_by_split.csv")
paired.round(4).to_csv(OUT / "paired_tests_loco.csv", index=False)
regime.round(4).to_csv(OUT / "loco_mae_by_regime.csv")
health.round(4).to_csv(OUT / "loco_mae_by_health_class.csv")
bias.round(4).to_csv(OUT / "loco_bias_by_health_class.csv")
per_cell.round(4).to_csv(OUT / "loco_mae_per_battery.csv")
cls.round(4).to_csv(OUT / "loco_classification_view.csv")
ens_split.round(4).to_csv(OUT / "averaged_mae_by_split.csv")
ens_paired.round(4).to_csv(OUT / "averaged_paired_tests.csv", index=False)
pd.concat([ens_regime, ens_health, ens_bias.add_prefix("bias "), ens_eol], axis=1).round(4) \
    .to_csv(OUT / "averaged_loco_breakdown.csv")
print(sorted(p.name for p in OUT.iterdir()))


# ## Summary: what to take away
# 
# 1. **The best models halve the error of guessing the average.** ElasticNet (4.25) and gradient boosting (4.27) lead; the mean baseline is 8.18.
# 2. **The top six models are statistically tied.** With 15 test batteries, differences under about 0.5 MAE are within noise (Holm-corrected Wilcoxon test).
# 3. **Non-linear isn't better on new batteries.** Trees are sharper on familiar, healthy batteries (RF 2.10 on interpolation, HGB 2.02 on healthy rows). Linear models are safer on new conditions and worn-out batteries (ElasticNet 9.66 on extrapolation vs RF 13.63).
# 4. **Random splits mislead.** The Gaussian process looks best on a random split (1.30) and is nearly worst on new batteries (7.72). Every learned model looks 1.5–6.4 points better than it really is.
# 5. **Four batteries cause most of the error.** All four were aged under conditions no other battery covers. That's a data limit, not a modelling bug, and it belongs in the limitations section.
# 6. **Every model calls worn-out batteries too healthy** (+8.6 to +12.6 points for the top six). For classification, Ridge and ElasticNet catch the most EOL batteries (76% and 72%).
# 7. **Averaging ElasticNet and gradient boosting (post-hoc) gives the lowest error, 3.97,** but the gain is not significant after correction, and it is slightly worse than ElasticNet on new conditions and worn-out batteries.
# 8. **Fixing the Gaussian process kernel didn't help** (7.63 vs 7.72, notebook 04 Section 8). The GP is reported as is: weak predictions, but its uncertainty flags unfamiliar batteries.
# 9. **On the locked hold-out, the tree models score best** (HGB 2.94, RF 3.89, ElasticNet 5.27), mostly because of one battery (2_092). Three batteries is too few to change the choice.
# 
# **Recommended final model: ElasticNet.** It ties for the lowest error, has the smallest random-vs-real gap, is best on extrapolation, catches the most EOL batteries after Ridge, and its coefficients can be explained. Report gradient boosting alongside it as the strongest non-linear model.
# 
# **Still to do before the report**
# * Shraddha: compare the direct classifier against the Section 5 table.
