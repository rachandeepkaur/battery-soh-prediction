#!/usr/bin/env python
# coding: utf-8

# # 05 · Evaluation protocol (fixed before training any model)
# 
# This notebook fixes **how every model is scored**, before any baseline or non-linear model is trained.
# Once these rules and folds are saved, nobody changes them based on model results. That's what makes the final comparison fair.
# 
# All the logic lives in [`src/evaluation.py`](../src/evaluation.py), and the leakage checks are in [`tests/test_evaluation.py`](../tests/test_evaluation.py).
# Every modelling notebook should import from there instead of writing its own CV loop.
# 
# | # | Rule | Why |
# |---|------|-----|
# | 1 | **Data:** 306 labelled rows (25 °C reference tests with SOH), 15 cells, 14 profiles | SOH is only defined at 25 °C |
# | 2 | **Inputs:** 42 numeric main inputs + `schedule` (from notebook 01). Comparison and leaky columns are never inputs | They are measured inside the test that defines SOH |
# | 3 | **Primary split: leave-one-cell-out (LOCO, 15 folds).** Also report leave-one-profile-out (LOPO, 14 folds). Random 5-fold is a reference only | The question is "how well do we predict a *new* cell?" |
# | 4 | **Fit everything inside the fold:** pruning, clipping, imputation, scaling, one-hot encoding and tuning are fitted on training rows only | Otherwise the held-out cell leaks into the model |
# | 5 | **Nested tuning:** inner 5-fold GroupKFold grouped like the outer split. Grids are declared up front and not changed after seeing outer scores | Keeps the outer score honest |
# | 6 | **Metrics on pooled out-of-fold predictions:** MAE (headline), RMSE, R², max error; MAE per cell, per health class, interpolation vs extrapolation | One number hides where the model fails |
# | 7 | **Model comparison:** paired per-fold MAE on the same folds, with a Wilcoxon signed-rank test | 15 folds is small, so differences need a test |
# | 8 | **Locked hold-out (secondary):** notebook 01's 3 test cells, scored once per *final* model | Course requirement and a sanity check, not for selection |

# In[1]:


import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from src import evaluation as ev

pd.set_option("display.width", 160)
ev.OUT_DIR.mkdir(parents=True, exist_ok=True)


# ## 1. Data and inputs

# In[2]:


df = ev.load_labelled()
NUM, CAT = ev.main_features()
COMP = ev.comparison_features()
print("labelled rows:", df.shape, "| cells:", df.cell_id.nunique(), "| profiles:", df.profile.nunique())
print("model inputs:", len(NUM), "numeric +", CAT)
print("comparison features (never inputs):", len(COMP))
assert not set(NUM) & set(COMP + ["cap_1C_Ah", "dis1C_energy_Wh", "cap_1C_spread_Ah", "cap_ref_Ah", "SOH"])
print("\nhealth classes:", df.health_class.value_counts().to_dict())


# ## 2. The fixed folds
# 
# The folds are saved to `results/evaluation_protocol/fold_assignment.csv`. Each column says which outer fold a row is **tested** in.
# LOCO and LOPO folds are deterministic (one cell or profile per fold). The random 5-fold uses seed 42.

# In[3]:


folds = ev.fold_assignment(df)
folds.to_csv(ev.OUT_DIR / "fold_assignment.csv", index=False)
folds.head()


# ### 2a. What each LOCO fold tests
# 
# The table shows, for each held-out cell:
# - its SOH range and how many EOL-risk rows it has
# - whether its operating profile lies **inside** the training range of the profile-design settings (interpolation) or **outside** it (extrapolation)
# 
# `V212` has no design parameters, so it is "unclassified".

# In[4]:


rows = []
for k, (tr, te) in enumerate(ev.outer_folds(df, "LOCO")):
    t = df.iloc[te]
    reg = pd.Series(ev.regime_flags(df, tr, te)).unique()
    rows.append(dict(fold=k, cell=t.cell_id.iloc[0], profile=t.profile.iloc[0], rows=len(t),
                     SOH_min=t.SOH.min(), SOH_last=t.SOH.iloc[-1],
                     EOL_rows=int((t.SOH < 80).sum()), regime=reg[0]))
loco_table = pd.DataFrame(rows).round(1)
loco_table.to_csv(ev.OUT_DIR / "loco_fold_overview.csv", index=False)
display(loco_table)
print("folds with no EOL-risk rows:", int((loco_table.EOL_rows == 0).sum()), "of", len(loco_table))
print("regimes:", loco_table.regime.value_counts().to_dict())


# **Reading this table:**
# - **Extrapolation cells.** A few cells were cycled under conditions that no other cell covers. Their errors will be the largest, and that's expected: the model has never seen those conditions. So always report interpolation and extrapolation separately.
# - **Folds without EOL rows.** Many folds contain no EOL-risk rows. That's why EOL error is computed on the **pooled** out-of-fold predictions, not averaged over folds.
# - **L03.** L03 has two cells (2_009 and 2_010). Under LOCO, its sister cell is in training, so LOCO is slightly easier than LOPO for L03.

# ## 3. Leakage checks
# 
# These are the same checks as in `tests/test_evaluation.py`, run here so the notebook stands on its own.

# In[5]:


cols = NUM + CAT + ["cell_id"]
checks = {}

# (a) outer folds never share a cell / profile, and every row is tested exactly once
for scheme in ["LOCO", "LOPO"]:
    g = ev.SCHEMES[scheme]
    sp = ev.outer_folds(df, scheme)
    checks[f"{scheme}: train/test groups disjoint"] = all(not set(df[g].iloc[tr]) & set(df[g].iloc[te]) for tr, te in sp)
    checks[f"{scheme}: every row tested once"] = (np.bincount(np.concatenate([te for _, te in sp]), minlength=len(df)) == 1).all()

# (b) preprocessing ignores the held-out rows: poison them and nothing learned changes
tr, te = ev.outer_folds(df, "LOCO")[0]
clean = ev.make_preprocessor(NUM, CAT).fit(df.iloc[tr][cols], df.SOH.iloc[tr])
bad = df.copy(); bad.loc[bad.index[te], NUM] = 1e6; bad.loc[bad.index[te], "SOH"] = -1e6
poisoned = ev.make_preprocessor(NUM, CAT).fit(bad.iloc[tr][cols], bad.SOH.iloc[tr])
checks["pruner selection unaffected by test rows"] = clean["prune"].selected_ == poisoned["prune"].selected_
checks["fitted transform unaffected by test rows"] = clean.transform(df.iloc[tr][cols]).equals(poisoned.transform(df.iloc[tr][cols]))

# (c) categorical handling
out = clean.transform(df.iloc[te][cols])
checks["schedule -> one 0/1 column (drop='if_binary')"] = [c for c in out.columns if c.startswith("schedule")] == ["schedule_RTP_highCurrent"]
checks["ids (cell_id, profile) are not model inputs"] = not {"cell_id", "profile"} & set(out.columns)

pd.Series(checks, name="passed").to_frame()


# In[6]:


assert all(checks.values())
print("features kept by the pruner in fold 0:", len(clean["prune"].selected_), "of", len(NUM))
print("model matrix shape (fold 0 test rows):", out.shape)


# ## 4. Reference numbers: the mean baseline under every split
# 
# The mean baseline predicts the average SOH of the training rows. Any useful model must beat it under the **same** split.
# It also shows that the splits are wired correctly: random CV is only slightly better for the mean model, because the mean has nothing to overfit.

# In[7]:


mean_spec = ev.ModelSpec("Mean baseline", DummyRegressor(), family="baseline")
ref = [ev.run_cv(df, mean_spec, s, verbose=False) for s in ev.SCHEMES]
ref_table = ev.summary_table(ref).round(3)
ref_table.to_csv(ev.OUT_DIR / "reference_mean_baseline.csv", index=False)
ref_table


# In[8]:


loco_mean = ref[0]
display(ev.breakdown(loco_mean.oof, "health_class").round(2))
display(ev.breakdown(loco_mean.oof, "regime").round(2))


# ## 5. How teammates use this
# 
# Use this pattern in the baseline and non-linear notebooks. The hyperparameter grid is written **before** running and is not edited after seeing LOCO scores.
# 
# ```python
# from sklearn.linear_model import Ridge
# from sklearn.ensemble import RandomForestRegressor
# from src import evaluation as ev
# 
# df = ev.load_labelled()
# specs = [
#     ev.ModelSpec("Ridge", Ridge(), {"alpha": np.logspace(-2, 3, 11)}, family="linear"),
#     ev.ModelSpec("Random forest", RandomForestRegressor(300, random_state=0),
#                  {"min_samples_leaf": [2, 5, 10], "max_features": [0.3, 0.6, 1.0]}, family="non-linear"),
# ]
# results = [ev.run_cv(df, s, scheme) for s in specs for scheme in ev.SCHEMES]
# 
# ev.summary_table(results)                       # pooled MAE / RMSE / R² / max error
# ev.generalization_gap(results)                  # grouped minus random MAE
# ev.breakdown(results[0].oof, "regime")          # interpolation vs extrapolation
# ev.breakdown(results[0].oof, "health_class")    # error per health class
# ev.paired_comparison(rf_loco, ridge_loco)       # same folds, Wilcoxon on per-fold MAE
# ev.holdout_check(df, final_spec)                # once, for the final models only
# ```
# 
# **Output files.** Each notebook saves `oof` (out-of-fold predictions) and `folds` (per-fold MAE and chosen hyperparameters) under `results/<notebook>/`. Shraddha can then build the classification view (thresholds 80 and 90) from the same out-of-fold predictions.
# 
# **Things not to do:**
# - **Don't fit any scaler, imputer or selector on the full table.** Use the pipeline.
# - **Don't report random-split scores as the main result.** They are only the "how optimistic is random CV" reference.
# - **Don't pick the model or grid by looking at the 3-cell hold-out.**
# - **Don't oversample (SMOTE) across cells.** If class balance matters, use class weights.

# ## 6. Saved files
# 
# | File | Content |
# |------|---------|
# | `results/evaluation_protocol/fold_assignment.csv` | The fixed outer folds for LOCO, LOPO and random 5-fold |
# | `results/evaluation_protocol/loco_fold_overview.csv` | What each held-out cell tests (SOH range, EOL rows, regime) |
# | `results/evaluation_protocol/reference_mean_baseline.csv` | Mean-baseline floor for each split |
