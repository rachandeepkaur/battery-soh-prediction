#!/usr/bin/env python
# coding: utf-8

# # 01 · Data preprocessing
# 
# This notebook goes from the raw battery files to clean, model-ready **train / validation / test** tables.
# 
# 
# **The key rule:** anything that *learns a number from the data* (a median, a percentile, a mean, a list of categories)
# is learned **after** the split and **from the training rows only**. The validation and test rows are only transformed with
# those learned numbers. Otherwise information from the test batteries would leak into training.
# 
# Sections 3 and 4 only *look* at the data. They don't change it, so it's fine to run them on all rows.
# 
# The raw-to-table step itself (HDF5 → one row per reference test) is in `src/features.py`; see the README for the cleaning rules
# (match tests by ID, temperature sentinels → missing, capacity measured within one step, SOH at 25 °C only).

# In[1]:


from pathlib import Path
import sys, warnings
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
from src.features import read_info

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 100)
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "preprocessed"; OUT.mkdir(parents=True, exist_ok=True)
SEED = 42


# ## 1. Data shape
# 
# ### 1a. Raw data: one HDF5 file per battery cell
# 
# Each file holds an `info` table (one line per test: name, start time, schedule) and a `tests/` folder with the raw
# time series of each test (voltage, current, temperature, … logged every few seconds).

# In[2]:


with h5py.File(RAW / "2_009.h5", "r") as f:
    print("top-level groups:", list(f.keys()))
    print("info table:", f["info"].shape, "rows | columns:", f["info"].dtype.names)
    first = f["tests"][sorted(f["tests"].keys())[0]]
    print("one test contains:", {k: first[k].shape for k in first.keys()})


# In[3]:


rows = []
for p in sorted(RAW.glob("*.h5")):
    with h5py.File(p, "r") as f:
        info = read_info(f)
        n_meas = sum(f["tests"][g]["voltage"].shape[0] for g in info.group)
        profile = next((e for e in info.expname if not e.startswith(("RTP", "SetSoC"))), None)
        rows.append(dict(cell_id=p.stem, profile=profile, n_tests=len(info),
                         n_reference_tests=int(info.expname.str.startswith("RTP").sum()),
                         n_RTP25=int((info.expname == "RTP25").sum()), measurement_rows=n_meas))
raw_summary = pd.DataFrame(rows)
display(raw_summary)
print(f"TOTAL: {len(raw_summary)} cells, {raw_summary.profile.nunique()} profiles, "
      f"{raw_summary.n_tests.sum()} tests, {raw_summary.measurement_rows.sum():,} raw measurement rows")


# ### 1b. The feature table
# 
# `src/features.py` turns each **reference test** (the periodic health check-up) into one row. Reference tests are run at
# 5, 15, 25, 35 and 45 °C; SOH is only defined on the 25 °C ones, because capacity changes with test temperature.

# In[4]:


df_all = pd.read_csv(ROOT / "results" / "rtp_features.csv", dtype={"cell_id": str})
print("feature table (all reference tests):", df_all.shape)
print("reference tests per test temperature:", df_all.rtp_T_set.value_counts().sort_index().to_dict())

df = df_all[(df_all.rtp_T_set == 25) & df_all.SOH.notna()].reset_index(drop=True)
print("\nlabelled table (25 °C reference tests with SOH):", df.shape)
print(f"{df.cell_id.nunique()} cells | {df.profile.nunique()} profiles")
print("column types:", df.dtypes.astype(str).replace({"str": "text", "object": "text"}).value_counts().to_dict())
df.head()


# ## 2. Data dictionary
# 
# Every column gets a **role**:
# 
# * **id**: identifies the row (never a model input; `cell_id` and `profile` are used to make the splits)
# * **target**: what we predict
# * **leaky**: is the target or is measured from the same full discharge as the target, so never an input
# * **comparison**: measured inside the target test; only used in the leakage comparison, not the main model
# * **input**: allowed in the main model

# In[5]:


D = {}   # column -> (group, meaning, unit, role)
def add(cols, group, meaning, unit, role):
    for c in cols if isinstance(cols, list) else [cols]:
        D[c] = (group, meaning, unit, role)

add("cell_id", "Identifier", "Battery cell ID (one HDF5 file)", "-", "id")
add("profile", "Identifier", "Ageing profile the cell was cycled under", "-", "id")
add("idtest", "Identifier", "Test number inside the cell's file", "-", "id")
add("expname", "Identifier", "Test name, e.g. RTP25 = reference test at 25 °C", "-", "id")
add("rtp_T_set", "Identifier", "Temperature of the reference test", "°C", "id")
add("schedule", "Test setting", "Check-up procedure: standard (RTP) or high-current (RTP_highCurrent)", "category", "input")
add("SOH", "Target", "State of health = capacity / the cell's first 25 °C capacity", "%", "target")
add(["cap_1C_Ah", "cap_ref_Ah"], "Capacity (leaky)", "Measured capacity / the cell's reference capacity (SOH's numerator / denominator)", "Ah", "leaky")
add("cap_1C_spread_Ah", "Capacity (leaky)", "Difference between the two full discharges in one check-up", "Ah", "leaky")
add("dis1C_energy_Wh", "Capacity (leaky)", "Energy of the full discharge (≈ capacity × voltage)", "Wh", "leaky")
add(["chg_CC_Ah", "chg_CV_Ah", "chg_CC_time_s", "chg_CC_fraction", "dis_C3_CC_Ah"], "Capacity measure",
    "Charge / time of a charge or C/3 discharge step; capacity in disguise", "Ah / s / ratio", "comparison")
add(["chg_CC_V0", "chg_CV_time_s"], "Recharge between the target discharges",
    "Voltage at the start of / time of the recharge right after a full discharge", "V / s", "comparison")
add(["dis1C_meanV", "dis1C_dT", "dis1C_Tstart", "dis1C_IRdrop_ohm", "dis1C_Q_3.8_3.5V", "dis1C_t_to_3.6V_s",
     "relax_dV_after_dis", "rtp_T_mean"], "Inside the target discharge",
    "Shape / temperature / partial window of the full 1C discharge", "V / °C / Ω / Ah / s", "comparison")
add([c for c in df.columns if c.startswith("ica_")], "Inside the target discharge",
    "Incremental-capacity (dQ/dV) peak height / position or area on the 1C discharge", "Ah/V / V / Ah", "comparison")
add([c for c in df.columns if c.startswith("R_") and not c.endswith("_n")], "Pulse resistance",
    "Internal resistance from 10 s current pulses (rises as the cell ages)", "Ω", "input")
add([c for c in df.columns if c.startswith("R_") and c.endswith("_n")], "Pulse resistance",
    "Number of pulses of that size found in the test", "count", "input")
add(["days_since_start", "rtp_index"], "Usage history", "Time since the first test / number of 25 °C check-ups so far", "days / count", "input")
add(["cum_dis_Ah", "cum_chg_Ah", "cum_EFC"], "Usage history", "Total charge moved so far (EFC = equivalent full cycles)", "Ah / cycles", "input")
add(["cum_lp_days", "cum_lp_Tdays", "cum_lp_T_mean"], "Usage history", "Time spent in ageing periods, and their mean temperature", "days / °C", "input")
add([c for c in df.columns if c.startswith("prev_lp_")], "Previous ageing period",
    "Measured conditions of the ageing period just before this check-up (temperature, voltage, current, rest share)", "mixed", "input")
add([c for c in df.columns if c.startswith("lp_")], "Profile design",
    "Designed setting of the profile (T_set, charge/discharge C-rate, frequency, SoC, ΔSoC); from the dataset paper", "mixed", "input")

missing_doc = set(df.columns) - set(D)
assert not missing_doc, f"undocumented columns: {missing_doc}"
dictionary = (pd.DataFrame.from_dict(D, orient="index", columns=["group", "meaning", "unit", "role"])
                .reindex(df.columns).rename_axis("column"))
dictionary["dtype"] = df.dtypes.astype(str).replace({"str": "text", "object": "text"})
dictionary.to_csv(OUT / "data_dictionary.csv")
print(dictionary.role.value_counts().to_string())
dictionary


# ## 3. Missing values
# 
# Inspection only; nothing is filled in here. Filling happens in Section 7, **after** the split.

# In[6]:


na = df.isna().sum()
miss = (pd.DataFrame({"n_missing": na, "pct": (100 * na / len(df)).round(1),
                      "cells_affected": [df.loc[df[c].isna(), "cell_id"].nunique() for c in df.columns],
                      "role": dictionary.role})
          .query("n_missing > 0").sort_values("n_missing", ascending=False))
print(f"{len(miss)} of {df.shape[1]} columns have missing values; {df.isna().any(axis=1).sum()} of {len(df)} rows have at least one")
miss


# In[7]:


# Where do the gaps come from?
first_rtp = df.groupby("cell_id").rtp_index.transform("min") == df.rtp_index
print("prev_lp_* missing rows that are the cell's FIRST 25 °C check-up:",
      int((df.prev_lp_T_mean.isna() & first_rtp).sum()), "of", int(df.prev_lp_T_mean.isna().sum()))
print("lp_* (profile design) missing for profiles:", df.loc[df.lp_T_set.isna(), "profile"].unique().tolist())
print("R_0p18C_* missing rows:", df.loc[df.R_0p18C_10s_mean.isna(), ["cell_id", "idtest"]].values.tolist())
print("chg_* missing rows:", df.loc[df.chg_CC_Ah.isna(), ["cell_id", "idtest"]].values.tolist())

fig, ax = plt.subplots(figsize=(11, 3.8))
m = df.set_index(["cell_id", "idtest"])[miss.index].isna()
ax.imshow(m.T.values, aspect="auto", cmap="Greys", interpolation="nearest")
ax.set_yticks(range(len(miss))); ax.set_yticklabels(miss.index, fontsize=7)
ax.set_xlabel("rows (sorted by cell)"); ax.set_title("Missing values (black = missing)")
plt.tight_layout(); plt.show()


# **Reading the result**
# 
# | Columns | Why missing | Plan (Section 7) |
# |---|---|---|
# | `prev_lp_*`, `cum_lp_T_mean` | The first check-up of a cell has no ageing period before it | Median imputation (from training rows) |
# | `lp_*` (profile design) | Profile **V212**'s row in the dataset paper is malformed, so its settings were not entered | Median imputation; flagged as a limitation |
# | `R_0p18C_*`, `relax_dV_after_dis` | That pulse / rest step was not found in a few tests | Median imputation |
# | `chg_*` | One test without the C/2 recharge step | Not used in the main model anyway |
# 
# No column is mostly missing, so none is dropped for missingness alone.

# ## 4. Outliers and physical-range checks
# 
# Inspection only. In battery data, an "extreme" value is often **real ageing** (e.g. a cell at 33 % SOH), so the rule here is:
# **flag, look, and only remove rows that are physically impossible**. Model-side protection against extreme values
# (clipping) is learned in Section 7 from the training rows.
# 
# ### 4a. Summary statistics

# In[8]:


num = df.select_dtypes("number").columns.drop(["idtest", "rtp_T_set"])
desc = df[num].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
q1, q3 = df[num].quantile(0.25), df[num].quantile(0.75)
iqr = q3 - q1
desc["n_IQR_outliers"] = ((df[num] < q1 - 1.5 * iqr) | (df[num] > q3 + 1.5 * iqr)).sum()
desc["role"] = dictionary.role
desc.round(4).sort_values("n_IQR_outliers", ascending=False)


# The IQR rule (outside 1.5 × the middle 50 %) flags many points, mostly in resistance and usage-history columns. That is
# expected: a few cells age much faster than the rest, so their resistance and SOH sit far from the bulk. These are the cells
# the project is about, so they must stay.
# 
# ### 4b. Physical-range checks (would mean a broken measurement)

# In[9]:


checks = {
    "SOH outside 20–105 %": ~df.SOH.between(20, 105),
    "capacity outside 0.5–3.6 Ah": ~df.cap_1C_Ah.between(0.5, 3.6),
    "resistance <= 0 or > 1 Ω": (df.filter(regex=r"^R_.*(mean|soc55|lowSOC)$") .le(0) | df.filter(regex=r"^R_.*(mean|soc55|lowSOC)$").gt(1)).any(axis=1),
    "ageing-period voltage outside 2.0–4.3 V": ~df.prev_lp_V_min.between(2.0, 4.3) & df.prev_lp_V_min.notna()
                                               | ~df.prev_lp_V_max.between(2.0, 4.3) & df.prev_lp_V_max.notna(),
    "temperature outside −20–70 °C": ~df.rtp_T_mean.between(-20, 70) | (df.prev_lp_T_max > 70),
    "negative throughput or time": (df[["cum_dis_Ah", "cum_chg_Ah", "days_since_start"]] < 0).any(axis=1),
}
phys = pd.DataFrame({k: [int(v.sum())] for k, v in checks.items()}, index=["rows failing"]).T
phys


# ### 4c. Within-cell checks: SOH jumps and disagreeing discharges
# 
# SOH should mostly go **down** over time. A large upward jump, or a big disagreement between the two full discharges inside
# one check-up (`cap_1C_spread_Ah`), points to a noisy measurement.

# In[10]:


d = df.sort_values(["cell_id", "rtp_index"]).copy()
d["SOH_change"] = d.groupby("cell_id").SOH.diff()
d["spread_pct"] = 100 * d.cap_1C_spread_Ah / d.cap_1C_Ah
flag = d[(d.SOH_change > 3) | (d.spread_pct > 3)]
print(f"{len(flag)} rows flagged (SOH jumps up by > 3 points, or the two discharges differ by > 3 %):")
display(flag[["cell_id", "profile", "idtest", "rtp_index", "SOH", "SOH_change", "spread_pct"]].round(2))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for cid, g in d.groupby("cell_id"):
    axes[0].plot(g.days_since_start, g.SOH, lw=1, alpha=0.7)
axes[0].scatter(flag.days_since_start, flag.SOH, s=60, facecolors="none", edgecolors="red", label="flagged")
axes[0].set(xlabel="days since start", ylabel="SOH (%)", title="SOH per cell, flagged points circled"); axes[0].legend()
axes[1].hist(d.spread_pct, bins=40, color="grey"); axes[1].axvline(3, color="red", ls="--")
axes[1].set(xlabel="difference between the two full discharges (% of capacity)", ylabel="rows",
            title="Agreement inside one check-up")
plt.tight_layout(); plt.show()


# **Decision:** no row is physically impossible, so **no rows are removed**. The flagged points are small capacity
# recoveries after rest periods and a few noisy check-ups. They are real measurements, and the report should mention them
# as label noise that no model can predict. Extreme *input* values are handled by clipping in Section 7.

# ## 5. Target and input columns
# 
# This step uses **fixed rules** (the roles from the data dictionary), not values learned from the data, so it is safe to do
# before the split.
# 
# * **Regression target:** `SOH`
# * **Classification target:** `health_class`, made from SOH with project-defined cut-offs: Healthy ≥ 90 %,
#   Degraded 80–90 %, EOL-risk < 80 % (80 % is the common automotive end-of-life point)
# * **Main inputs:** columns with role *input*: usage history, previous ageing period, profile design, pulse resistance and
#   the categorical `schedule`
# * **Not inputs:** ids, target, leaky and comparison columns
# 
# **Why `cell_id` and `profile` are not one-hot encoded.** They are the labels we hold out. In a leave-cell-out test, the
# test cell's ID never appears in training, so its one-hot column would be all zeros: the model learns nothing it can use and
# could memorise individual cells instead. The *conditions* of a profile are already given as numbers (`lp_*`, `prev_lp_*`),
# and those do carry over to an unseen profile.

# In[11]:


TARGET = "SOH"
df["health_class"] = pd.cut(df.SOH, [-np.inf, 80, 90, np.inf], right=False,
                            labels=["EOL-risk (<80)", "Degraded (80-90)", "Healthy (>=90)"])

CAT_FEATURES = ["schedule"]
NUM_FEATURES = [c for c in dictionary.index[dictionary.role == "input"] if c not in CAT_FEATURES]
print(f"{len(NUM_FEATURES)} numeric inputs + {len(CAT_FEATURES)} categorical input")
print("by group:", dictionary.loc[NUM_FEATURES, "group"].value_counts().to_dict())
print("\nhealth classes (all labelled rows):", df.health_class.value_counts().reindex(df.health_class.cat.categories).to_dict())
print("schedule values:", df.schedule.value_counts().to_dict())


# ## 6. Split into train / validation / test
# 
# ### Why split by cell, not by row
# Neighbouring check-ups of the same cell are almost identical. With a random row split, the model sees check-up 7 of a cell
# in training and is tested on check-up 8 of the same cell, which is close to copying the answer. So whole **cells** go to
# one side only.
# 
# ### The split
# 1. **Test:** 3 cells (20 %), held out first and not touched until the final evaluation.
# 2. **Validation:** 3 of the remaining cells, used to compare models and tune settings.
# 3. **Train:** the other 9 cells.
# 
# This single split is the example used to show the preprocessing. The final model results (notebooks 03–04) repeat the same
# preprocessing inside every fold of leave-one-cell-out and leave-one-profile-out cross-validation, so every cell gets a turn
# as the test cell. With only 15 cells, one fixed test set of 3 cells would be too small to judge models on its own.

# In[12]:


groups = df.cell_id.values
outer = GroupShuffleSplit(n_splits=1, test_size=3, random_state=SEED)
trval_idx, test_idx = next(outer.split(df, groups=groups))
inner = GroupShuffleSplit(n_splits=1, test_size=3, random_state=SEED)
tr_rel, val_rel = next(inner.split(df.iloc[trval_idx], groups=groups[trval_idx]))
train_idx, val_idx = trval_idx[tr_rel], trval_idx[val_rel]

df["split"] = ""
df.loc[train_idx, "split"], df.loc[val_idx, "split"], df.loc[test_idx, "split"] = "train", "validation", "test"
split_tbl = (df.groupby("split")
               .agg(rows=("SOH", "size"), cells=("cell_id", "nunique"), profiles=("profile", "nunique"),
                    SOH_min=("SOH", "min"), SOH_mean=("SOH", "mean"))
               .reindex(["train", "validation", "test"]).round(1))
split_tbl = split_tbl.join(pd.crosstab(df.split, df.health_class))
display(split_tbl)
for s in ["train", "validation", "test"]:
    sub = df[df.split == s]
    print(f"{s:10s} cells: {sorted(sub.cell_id.unique())}  profiles: {sorted(sub.profile.unique())}")

# no cell may appear in two parts
sets = {s: set(df.loc[df.split == s, "cell_id"]) for s in ["train", "validation", "test"]}
assert not (sets["train"] & sets["validation"]) and not (sets["train"] & sets["test"]) and not (sets["validation"] & sets["test"])
print("\nOK: no cell appears in more than one part")


# In[13]:


# For comparison: what a random ROW split would have done
rng = np.random.default_rng(SEED)
rand_test = rng.random(len(df)) < 0.2
shared = set(df.cell_id[rand_test]) & set(df.cell_id[~rand_test])
print(f"random 80/20 row split: {len(shared)} of 15 cells have rows in BOTH train and test -> optimistic results")


# **Things to keep in mind about this split**
# * Some profiles have only one cell, so a held-out cell usually also means a held-out profile.
# * The EOL-risk class comes from only 8 cells, so a 3-cell validation or test set may contain few or none of them.
#   That's another reason the final results use cross-validation over all cells.

# ## 7. Preprocessing: fitted on the training rows only
# 
# Each step is a scikit-learn transformer. `fit` learns its numbers from **train**; `transform` applies them to train,
# validation and test.
# 
# | Step | Columns | Learns from train | Why |
# |---|---|---|---|
# | 1. Drop unusable columns | numeric | which columns are entirely missing or constant in train | A column with no information in train can't be learned from |
# | 2. Clip outliers | numeric | 1st and 99th percentile of each column | Stops one extreme value from dominating linear models and SVR; values are capped, not removed |
# | 3. Impute missing values | numeric | median of each column | Models can't take blanks; the median isn't pulled by outliers |
# | 4. Scale | numeric | mean and standard deviation | Puts resistances (≈0.05 Ω) and throughput (≈1000 Ah) on the same scale for Ridge / SVR |
# | 5. Impute missing values | categorical | most frequent value | (no gaps today, but safe) |
# | 6. One-hot encode | categorical | list of categories | Turns `schedule` into 0/1 columns; a category never seen in train becomes all zeros |
# 
# Tree models (random forest, gradient boosting) don't need step 4, but it doesn't hurt them.

# In[14]:


class DropUnusable(BaseEstimator, TransformerMixin):
    # Drop columns that are all-missing or constant in the TRAINING rows.
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.keep_ = [c for c in X.columns if X[c].notna().any() and X[c].nunique(dropna=True) > 1]
        self.dropped_ = [c for c in X.columns if c not in self.keep_]
        return self
    def transform(self, X):
        return pd.DataFrame(X)[self.keep_]
    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.keep_, dtype=object)

class PercentileClipper(BaseEstimator, TransformerMixin):
    # Cap each column at percentiles learned from the TRAINING rows (missing values are left missing).
    def __init__(self, lower=1, upper=99):
        self.lower, self.upper = lower, upper
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.low_ = X.quantile(self.lower / 100)
        self.high_ = X.quantile(self.upper / 100)
        return self
    def transform(self, X):
        return pd.DataFrame(X).clip(self.low_, self.high_, axis=1)
    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.low_.index, dtype=object)

numeric_pipe = Pipeline([
    ("drop_unusable", DropUnusable()),
    ("clip", PercentileClipper(1, 99)),
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocess = ColumnTransformer([("num", numeric_pipe, NUM_FEATURES),
                                ("cat", categorical_pipe, CAT_FEATURES)],
                               verbose_feature_names_out=False).set_output(transform="pandas")

X = df[NUM_FEATURES + CAT_FEATURES]
y = df[TARGET].values
X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

preprocess.fit(X_train)                       # <- the ONLY fit: training rows
Xp_train = preprocess.transform(X_train)
Xp_val = preprocess.transform(X_val)
Xp_test = preprocess.transform(X_test)
print("before:", X_train.shape, X_val.shape, X_test.shape)
print("after: ", Xp_train.shape, Xp_val.shape, Xp_test.shape)


# ### 7a. What each step learned from the training rows

# In[15]:


num = preprocess.named_transformers_["num"]
drop, clip, imp, sc = (num.named_steps[k] for k in ["drop_unusable", "clip", "impute", "scale"])
print("1. dropped (all-missing or constant in train):", drop.dropped_ or "none")

learned = pd.DataFrame({"clip_low (p1)": clip.low_, "clip_high (p99)": clip.high_,
                        "impute_median": imp.statistics_, "scale_mean": sc.mean_, "scale_std": sc.scale_},
                       index=drop.keep_)
learned["missing_in_train"] = X_train[drop.keep_].isna().sum()
display(learned.round(4))

ohe = preprocess.named_transformers_["cat"].named_steps["onehot"]
print("6. one-hot categories learned:", dict(zip(CAT_FEATURES, [list(c) for c in ohe.categories_])))


# In[16]:


# How much did clipping change? (counted on the raw values, per part)
def n_clipped(Xpart):
    v = Xpart[drop.keep_]
    return int(((v < clip.low_) | (v > clip.high_)).sum().sum())
print({name: n_clipped(part) for name, part in [("train", X_train), ("validation", X_val), ("test", X_test)]},
      "values capped")

# One-hot result
print("\none-hot columns:", [c for c in Xp_train.columns if c.startswith("schedule")])
display(pd.concat([X_train.schedule.rename("schedule (raw)"),
                   Xp_train.filter(like="schedule_")], axis=1).drop_duplicates())


# **Note on clipping.** Far more values are capped in validation and test than in train. That isn't an error: the held-out
# cells come from profiles whose conditions (temperature, current, ageing speed) partly lie **outside** the training range,
# and clipping pulls them back to the edge of what the model has seen. This makes linear models and SVR safer on extreme
# inputs, but it also means the model can't extrapolate beyond the training range. Keep this in mind for the
# interpolation vs extrapolation analysis, and check on the validation cells whether clipping helps or hurts
# (set `PercentileClipper(0, 100)` to switch it off).

# ### 7b. Why fitting on train only matters: a small demonstration
# 
# If the imputer and scaler were fitted on **all** rows, the medians and means would partly come from the validation and
# test cells. The table below compares the two. The differences are large (temperature-related columns move by more than
# 60 %), because the held-out cells were aged under different conditions from the training cells. Fitting on all rows would
# quietly tell the model what the "unseen" conditions look like, which is exactly the leak we are trying to avoid.

# In[17]:


leaky_scaler = Pipeline([("drop", DropUnusable()), ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
leaky_scaler.fit(X[NUM_FEATURES])                         # WRONG: includes validation + test cells
k = leaky_scaler.named_steps["drop"].keep_
cmp = pd.DataFrame({"mean (train only)": pd.Series(sc.mean_, drop.keep_),
                    "mean (all rows: leaky)": pd.Series(leaky_scaler.named_steps["scale"].mean_, k)}).dropna()
cmp["rel_diff_%"] = 100 * (cmp.iloc[:, 1] - cmp.iloc[:, 0]).abs() / cmp.iloc[:, 0].abs().replace(0, np.nan)
cmp.sort_values("rel_diff_%", ascending=False).head(8).round(3)


# ## 8. Final checks and saved outputs

# In[18]:


for name, part in [("train", Xp_train), ("validation", Xp_val), ("test", Xp_test)]:
    assert part.isna().sum().sum() == 0, f"{name} still has missing values"
    assert list(part.columns) == list(Xp_train.columns)
print("OK: no missing values left, and all three parts have the same", Xp_train.shape[1], "columns")
stds = Xp_train[drop.keep_].std(ddof=0)
print("train columns after scaling: largest |mean| =", round(float(Xp_train[drop.keep_].mean().abs().max()), 6),
      "| median std =", round(float(stds.median()), 3))
print("columns that became constant after clipping (std 0, carry no information):", stds[stds < 1e-9].index.tolist() or "none")
print("validation / test are NOT exactly mean 0 / std 1 (as expected, they were only transformed):",
      round(float(Xp_val[drop.keep_].mean().abs().mean()), 3), "/", round(float(Xp_test[drop.keep_].mean().abs().mean()), 3))


# In[19]:


keep_ids = ["cell_id", "profile", "idtest", "rtp_index"]
for name, idx, Xp in [("train", train_idx, Xp_train), ("validation", val_idx, Xp_val), ("test", test_idx, Xp_test)]:
    out = pd.concat([df.iloc[idx][keep_ids].reset_index(drop=True), Xp.reset_index(drop=True),
                     df.iloc[idx][["SOH", "health_class"]].reset_index(drop=True)], axis=1)
    out.to_csv(OUT / f"{name}.csv", index=False)
df[keep_ids + ["split"]].to_csv(OUT / "split_assignment.csv", index=False)
learned.to_csv(OUT / "fitted_preprocessing_parameters.csv")
pd.Series(NUM_FEATURES + CAT_FEATURES, name="feature").to_csv(OUT / "main_feature_list.csv", index=False)
print("saved to", OUT.relative_to(ROOT), ":", sorted(p.name for p in OUT.iterdir()))


# ## Summary
# 
# * **Raw data:** 15 cells / 14 profiles, turned by `src/features.py` into 422 reference-test rows, of which **306** are
#   25 °C check-ups with an SOH label.
# * **Columns:** 75, each documented with a role (`data_dictionary.csv`). The main model uses the *input* columns only;
#   capacity and same-discharge measurements are excluded to avoid target leakage.
# * **Missing values:** explained by the data (first check-up has no previous ageing period, V212 has no design
#   settings, a few missing pulse steps), filled with **training medians**.
# * **Outliers:** no physically impossible values, so no rows are removed. A few noisy check-ups are flagged as label
#   noise; extreme inputs are capped at the **training** 1st / 99th percentiles.
# * **Encoding:** `schedule` is one-hot encoded. `cell_id` / `profile` are not encoded: they define the splits, and an
#   unseen cell or profile would have no matching column.
# * **Split:** by cell into train (9) / validation (3) / test (3); no cell appears in two parts. Every preprocessing step is
#   fitted on train only. The final evaluation repeats this inside leave-cell-out and leave-profile-out cross-validation.
