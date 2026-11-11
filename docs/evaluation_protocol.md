# Evaluation protocol

Every SOH model in this project (baselines, linear and non-linear) is scored the same way.
The code is in `src/evaluation.py`, the walkthrough is `notebooks/05_evaluation_protocol.ipynb`, and the
leakage checks are `tests/test_evaluation.py` (run `python -m pytest tests -q`).

## Rules

1. **Data:** the 306 labelled rows: 25 °C reference tests with SOH, 15 cells and 14 profiles.
   Load them with `ev.load_labelled()`.
2. **Inputs:** the 42 numeric main inputs plus `schedule` (`ev.main_features()`).
   - Never use the 22 comparison features or the 4 leaky ones as inputs. They are measured inside the test that defines SOH.
   - `chg_CC_V0` and `chg_CV_time_s` are in that group.
3. **Splits:**
   - Leave-one-cell-out (LOCO, 15 folds) is the **headline**.
   - Also report leave-one-profile-out (LOPO, 14 folds).
   - Random 5-fold (seed 42) is a reference only.
   - Folds are fixed in `results/evaluation_protocol/fold_assignment.csv`.
4. **Fit everything inside the fold:** redundancy pruning (|ρ| ≥ 0.95 blocks), 1–99 % clipping, median imputation, scaling and one-hot encoding of `schedule` (`drop="if_binary"`). These run in one pipeline built by `ev.make_pipeline` and are fitted on the training rows of each fold only.
5. **Tuning:**
   - Nested: an inner 5-fold `GroupKFold`, grouped like the outer split.
   - Write the grid in a `ModelSpec` before running, and don't change it after seeing LOCO scores.
6. **Metrics:** use pooled out-of-fold predictions.
   - MAE is the headline; also report RMSE, R² and max error.
   - Break MAE down per cell, per health class (<80, 80–90, ≥90) and by interpolation vs extrapolation. Extrapolation means the held-out profile's design lies outside the training range.
7. **Comparing models:** use the same folds and compare per-fold MAE with a Wilcoxon signed-rank test (`ev.paired_comparison`).
   - A difference only counts if it is consistent across folds, not just a lower pooled MAE.
   - Also report the generalization gap, grouped MAE minus random MAE (`ev.generalization_gap`).
8. **Locked hold-out:** `ev.holdout_check` trains on the 12 train and validation cells from notebook 01 and scores the 3 test cells (2_009, 2_092, 2_098).
   - Run it once per **final** model.
   - Never use it to choose models. It contains no EOL-risk rows.

## Reference numbers (do not beat these by changing the protocol)

| Model | LOCO MAE | LOPO MAE | Random 5-fold MAE |
|---|---|---|---|
| Mean baseline | 8.18 | 8.24 | 7.89 |
| Ridge (smoke test, alpha tuned) | 4.76 | 4.89 | 2.79 |

The Ridge row is a smoke test with the honest inputs. Earlier modelling reported Ridge LOCO ≈ 2.5, which included `chg_CC_V0` and `chg_CV_time_s`. Under this protocol, adding those two features brings Ridge back to about 2.9. So most of that earlier gain came from leakage, not from the model.

## Known limits

- **EOL rows are concentrated.** 7 of the 15 LOCO folds have no EOL-risk rows, so report EOL error on the pooled predictions.
- **Extrapolation cells are hardest.** Four held-out cells need extrapolation: 2_012 (L35), 2_076 (L223), 2_092 (L202) and 2_095 (L206). Their errors are expected to be the largest.
- **L03 has two cells.** That makes LOCO slightly easier than LOPO for that profile.
