# battery-soh-prediction
Supervised machine learning for battery State of Health (SOH) prediction under unseen conditions using Panasonic NCR18650B dataset.

## Repository layout

| Path | What it is |
|---|---|
| `src/features.py` | Raw HDF5 → one row per Reference Test (RTP): SOH target + candidate features |
| `notebooks/02_eda_feature_discovery.ipynb` | EDA for feature selection (leakage groups, within-cell correlation, leave-cell-out screening) |
| `src/modelling.py` | Feature groups, fold-safe redundancy pruner, model grid (random / leave-cell-out / leave-profile-out) |
| `notebooks/03_modelling.ipynb` | Model comparison with and without partial-capacity HIs |
| `notebooks/battery_data_processing_intermediate_2_009.ipynb` | Pilot notebook (superseded, see note below) |
| `results/rtp_features.csv` | Feature table produced by `src/features.py` |
| `results/feature_screening_summary.csv` | Per-feature screening metrics from the EDA notebook |
| `results/model_comparison.csv`, `results/oof_predictions.csv` | Metrics and out-of-fold predictions from notebook 03 |
| `results/figures/` | EDA figures |
| `data/raw/` | Cell `.h5` files from [Zenodo 7896429](https://zenodo.org/records/7896429) (git-ignored) |

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install h5py pandas numpy scipy matplotlib seaborn scikit-learn jupyter
# download cells, e.g.
curl -L -o data/raw/2_009.h5 "https://zenodo.org/records/7896429/files/2_009.h5?download=1"
.venv/bin/python -m src.features data/raw results/rtp_features.csv
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/02_eda_feature_discovery.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 notebooks/03_modelling.ipynb  # ~4 min
```

## Data notes (important)

* `tests/test_XXXX` matches `info.idtest == XXXX`. The `info` rows are **not** stored in test order, so pairing them
  with sorted group names (as the pilot notebook does) mislabels tests.
* SOH = 1C discharge capacity (4.2 → 2.5 V) in an RTP@25 °C ÷ the cell's first RTP@25 °C capacity. RTPs at 5/15/35/45 °C
  give temperature-shifted capacities and are not used as SOH labels.
* `chargecap`/`dischargecap` reset within a test, so capacities are taken as differences inside a single step.
* RTP pulse currents are scaled to the cell's latest measured capacity; resistances are computed as ΔV / actual current.
