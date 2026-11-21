# How Reliable Are Supervised ML Models for Lithium-Ion Battery Health Prediction Under Unseen Operating Conditions?

DATA 245 Machine Learning, Fall 2026, San José State University. Group 09:
Kazi Samina Maraj Mumu, Sadaf Fatima Syeda, Kshitija Shinde, Shraddhaben Patel, Rachandeep Kaur.
Instructor: Dr. Vishnu S. Pendyala.

Public GitHub repository: https://github.com/rachandeepkaur/battery-soh-prediction

## Project purpose

We predict the State of Health (SOH) of Panasonic NCR18650B lithium-ion cells and ask how reliable supervised
models are on batteries and operating profiles they have **never seen**. Every model is tested with the same
protocol: leave-one-cell-out (main test), leave-one-profile-out, and random 5-fold (reference only).
We compare 11 regression models and several classifiers. The final model is ElasticNet
(4.25 SOH points of error on unseen cells).

SOH (%) = 1C discharge capacity in a 25 °C reference test ÷ the same cell's first 25 °C capacity × 100.

## Raw data (not included)

The raw HDF5 files are **not** in this repository (`data/raw/` is git-ignored). The full public dataset is about 27 GB:
"Ageing Data of Lithium-ion Panasonic NCR18650B Battery Cells", Zenodo 7896429,
https://zenodo.org/records/7896429. We used 15 of its 116 cells.

To rebuild everything from scratch, download the 15 cells into `data/raw/` (for example `data/raw/2_009.h5`)
and run `python -m src.features data/raw results/rtp_features.csv`.
This is only needed for notebooks 01 and 02, which read the raw files for the pilot inspection and EDA plots.

## Repository layout

| Folder | Contents |
|---|---|
| `notebooks/` | The 9 active notebooks, each as `.ipynb` (with outputs) and `.py` |
| `src/` | Shared code: `features.py` (raw HDF5 → one row per check-up), `evaluation.py` (shared protocol), `modelling.py` (feature pruner), `classification.py` |
| `tests/` | `test_evaluation.py`: 9 automated checks of the evaluation protocol (run `python -m pytest tests`) |
| `models/` | Saved models (see below) |
| `results/` | The feature table and every result file the notebooks read or write |
| `docs/` | Dataset documentation and the written evaluation protocol |
| `archive/` | Superseded notebooks (one-cell pilot, first modelling notebook); kept for history, not part of the final results |

| Notebook | What it does |
|---|---|
| 01_data_preprocessing | Builds and checks the 15-cell feature table, labels, splits and data dictionary |
| 02_eda_feature_discovery | Exploratory analysis, feature timing and within-cell correlation |
| 04_nonlinear_models | SVR and Gaussian process (with uncertainty) |
| 05_evaluation_protocol | The shared evaluation protocol and leakage checks |
| 06_modelling_random_forest_hgb | Random Forest and HistGradientBoosting |
| 07_baselines | Training mean, single-feature models, MLR, Ridge, Lasso, ElasticNet |
| 08_model_comparison | All models on the same folds, paired tests, error analysis, locked hold-out |
| 09_classification | Health classes (Healthy / Degraded / EOL-risk) |
| 10_final_model_and_demo | Saves the final model and runs the demo |

## How to run the notebooks

1. Install Python 3.11 or newer and the packages:
   `pip install numpy pandas scipy scikit-learn matplotlib seaborn joblib h5py jupyter`
   (results were produced with scikit-learn 1.9.1; other versions can change some numbers slightly).
2. Clone the repository and start Jupyter **inside it** (`jupyter notebook`).
   The notebooks find `src/`, `results/` and `models/` whether they are run from `notebooks/` or from the main folder.
3. Run the notebooks in number order. Notebooks 04 to 10 work **without the raw data**: they use the
   included `results/rtp_features.csv`, `results/preprocessed/` and the other `results/` folders.
   Notebook 08 reads the saved predictions of notebooks 04, 06 and 07, and notebook 09 reads those of 04, 06 and 07.

## Who did what

| Member | Main work | Notebooks / code | Pull requests |
|---|---|---|---|
| Rachandeep Kaur | Feature extraction for 15 cells, preprocessing, shared evaluation protocol, tests, model comparison | 01, 05, 08; `src/features.py`, `src/evaluation.py`, `src/modelling.py`, `tests/` | commits to `main` |
| Kshitija Shinde | EDA, feature-timing audit, Random Forest, HistGradientBoosting | 02, 06 | #2 |
| Sadaf Fatima Syeda | SVR and Gaussian process, dataset documentation, literature survey | 04; `docs/dataset_documentation.md` | #4, #6 |
| Shraddhaben Patel | Health classes and classification models | 09; `src/classification.py` | #7 |
| Kazi Samina Maraj Mumu | Baselines and linear models, saved final model and demo, coordination | 07, 10; `models/` | #1, #3, #5, #8–#11 |

Roles follow the CRediT taxonomy; see the slides and reports for the full statement.

## Saved model and demo

* `models/elasticnet_final.joblib` (15.5 KB): the final ElasticNet pipeline trained on all 15 cells
  (alpha 0.1, l1_ratio 0.8, 25 kept inputs). Details in `models/elasticnet_final_info.json`.
* `models/elasticnet_demo_12cells.joblib`: the same pipeline trained on 12 cells, used for the demo.
* Demo: open `notebooks/10_final_model_and_demo.ipynb` and run all cells (under a minute). It loads the demo model,
  predicts the 3 locked cells it never saw (2_009, 2_092, 2_098) and reports MAE 5.27, matching notebook 08.
  Figures are written to `results/final_model/figures/`.

To load a saved model in your own code, run Python from this folder (so `src` can be imported) and use
`joblib.load("models/elasticnet_final.joblib")`.
