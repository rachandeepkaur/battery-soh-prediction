"""Leakage and consistency checks for the shared evaluation protocol (run: python -m pytest tests -q)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from src import evaluation as ev


@pytest.fixture(scope="module")
def data():
    df = ev.load_labelled()
    num, cat = ev.main_features()
    return df, num, cat


def test_dataset_shape(data):
    df, num, cat = data
    assert len(df) == 306 and df.cell_id.nunique() == 15 and df.profile.nunique() == 14
    assert len(num) == 42 and cat == ["schedule"]


def test_no_leaky_or_comparison_inputs(data):
    _, num, cat = data
    dd = pd.read_csv(ev.PREPROCESSED / "data_dictionary.csv").set_index("column")
    assert set(dd.loc[num + cat, "role"]) == {"input"}
    for bad in ["cap_1C_Ah", "chg_CC_V0", "chg_CV_time_s", "chg_CC_Ah", "SOH"]:
        assert bad not in num


@pytest.mark.parametrize("scheme", ["LOCO", "LOPO"])
def test_outer_folds_are_group_disjoint_and_cover_every_row(data, scheme):
    df, _, _ = data
    col = ev.SCHEMES[scheme]
    seen = np.zeros(len(df), int)
    for tr, te in ev.outer_folds(df, scheme):
        assert not set(df[col].iloc[tr]) & set(df[col].iloc[te])
        seen[te] += 1
    assert (seen == 1).all()


def test_fold_assignment_is_deterministic(data):
    df, _, _ = data
    pd.testing.assert_frame_equal(ev.fold_assignment(df), ev.fold_assignment(df))


def test_preprocessor_learns_only_from_training_rows(data):
    """Changing the held-out rows must not change anything the preprocessor learned."""
    df, num, cat = data
    cols = num + cat + ["cell_id"]
    tr, te = ev.outer_folds(df, "LOCO")[0]
    a = ev.make_preprocessor(num, cat).fit(df.iloc[tr][cols], df.SOH.iloc[tr])

    poisoned = df.copy()
    poisoned.loc[poisoned.index[te], num] = 1e6          # absurd test values
    poisoned.loc[poisoned.index[te], "SOH"] = -1e6
    b = ev.make_preprocessor(num, cat).fit(poisoned.iloc[tr][cols], poisoned.SOH.iloc[tr])

    assert a.named_steps["prune"].selected_ == b.named_steps["prune"].selected_
    pd.testing.assert_frame_equal(a.transform(df.iloc[tr][cols]), b.transform(df.iloc[tr][cols]))


def test_test_values_are_clipped_to_training_range(data):
    df, num, cat = data
    cols = num + cat + ["cell_id"]
    tr, te = ev.outer_folds(df, "LOCO")[0]
    pre = ev.make_preprocessor(num, cat).fit(df.iloc[tr][cols], df.SOH.iloc[tr])
    extreme = df.iloc[te][cols].copy()
    extreme[num] = 1e9
    out = pre.transform(extreme)
    assert np.isfinite(out.to_numpy()).all() and out.abs().to_numpy().max() < 1e3


def test_schedule_is_one_binary_column(data):
    df, num, cat = data
    out = ev.make_preprocessor(num, cat).fit_transform(df[num + cat + ["cell_id"]], df.SOH)
    assert [c for c in out.columns if c.startswith("schedule")] == ["schedule_RTP_highCurrent"]
    assert "cell_id" not in out.columns and "profile" not in out.columns


def test_run_cv_predicts_every_row_once(data):
    df, _, _ = data
    r = ev.run_cv(df, ev.ModelSpec("Mean", DummyRegressor()), "LOCO", verbose=False)
    assert r.oof.y_pred.notna().all() and len(r.folds) == 15
    # A mean predictor trained without the held-out cell predicts that cell's training mean, not its own.
    for k, (tr, te) in enumerate(ev.outer_folds(df, "LOCO")):
        assert np.allclose(r.oof.y_pred.iloc[te], df.SOH.iloc[tr].mean())
