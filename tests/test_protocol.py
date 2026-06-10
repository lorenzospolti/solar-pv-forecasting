"""Regression tests for the protocol fixes of 10 Jun 2026 (see CRITICAL_REVIEW.md).

A1  Open-Meteo backward-averaged radiation realigned to hour-beginning labels.
A2  Day blocks / issue boundaries defined on LOCAL calendar days (day_tz).
A3  Clear-sky template fit on an initial window only (no test-period leakage).
A5  Per-quantile coverage columns emitted for the reliability diagram.
B5  Models never see target-derived columns (power, k) for the horizon.
B6  Daytime data gaps stay NaN in k (not fake zeros).
B4  One fit per fold scores both daytime/all-hours masks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pvfc.data import synthetic_station
from pvfc.models.baselines import Persistence
from pvfc.normalize import clearsky_index
from pvfc.runner import run_model
from pvfc.splits import ExpandingWindowSplitter

HKT = "Asia/Hong_Kong"


# --------------------------------------------------------------------- A2 ---
def test_local_day_blocks_are_whole_hkt_days():
    idx = pd.date_range("2021-01-01", periods=400 * 24, freq="1h", tz="UTC")
    sp = ExpandingWindowSplitter(initial_train_days=300, test_days=10,
                                 embargo_days=1, max_folds=2, day_tz=HKT)
    folds = list(sp.split(idx))
    assert folds, "expected folds"
    for fold in folds:
        local = fold.test_idx.tz_convert(HKT)
        # every local day in the test block is complete (24 hourly stamps)
        assert (pd.Series(1, index=local).groupby(local.normalize()).sum() == 24).all()
        # train ends before test starts, with the embargo day in between
        train_days = set(fold.train_idx.tz_convert(HKT).normalize())
        test_days = set(local.normalize())
        assert max(train_days) < min(test_days) - pd.Timedelta(days=sp.embargo_days - 1)


def test_utc_vs_local_blocks_differ():
    idx = pd.date_range("2021-01-01", periods=320 * 24, freq="1h", tz="UTC")
    kw = dict(initial_train_days=300, test_days=5, embargo_days=1, max_folds=1)
    f_utc = next(ExpandingWindowSplitter(**kw).split(idx))
    f_hkt = next(ExpandingWindowSplitter(**kw, day_tz=HKT).split(idx))
    assert not f_utc.test_idx.equals(f_hkt.test_idx)  # 8-hour offset shifts blocks


# --------------------------------------------------------------------- A3 ---
def test_clearsky_template_ignores_future_data():
    from pvfc import clearsky as cs

    s = synthetic_station(days=540, seed=3)
    power = s.frame["power"].copy()
    ghi_clear = cs.clearsky_ghi(power.index, s.lat, s.lon, s.altitude)
    fit_mask = pd.Series(power.index < power.index.min() + pd.Timedelta(days=365),
                         index=power.index)
    base = cs.empirical_clearsky_power(power, ghi_clear, capacity=s.capacity,
                                       fit_mask=fit_mask)
    corrupted = power.copy()
    corrupted.iloc[-90 * 24:] *= 7.0  # wild change AFTER the fit window
    redo = cs.empirical_clearsky_power(corrupted, ghi_clear, capacity=s.capacity,
                                       fit_mask=fit_mask)
    pd.testing.assert_series_equal(base, redo)  # template must be unaffected


# --------------------------------------------------------------------- B6 ---
def test_daytime_gaps_stay_nan_nights_zero():
    p_clear = pd.Series([0.0, 5.0, 5.0, 0.0], dtype=float)
    power = pd.Series([0.0, np.nan, 4.0, np.nan], dtype=float)
    k = clearsky_index(power, p_clear)
    assert k.iloc[0] == 0.0          # night -> 0
    assert np.isnan(k.iloc[1])       # daytime gap -> NaN (not a fake zero)
    assert k.iloc[2] == 0.8
    assert k.iloc[3] == 0.0          # night gap -> 0 (irrelevant for fitting)


# ---------------------------------------------------------------- B5 / B4 ---
class _ProbeModel(Persistence):
    """Persistence that asserts the leakage guard and counts fit calls."""

    name = "probe"

    def __init__(self):
        self.n_fit = 0

    def fit(self, train):
        self.n_fit += 1
        return self

    def predict(self, history, future):
        assert "power" not in future.columns and "k" not in future.columns, \
            "future must not carry target-derived columns"
        assert "p_clear" in future.columns  # legitimately known ahead
        return super().predict(history, future)


def test_runner_strips_target_one_fit_two_masks_and_coverage():
    series = synthetic_station(days=360, seed=4)
    sp = ExpandingWindowSplitter(initial_train_days=300, test_days=20,
                                 embargo_days=1, max_folds=2, day_tz=HKT)
    probe = _ProbeModel()
    rows = run_model(probe, series, sp)
    n_folds = sp.n_splits(series.frame.index)
    assert probe.n_fit == n_folds                      # B4: one fit per fold
    assert len(rows) == 2 * n_folds                    # both masks scored
    per_fold = {r["fold"] for r in rows}
    assert all({r["daytime_only"] for r in rows if r["fold"] == f} == {True, False}
               for f in per_fold)
    assert {"fit_s", "predict_s", "fit_predict_s"} <= set(rows[0])


def test_quantile_coverage_columns_emitted():
    from pvfc.evaluation import evaluate

    rng = np.random.default_rng(0)
    y = rng.uniform(0, 10, 500)
    q_pred = {tau: np.quantile(y, tau) * np.ones_like(y)
              for tau in (0.1, 0.5, 0.9)}
    row = evaluate(y, y, capacity=10.0, q_pred=q_pred).as_row()
    for tau in (0.1, 0.5, 0.9):
        assert abs(row[f"cov_q{tau}"] - tau) < 0.05    # A5: reliability data present


# --------------------------------------------------------------------- A1 ---
def test_open_meteo_radiation_shift():
    from pvfc.weather import _align_om_conventions

    idx = pd.date_range("2022-01-01", periods=6, freq="1h", tz="UTC")
    df = pd.DataFrame({"ghi": np.arange(6.0), "temp_air": np.arange(6.0)}, index=idx)
    out = _align_om_conventions(df)
    assert out["ghi"].iloc[0] == 1.0          # backward-averaged -> shifted -1 h
    assert out["temp_air"].iloc[0] == 0.0     # instantaneous -> untouched
