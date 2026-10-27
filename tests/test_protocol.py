"""Regression tests for the protocol fixes of 10 Jun 2026 (see CRITICAL_REVIEW.md).

A1  Open-Meteo backward-averaged radiation realigned to hour-beginning labels.
A2  Day blocks / issue boundaries defined on LOCAL calendar days (day_tz).
A3  Clear-sky template fit on an initial window only (no test-period leakage).
A5  Per-quantile coverage columns emitted for the reliability diagram.
B5  Models never see target-derived columns (power, k) for the horizon.
B6  Daytime data gaps stay NaN in k (not fake zeros).
B4  One fit per fold scores both daytime/all-hours masks.

and for the ETL v2 / protocol fixes of 1 Oct 2026:

C1  Clear-sky template binned by sun position: no hole when a month is missing.
C2  Calendar-anchored folds: stations with different starts share test windows.
C3  Persistence fallback uses the same hour on an earlier day, never a night zero.
C4  Hourly files relabelled to interval start; 15-min files untouched.
C5  Power above 1.1 x nameplate is removed before resampling.
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
def _template_inputs(s):
    from pvfc import clearsky as cs

    idx = s.frame.index
    ghi_clear = cs.clearsky_ghi_interval(idx, s.lat, s.lon, s.altitude)
    decl, ha = cs.sun_coordinates(idx + pd.Timedelta(minutes=30), s.lon)
    fit_mask = pd.Series(idx < idx.min() + pd.Timedelta(days=365), index=idx)
    return ghi_clear, decl, ha, fit_mask


def test_clearsky_template_ignores_future_data():
    from pvfc import clearsky as cs

    s = synthetic_station(days=540, seed=3)
    power = s.frame["power"].copy()
    ghi_clear, decl, ha, fit_mask = _template_inputs(s)
    base = cs.empirical_clearsky_power(power, ghi_clear, decl, ha, capacity=s.capacity,
                                       fit_mask=fit_mask)
    corrupted = power.copy()
    corrupted.iloc[-90 * 24:] *= 7.0  # wild change AFTER the fit window
    redo = cs.empirical_clearsky_power(corrupted, ghi_clear, decl, ha, capacity=s.capacity,
                                       fit_mask=fit_mask)
    pd.testing.assert_series_equal(base, redo)  # template must be unaffected


# --------------------------------------------------------------------- C1 ---
def test_template_has_no_hole_when_a_month_is_missing():
    """ETL v1 regression: no February in the fit year gave p_clear = 0 all February."""
    from pvfc import clearsky as cs

    s = synthetic_station(days=540, seed=7)
    power = s.frame["power"].copy()
    power[power.index.month == 2] = np.nan  # a station outage covering February
    ghi_clear, decl, ha, fit_mask = _template_inputs(s)
    p_clear = cs.empirical_clearsky_power(power, ghi_clear, decl, ha, capacity=s.capacity,
                                          fit_mask=fit_mask)
    local = p_clear.tz_convert(HKT)
    daily_max = local.groupby(local.index.date).max()
    complete = local.groupby(local.index.date).size() == 24
    daily_max = daily_max[complete]
    assert (daily_max > 0.4 * s.capacity).all(), "a day without a clear-sky template"
    assert daily_max.pct_change().abs().max() < 0.05, "template steps between days"


# --------------------------------------------------------------------- C2 ---
def test_first_test_day_aligns_stations_with_different_starts():
    kw = dict(initial_train_days=365, test_days=30, embargo_days=1, step_days=60,
              max_folds=3, day_tz=HKT, first_test_day="2022-07-05")
    early = pd.date_range("2021-03-24", "2023-12-31", freq="1h", tz=HKT).tz_convert("UTC")
    late = pd.date_range("2021-07-01", "2023-12-31", freq="1h", tz=HKT).tz_convert("UTC")
    f_early = list(ExpandingWindowSplitter(**kw).split(early))
    f_late = list(ExpandingWindowSplitter(**kw).split(late))
    assert len(f_early) == len(f_late) == 3
    for a, b in zip(f_early, f_late):
        assert a.test_idx.equals(b.test_idx)          # identical scored windows
    assert f_early[0].test_idx.tz_convert(HKT).min() == pd.Timestamp("2022-07-05", tz=HKT)
    assert len(f_early[0].train_idx) > len(f_late[0].train_idx)  # all history is used


def test_first_test_day_requires_a_full_training_year():
    import pytest

    short = pd.date_range("2021-12-01", "2023-12-31", freq="1h", tz=HKT).tz_convert("UTC")
    sp = ExpandingWindowSplitter(initial_train_days=365, day_tz=HKT, first_test_day="2022-07-05")
    with pytest.raises(ValueError, match="training days before first_test_day"):
        next(sp.split(short))


# --------------------------------------------------------------------- C3 ---
def test_persistence_fallback_uses_same_hour_of_an_earlier_day():
    from pvfc.models.baselines import SmartPersistence

    idx = pd.date_range("2022-06-01", periods=4 * 24, freq="1h", tz="UTC")
    k = pd.Series(np.tile(np.r_[np.zeros(6), np.full(12, 0.8), np.zeros(6)], 4), index=idx)
    k.iloc[1 * 24:2 * 24] = np.r_[np.zeros(6), np.full(12, 0.5), np.zeros(6)]  # day 2: k = 0.5
    k.iloc[2 * 24 + 8] = np.nan                       # day 3, hour 8: a data gap
    frame = pd.DataFrame({"k": k, "p_clear": 10.0, "power": k * 10.0})
    history, future = frame.iloc[:3 * 24], frame.iloc[3 * 24:].drop(columns=["power", "k"])
    pred = SmartPersistence().predict(history, future)
    # the gap hour falls back to the SAME hour one day earlier (k = 0.5 -> 5 kW), not 0 kW
    assert pred["y_hat"].iloc[8] == 5.0
    assert pred["y_hat"].iloc[9] == 8.0               # untouched hours use yesterday


# --------------------------------------------------------------------- C4 ---
def test_hourly_files_are_relabelled_to_interval_start(tmp_path):
    from pvfc.data import _read_power

    hourly = tmp_path / "hourly.csv"
    hourly.write_text("Time,generation(kWh),power(W)\n2022-06-15 07:00:00,1.0,1000\n"
                      "2022-06-15 08:00:00,2.0,2000\n2022-06-15 09:00:00,3.0,3000\n")
    q15 = tmp_path / "q15.csv"
    q15.write_text("Time,generation(kWh),power(W)\n2022-06-15 07:00:00,0.25,1000\n"
                   "2022-06-15 07:15:00,0.5,2000\n2022-06-15 07:30:00,0.75,3000\n")
    p_h, step_h = _read_power(hourly)
    p_q, step_q = _read_power(q15)
    first_h = p_h.index[0].tz_convert(HKT)
    assert step_h == pd.Timedelta(hours=1) and first_h.hour == 6   # 07:00 stamp = 06:00-07:00
    assert p_q.index[0].tz_convert(HKT).hour == 7 and step_q == pd.Timedelta(minutes=15)


# --------------------------------------------------------------------- C5 ---
def test_power_spikes_above_nameplate_are_removed():
    from pvfc.data import _qc_power

    raw = pd.Series([5.0, 10.9, 11.1, 3000.0, np.nan])
    clean, counts = _qc_power(raw, capacity=10.0)
    assert counts["over_capacity_removed"] == 2 and clean.isna().sum() == 3
    assert clean.iloc[1] == 10.9


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
