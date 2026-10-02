"""Tests for the climatology and CLIPER references (pvfc.models.baselines).

CLIPER (Yang 2019) forecasts y_hat = [rho k(t - 1 d) + (1 - rho) k_bar] * p_clear, so
its two ends must BE the existing references: rho = 1 is smart persistence (gap
fallback included) and rho = 0 is climatology. Its weight rho is the lag-24 h
autocorrelation of the DAYTIME clear-sky index, with pairs matched by timestamp so
that gaps in the index cannot shift them. k_bar and rho are fitted once per fold on
the training window; the issue-time history feeds only the persistence term.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pvfc.data import synthetic_station
from pvfc.models import build
from pvfc.models.baselines import CLIPER, Climatology, SmartPersistence
from pvfc.runner import run_model
from pvfc.splits import ExpandingWindowSplitter

HKT = "Asia/Hong_Kong"


def _station_frame(k: np.ndarray, days: int) -> pd.DataFrame:
    """Hourly frame: daytime 06:00-17:00 (p_clear = 10 kW), night p_clear = 0, k = 0."""
    idx = pd.date_range("2022-06-01", periods=days * 24, freq="1h", tz="UTC")
    day = (idx.hour >= 6) & (idx.hour < 18)
    p_clear = np.where(day, 10.0, 0.0)
    k = np.where(day, k, 0.0)
    return pd.DataFrame({"power": k * p_clear, "p_clear": p_clear, "k": k, "is_day": day},
                        index=idx)


def _gappy_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three days of history (k = 0.8, except 0.5 on day 2) and day 4 as the future.

    Day 3 has a value gap at 08:00 (k = NaN) and a row gap at 10:00 (no row), so
    persistence must fall back to day 2 at those hours: 0.5 x 10 kW = 5 kW, not 0.
    """
    k = np.repeat([0.8, 0.5, 0.8, 0.8], 24)
    frame = _station_frame(k, days=4)
    frame.iloc[2 * 24 + 8, frame.columns.get_indexer(["power", "k"])] = np.nan
    frame = frame.drop(index=frame.index[2 * 24 + 10])
    issue = frame.index[0] + pd.Timedelta(days=3)
    history = frame.loc[frame.index < issue]
    future = frame.loc[frame.index >= issue].drop(columns=["power", "k"])  # as the runner does
    return history, future


# ------------------------------------------------------------------ (a) ---
def test_climatology_is_the_mean_daytime_k_on_the_template():
    frame = _station_frame(np.repeat([0.2, 0.8, 0.5], 24), days=3)
    frame.iloc[24 + 9, frame.columns.get_indexer(["power", "k"])] = np.nan  # daytime gap
    twilight = frame.index.hour == 18                  # not daytime, but p_clear > 0 ...
    frame.loc[twilight, ["p_clear", "k", "power"]] = [1.0, 1.3, 1.3]  # ... and a noisy low-sun k
    train, future = frame.iloc[:48], frame.iloc[48:].drop(columns=["power", "k"])

    model = Climatology().fit(train)
    k_bar = (12 * 0.2 + 11 * 0.8) / 23                 # daytime, finite: no night/twilight/gap
    assert model.k_bar_ == pytest.approx(k_bar)
    pred = model.predict(train, future)
    np.testing.assert_allclose(pred["y_hat"], k_bar * future["p_clear"])
    assert pred.loc[future.index.hour == 18, "y_hat"].iloc[0] == pytest.approx(k_bar)


def test_k_bar_comes_from_the_training_window_not_the_history():
    # the runner fits on `train` once per fold, then predicts each test day from the
    # whole past (`history`: train + embargo + earlier test days), here 2 days at k = 0.2
    frame = _station_frame(np.repeat([0.8] * 5 + [0.2] * 3, 24), days=8)
    train, history = frame.iloc[:5 * 24], frame.iloc[:7 * 24]
    future = frame.iloc[7 * 24:].drop(columns=["power", "k"])
    p_clear = future["p_clear"].to_numpy()

    np.testing.assert_allclose(Climatology().fit(train).predict(history, future)["y_hat"],
                               0.8 * p_clear)
    model = CLIPER().fit(train)
    model.rho_ = 0.0
    np.testing.assert_allclose(model.predict(history, future)["y_hat"], 0.8 * p_clear)
    model.rho_ = 0.5                                   # yesterday's k = 0.2 does come from history
    np.testing.assert_allclose(model.predict(history, future)["y_hat"],
                               (0.5 * 0.2 + 0.5 * 0.8) * p_clear)


# ------------------------------------------------------------------ (b) ---
def test_cliper_ends_are_smart_persistence_and_climatology():
    history, future = _gappy_split()
    sp = SmartPersistence().predict(history, future)["y_hat"].to_numpy()
    clim = Climatology().fit(history).predict(history, future)["y_hat"].to_numpy()
    assert sp[8] == sp[10] == 5.0                      # both gap paths fell back to day 2

    model = CLIPER().fit(history)
    model.rho_ = 1.0
    np.testing.assert_array_equal(model.predict(history, future)["y_hat"], sp)
    model.rho_ = 0.0
    np.testing.assert_array_equal(model.predict(history, future)["y_hat"], clim)
    model.rho_ = 0.3                                   # in between: the convex blend itself
    np.testing.assert_allclose(model.predict(history, future)["y_hat"], 0.3 * sp + 0.7 * clim)


# ------------------------------------------------------------------ (c) ---
def test_rho_is_the_lag_one_day_autocorrelation_despite_gaps():
    rng = np.random.default_rng(0)
    days, phi = 500, 0.6
    anomaly = np.zeros((days, 24))
    for d in range(1, days):                           # one AR(1) across days per hour
        anomaly[d] = phi * anomaly[d - 1] + rng.normal(0.0, 0.1, 24)
    frame = _station_frame(0.55 + anomaly.ravel(), days=days)
    frame.loc[rng.random(len(frame)) < 0.05, "k"] = np.nan   # value gaps
    frame = frame[rng.random(len(frame)) >= 0.10]            # 10% of the rows missing
    # matching pairs by row position (shift(24)) would give ~0.06 here, and letting
    # the night zeros in ~0.96: only timestamp-matched daytime pairs recover phi
    assert abs(CLIPER().fit(frame).rho_ - phi) < 0.05


def test_rho_ignores_twilight_hours():
    # twilight: the sun is up (p_clear > 0) but is_day is False (zenith >= 85 deg), and k
    # carries a low-sun bias that repeats every day; admitting it would pull rho to ~0.97
    rng = np.random.default_rng(0)
    hours = np.arange(60 * 24)
    k = 0.55 + 0.1 * np.sin(2 * np.pi * hours / (10 * 24)) + rng.normal(0.0, 0.05, hours.size)
    frame = _station_frame(k, days=60)                 # a 10-day weather cycle: rho ~ 0.55
    dawn, dusk = frame.index.hour == 5, frame.index.hour == 18
    frame.loc[dawn | dusk, "p_clear"] = 0.5
    frame.loc[dawn, "k"] = 0.2
    frame.loc[dusk, "k"] = 1.6
    frame["power"] = frame["k"] * frame["p_clear"]

    rho = CLIPER().fit(frame.loc[~(dawn | dusk)]).rho_
    assert CLIPER().fit(frame).rho_ == pytest.approx(rho, abs=1e-12)
    lit = frame["k"].where(frame["p_clear"] > 0)      # the fixture discriminates
    assert lit.corr(lit.reindex(lit.index - pd.Timedelta(days=1)).set_axis(lit.index)) > rho + 0.2


def test_negative_rho_is_clipped_to_pure_climatology():
    frame = _station_frame(np.repeat(np.tile([0.3, 0.9], 5), 24), days=10)  # k flips daily
    model = CLIPER().fit(frame)
    assert model.rho_ == 0.0
    future = frame.iloc[-24:].drop(columns=["power", "k"])
    np.testing.assert_array_equal(model.predict(frame, future)["y_hat"],
                                  Climatology().fit(frame).predict(frame, future)["y_hat"])


# ------------------------------------------------------------------ (d) ---
def test_predict_needs_no_target_and_night_is_zero():
    history, future = _gappy_split()
    future = future.copy()
    future.iloc[2, future.columns.get_loc("p_clear")] = -0.01  # template noise below zero
    assert not {"power", "k"} & set(future.columns)
    for model in (Climatology(), CLIPER()):
        pred = model.fit(history).predict(history, future)
        assert pred.index.equals(future.index) and list(pred.columns) == ["y_hat"]
        assert np.isfinite(pred["y_hat"]).all()
        assert (pred["y_hat"][future["p_clear"] <= 0] == 0).all()
        assert (pred["y_hat"][future["p_clear"] > 0] > 0).all()


def test_unfitted_or_dataless_models_fail_loudly():
    history, future = _gappy_split()
    for model in (Climatology(), CLIPER()):
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(history, future)
        with pytest.raises(ValueError, match="no finite daytime"):
            model.fit(history.assign(k=np.nan))


# ------------------------------------------------------------------ (e) ---
def test_registry_builds_both_by_name():
    for name, cls in (("climatology", Climatology), ("cliper", CLIPER)):
        model = build(name)
        assert isinstance(model, cls) and model.name == name
        assert not model.supports_quantiles


def test_cliper_beats_smart_persistence_through_the_runner():
    series = synthetic_station(days=360, seed=2)
    sp = ExpandingWindowSplitter(initial_train_days=300, test_days=20, embargo_days=1,
                                 max_folds=2, day_tz=HKT)
    model = CLIPER()
    rows = run_model(model, series, sp)
    assert len(rows) == 2 * sp.n_splits(series.frame.index)
    assert 0.0 <= model.rho_ <= 1.0
    assert all(r["skill_rmse"] > 0 for r in rows if r["daytime_only"])
