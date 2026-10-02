"""Gap filling of the clear-sky index for models that need a series without gaps.

Panel v2 was started with a forward fill in the darts target builder: a missing day
(an ETL v3 outage day) took the last valid hour, a night-time 0, so every darts model
saw it as a day without output, while the references fell back to an earlier day.
"""
import numpy as np
import pandas as pd

from pvfc.models.darts_wrapper import _target_series
from pvfc.normalize import GAP_FILL_BACK_DAYS, fill_index_gaps


def _series(days: int = 10, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-06-01", periods=24 * days, freq="h")
    is_day = pd.Series((idx.hour >= 6) & (idx.hour < 18), index=idx)
    k = pd.Series(np.where(is_day, rng.uniform(0.2, 1.1, len(idx)), 0.0), index=idx)
    return k, is_day


def test_a_missing_day_takes_the_previous_day_not_zero():
    k, is_day = _series()
    gap = (k.index.normalize() == pd.Timestamp("2022-06-05")) & is_day
    filled = fill_index_gaps(k.where(~gap), is_day)
    previous_day = k.shift(freq=pd.Timedelta(days=1)).reindex(k.index)
    np.testing.assert_allclose(filled[gap], previous_day[gap])
    assert (filled[gap] > 0).all()


def test_observed_hours_are_unchanged_and_no_gap_is_left():
    k, is_day = _series()
    holes = k.where(np.random.default_rng(1).random(len(k)) > 0.1)
    filled = fill_index_gaps(holes, is_day)
    assert filled.notna().all()
    pd.testing.assert_series_equal(filled[holes.notna()], holes[holes.notna()])


def test_a_gap_skips_back_over_days_that_are_missing_too():
    k, is_day = _series()
    day = k.index.normalize()
    gap = (day >= pd.Timestamp("2022-06-04")) & (day <= pd.Timestamp("2022-06-06")) & is_day
    filled = fill_index_gaps(k.where(~gap), is_day)
    last_observed = k[(day == pd.Timestamp("2022-06-03")) & is_day].to_numpy()
    for d in pd.date_range("2022-06-04", "2022-06-06"):
        np.testing.assert_allclose(filled[(day == d) & is_day].to_numpy(), last_observed)


def test_beyond_the_look_back_a_gap_takes_the_mean_daytime_index():
    k, is_day = _series(days=12)
    day = k.index.normalize()
    start = day[0] + pd.Timedelta(days=2)
    gap = (day >= start) & (day < start + pd.Timedelta(days=GAP_FILL_BACK_DAYS + 2)) & is_day
    holes = k.where(~gap)
    filled = fill_index_gaps(holes, is_day)
    beyond = gap & (day >= start + pd.Timedelta(days=GAP_FILL_BACK_DAYS))
    assert beyond.sum() == 2 * 12  # the last two days of the gap, 12 daytime hours each
    np.testing.assert_allclose(filled[beyond], holes[is_day].mean())


def test_the_first_day_has_no_earlier_day_and_falls_back():
    k, is_day = _series()
    first = k.index.normalize() == k.index[0].normalize()
    holes = k.where(~first)
    filled = fill_index_gaps(holes, is_day)
    assert (filled[first & ~is_day] == 0).all()
    np.testing.assert_allclose(filled[first & is_day], holes[is_day].mean())


def test_darts_target_series_sees_no_dark_day_after_an_outage():
    k, is_day = _series()
    gap = (k.index.normalize() == pd.Timestamp("2022-06-05")) & is_day
    frame = pd.DataFrame({"k": k.where(~gap), "is_day": is_day})
    frame.index = frame.index.tz_localize("UTC")
    values = _target_series(frame).values().ravel()
    assert np.isfinite(values).all()
    assert (values[gap.to_numpy()] > 0).all()
