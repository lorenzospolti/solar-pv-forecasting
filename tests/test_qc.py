"""Tests for pvfc.qc, the outage rule of ETL v3: which hours are candidates (the
thresholds are inclusive and scale with the capacity), that only a run of at least
three consecutive candidates counts, that a gap, a missing timestamp, a night or a
test-window boundary ends a run, and that row order does not matter."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pvfc.qc import (
    OUTAGE_MAX_POWER_RATIO,
    OUTAGE_MIN_CLEAR_RATIO,
    OUTAGE_MIN_RUN_HOURS,
    outage_candidates,
    outage_mask,
    outage_mask_frame,
)

C = 20.0  # kW
HKT = "Asia/Hong_Kong"


def _hours(n: int, start: str = "2023-05-08 08:00") -> pd.DatetimeIndex:
    """``n`` consecutive hours from ``start`` (Hong Kong time), as UTC like the runner's."""
    return pd.date_range(start, periods=n, freq="h", tz=HKT).tz_convert("UTC")


def _bright(power) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """Consecutive daytime hours with the sun well up (p_clear = C / 2) and ``power``."""
    power = np.asarray(power, dtype=float)
    return _hours(len(power)), power, np.full(len(power), 0.5 * C), np.ones(len(power), bool)


def test_the_thresholds_are_the_decided_ones():
    assert OUTAGE_MAX_POWER_RATIO == 0.001
    assert OUTAGE_MIN_CLEAR_RATIO == 0.1
    assert OUTAGE_MIN_RUN_HOURS == 3


def test_only_runs_of_three_or_more_zero_hours_are_outages():
    t, p, pc, d = _bright([5, 0, 0, 0, 5, 0, 0, 5, 0, 5, 0, 0, 0, 0])
    np.testing.assert_array_equal(outage_mask(t, p, pc, d, C).astype(int),
                                  [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])


def test_candidate_thresholds_are_inclusive():
    power_max, clear_min = OUTAGE_MAX_POWER_RATIO * C, OUTAGE_MIN_CLEAR_RATIO * C
    power = np.array([power_max, 1.01 * power_max, 0.0, 0.0, 0.0, np.nan])
    p_clear = np.array([clear_min, clear_min, clear_min, 0.99 * clear_min, np.nan, C])
    np.testing.assert_array_equal(outage_candidates(power, p_clear, np.ones(6, bool), C),
                                  [True, False, True, False, False, False])
    # night is never a candidate, whatever the readings
    assert not outage_candidates(np.zeros(3), np.full(3, C), np.zeros(3, bool), C).any()


def test_the_thresholds_scale_with_the_capacity():
    power = np.array([0.0, 0.015, 0.03, 0.0, 0.0])
    p_clear = np.array([2.0, 2.0, 2.0, 1.5, 3.0])
    day = np.ones(5, bool)
    # C = 20: power <= 0.02 and p_clear >= 2; C = 30: power <= 0.03 and p_clear >= 3
    np.testing.assert_array_equal(outage_candidates(power, p_clear, day, 20.0),
                                  [True, True, False, False, True])
    np.testing.assert_array_equal(outage_candidates(power, p_clear, day, 30.0),
                                  [False, False, False, False, True])


@pytest.mark.parametrize("spoil", ["night", "output", "gap", "dim sky"])
def test_one_non_candidate_hour_splits_a_run(spoil):
    t, p, pc, d = _bright(np.zeros(5))
    assert outage_mask(t, p, pc, d, C).all()
    if spoil == "night":
        d[2] = False
    elif spoil == "output":
        p[2] = 0.01 * C
    elif spoil == "gap":
        p[2] = np.nan
    else:
        pc[2] = 0.05 * C
    assert not outage_mask(t, p, pc, d, C).any()  # two runs of two hours


def test_a_missing_timestamp_ends_a_run():
    t, p, pc, d = _bright(np.zeros(7))
    keep = [0, 1, 2, 4, 5]          # 08:00-10:00, then 12:00-13:00: 11:00 has no row
    np.testing.assert_array_equal(outage_mask(t[keep], p[keep], pc[keep], d[keep], C),
                                  [True, True, True, False, False])
    keep = [0, 1, 3, 4, 6]          # never three hours one hour apart
    assert not outage_mask(t[keep], p[keep], pc[keep], d[keep], C).any()


def test_runs_never_cross_a_test_window():
    t, p, pc, d = _bright(np.zeros(5))
    assert outage_mask(t, p, pc, d, C).all()
    np.testing.assert_array_equal(outage_mask(t, p, pc, d, C, window=[0, 0, 1, 1, 1]),
                                  [False, False, True, True, True])
    assert not outage_mask(t, p, pc, d, C, window=[7, 7, 9, 9, 8]).any()


def _two_days(seed: int = 0) -> pd.DataFrame:
    """Two local days at a 20 kW station: day one cloudy but generating, day two an
    outage from 09:00 (0 kW through the sunny hours and the dusk hour)."""
    t = _hours(48, "2023-05-07 00:00")
    hour = t.tz_convert(HKT).hour.to_numpy()
    p_clear = C * np.clip(np.sin(np.pi * (hour - 5.7) / 12.6), 0.0, None)  # < 0.1 C at 6, 18
    is_day = p_clear > 0
    power = p_clear * np.random.default_rng(seed).uniform(0.05, 1.0, size=48)
    power[24 + 9:] = 0.0
    return pd.DataFrame({"y_true": power, "p_clear": p_clear, "is_day": is_day,
                         "fold": 3}, index=t.rename("time"))


def test_an_outage_day_loses_its_sunny_hours_not_its_dawn_dusk_or_night():
    frame = _two_days()
    flags = outage_mask(frame.index, frame["y_true"], frame["p_clear"], frame["is_day"], C)
    hour = frame.index.tz_convert(HKT).hour
    # 09:00-17:00 of day two; 18:00 reads 0 too, but its clear-sky power is under 0.1 C
    expected = (np.arange(48) >= 24) & (hour >= 9) & (hour <= 17)
    np.testing.assert_array_equal(flags, expected)
    assert frame["y_true"].to_numpy()[24 + 18] == 0.0 and not flags[24 + 18]


def test_row_order_does_not_matter():
    parts = []
    for fold, seed in enumerate((1, 2, 3)):  # three windows of two days, one after another
        part = _two_days(seed).assign(fold=fold)
        part.index = part.index + pd.Timedelta(days=2 * fold)
        parts.append(part)
    frame = pd.concat(parts)
    args = (frame.index, frame["y_true"], frame["p_clear"], frame["is_day"], C)
    want = outage_mask(*args, window=frame["fold"])
    assert want.sum() == 3 * 9
    perm = np.random.default_rng(4).permutation(len(frame))
    shuffled = frame.iloc[perm]
    got = outage_mask(shuffled.index, shuffled["y_true"], shuffled["p_clear"],
                      shuffled["is_day"], C, window=shuffled["fold"])
    np.testing.assert_array_equal(got, want[perm])


def test_the_frame_version_flags_test_window_by_test_window():
    frame = _two_days()
    frame["fold"] = np.where(np.arange(48) < 24 + 11, 0, 1)  # a boundary at 11:00 of day two
    got = outage_mask_frame(frame, C)
    np.testing.assert_array_equal(got, outage_mask(
        frame.index, frame["y_true"], frame["p_clear"], frame["is_day"], C, window=frame["fold"]))
    # 09:00-10:00 is a run of two in the first window; 11:00-17:00 one of seven in the next
    assert got.sum() == 7 and not got[24 + 9:24 + 11].any()
    with pytest.raises(KeyError, match="p_clear"):
        outage_mask_frame(frame.drop(columns="p_clear"), C)


def test_bad_inputs_are_refused():
    t, p, pc, d = _bright(np.zeros(3))
    for capacity in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="capacity"):
            outage_mask(t, p, pc, d, capacity)
    with pytest.raises(TypeError, match="datetime"):
        outage_mask(np.arange(3), p, pc, d, C)  # integers would pass for nanoseconds
    with pytest.raises(ValueError, match="same length"):
        outage_mask(t[:2], p, pc, d, C)
    with pytest.raises(ValueError, match="same length"):
        outage_mask(t, p, pc, d, C, window=[0, 0])
    with pytest.raises(ValueError, match="equal length"):
        outage_mask(t, p[:2], pc, d, C)
    assert outage_mask(t[:0], p[:0], pc[:0], d[:0], C).shape == (0,)
