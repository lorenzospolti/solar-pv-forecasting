"""Quality control of the measured power: outage hours.

Some stations report exactly 0 kW for hours or days while the sun is up and the other
stations generate: SQ1 from 13:00 on 7 May to 13:00 on 16 May 2023 (Hong Kong time),
SQ15 from 10:00 on 9 November to 09:00 on 11 November 2022 on clear days. These are
inverter outages and shutdowns that the data feed stores as measured zeros, not as
missing data. Left in, they teach the models that a sunny day can produce nothing,
they penalise (as truth) every forecast of the sunshine that did fall, and they flatter
smart persistence, which repeats the zeros the next day.

The rule (``outage_mask``). With C the station's nameplate capacity, an hour is a
*candidate* when

* it is a daytime hour (``is_day``),
* its power is finite and at most ``OUTAGE_MAX_POWER_RATIO`` x C (0.1 % of C), and
* its clear-sky power ``p_clear`` is at least ``OUTAGE_MIN_CLEAR_RATIO`` x C (10 % of C).

An *outage hour* is a candidate inside a run of at least ``OUTAGE_MIN_RUN_HOURS`` (3)
consecutive candidates, consecutive meaning timestamps exactly one hour apart: a
non-candidate hour, a missing value or a missing timestamp ends a run.

Why. Cloud dims a PV array; it does not switch it off. Real overcast never yields zero
output for hours while the sun is well up (a clear-sky power of a tenth of C or more):
diffuse light alone keeps the array generating. A run of zero output over such hours is
therefore a state of the system (an inverter outage, a site-wide shutdown, a logger
writing zeros), not weather. The run length keeps the rule conservative: it removes
sustained zero output, not isolated readings.

Use (ETL v3 and later, ``pvfc.data``). The ETL flags outage hours on each station's whole hourly
series and sets their power to missing, like any other data gap, before it refits the
clear-sky template and computes the clear-sky index: the template fit never sees the
outage zeros, no model sees them as hours without output (a model that needs a series
without gaps fills them from the same hour of an earlier day,
``pvfc.normalize.fill_index_gaps``) and no outage hour is scored as truth.
``outage_mask_frame`` applies the same rule to a saved predictions file test window by
test window (``window``); it serves audits of forecasts made before ETL v3 (panel v1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: a candidate hour reads at most this share of nameplate capacity C (0.1 %)
OUTAGE_MAX_POWER_RATIO = 0.001
#: ... while its clear-sky power is at least this share of C (the sun is well up)
OUTAGE_MIN_CLEAR_RATIO = 0.1
#: candidates are outage hours only in a run of at least this many consecutive hours
OUTAGE_MIN_RUN_HOURS = 3

_HOUR_NS = 3_600_000_000_000
#: the predictions-file columns ``outage_mask_frame`` reads
FRAME_COLUMNS = ("y_true", "p_clear", "is_day", "fold")


def outage_candidates(power, p_clear, is_day, capacity: float) -> np.ndarray:
    """Hours that could be part of an outage: daytime, finite power at most
    ``OUTAGE_MAX_POWER_RATIO`` x C, clear-sky power at least ``OUTAGE_MIN_CLEAR_RATIO``
    x C (a missing ``p_clear`` never qualifies). Array-likes of equal length; C in the
    unit of ``power``."""
    capacity = float(capacity)
    if not (np.isfinite(capacity) and capacity > 0):
        raise ValueError(f"capacity must be a positive number, got {capacity!r}")
    power = np.asarray(power, dtype=float)
    p_clear = np.asarray(p_clear, dtype=float)
    is_day = np.asarray(is_day, dtype=bool)
    if not power.shape == p_clear.shape == is_day.shape or power.ndim != 1:
        raise ValueError("power, p_clear and is_day must be 1-d and of equal length, got "
                         f"{power.shape}, {p_clear.shape}, {is_day.shape}")
    return (is_day & np.isfinite(power) & (power <= OUTAGE_MAX_POWER_RATIO * capacity)
            & (p_clear >= OUTAGE_MIN_CLEAR_RATIO * capacity))


def outage_mask(time, power, p_clear, is_day, capacity: float, *, window=None) -> np.ndarray:
    """Boolean array, True at the outage hours (module docstring: the rule and why).

    ``time`` holds the hourly timestamps (a DatetimeIndex or datetime array) of the
    ``power``, ``p_clear`` and ``is_day`` values, in any order; the result follows the
    input order. ``window`` (optional, one label per hour, e.g. the fold): runs are
    found within each window separately, so a run never continues from one window into
    the next. Without it, all hours form one window.
    """
    if not pd.api.types.is_datetime64_any_dtype(time):
        raise TypeError("time must be datetime-like (e.g. a DatetimeIndex), got "
                        f"{type(time).__name__}")
    stamps = pd.DatetimeIndex(time).as_unit("ns").asi8
    candidate = outage_candidates(power, p_clear, is_day, capacity)
    n = len(stamps)
    groups = np.zeros(n, dtype=np.int64) if window is None else pd.factorize(
        np.asarray(window))[0]
    if candidate.size != n or len(groups) != n:
        raise ValueError(f"time ({n}), the values ({candidate.size}) and window "
                         f"({len(groups)}) must have the same length")
    order = np.lexsort((stamps, groups))  # by window, then by time
    c, t, g = candidate[order], stamps[order], groups[order]
    joined = np.zeros(n, dtype=bool)      # row i continues the run of row i - 1
    joined[1:] = c[1:] & c[:-1] & (g[1:] == g[:-1]) & (np.diff(t) == _HOUR_NS)
    run = np.cumsum(~joined)               # run id: a new one wherever a row is not joined
    length = np.bincount(run, weights=c)[run]  # candidates in the row's run
    out = np.empty(n, dtype=bool)
    out[order] = c & (length >= OUTAGE_MIN_RUN_HOURS)
    return out


def outage_mask_frame(frame: pd.DataFrame, capacity: float) -> np.ndarray:
    """``outage_mask`` of a predictions frame as the runner saves it (a time index and
    the ``y_true``, ``p_clear``, ``is_day`` and ``fold`` columns), test window by test
    window (``fold``): the outage hours left in forecasts scored before ETL v3."""
    missing = [c for c in FRAME_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(f"cannot flag outage hours without the columns {missing}")
    return outage_mask(frame.index, frame["y_true"], frame["p_clear"], frame["is_day"],
                       capacity, window=frame["fold"])
