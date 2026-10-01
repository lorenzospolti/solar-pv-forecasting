"""Rung-0 baselines — mandatory references that define the skill score.

- Persistence: naive same-hour-yesterday power. Easy to beat; meaningless bar.
- SmartPersistence: persist *yesterday's clear-sky index* onto today's clear-sky
  template. This is the real bar (plan §Risk-2). y_hat(t) = k(t-1d) * p_clear(t).
- SeasonalNaive: same-hour from `season` periods ago (24h or 168h).

All operate on the (history, future) protocol and need columns ``power`` and
``p_clear`` to be present and hourly-indexed (tz-aware).
"""
from __future__ import annotations

import pandas as pd

from .base import Forecaster


def _lookup_same_hour(series: pd.Series, target_index: pd.DatetimeIndex, lag: pd.Timedelta,
                      max_back_days: int = 7) -> pd.Series:
    """For each t in target_index, fetch series[t - lag]. Where that value is
    missing (a data gap), step back whole days — t - lag - 1 d, - 2 d, ... up to
    ``max_back_days`` — i.e. use the most recent value observed AT THE SAME HOUR.

    Never fall back to the last value of the series: at issue time (local
    midnight) that is a night-time zero, which would turn every post-gap hour of
    the reference forecast into 0 kW and inflate every skill score. Values still
    missing after ``max_back_days`` (a week-long outage) are 0.
    """
    vals = series.reindex(target_index - lag)
    vals.index = target_index
    for back in range(1, max_back_days + 1):
        if not vals.isna().any():
            break
        older = series.reindex(target_index - lag - pd.Timedelta(days=back))
        older.index = target_index
        vals = vals.fillna(older)
    return vals.fillna(0.0)


class Persistence(Forecaster):
    name = "persistence"

    def fit(self, train: pd.DataFrame) -> "Persistence":
        return self

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        y = _lookup_same_hour(history["power"], future.index, pd.Timedelta(days=1))
        return pd.DataFrame({"y_hat": y.values}, index=future.index)


class SmartPersistence(Forecaster):
    """Clear-sky-index persistence: the professional reference."""

    name = "smart_persistence"

    def __init__(self, k_col: str = "k", p_clear_col: str = "p_clear"):
        self.k_col = k_col
        self.p_clear_col = p_clear_col

    def fit(self, train: pd.DataFrame) -> "SmartPersistence":
        return self

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        k_ref = _lookup_same_hour(history[self.k_col], future.index, pd.Timedelta(days=1))
        y = (k_ref.values * future[self.p_clear_col].clip(lower=0.0).values)
        return pd.DataFrame({"y_hat": y}, index=future.index)


class SeasonalNaive(Forecaster):
    def __init__(self, season_hours: int = 24):
        self.season_hours = season_hours
        self.name = f"seasonal_naive_{season_hours}h"

    def fit(self, train: pd.DataFrame) -> "SeasonalNaive":
        return self

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        y = _lookup_same_hour(
            history["power"], future.index, pd.Timedelta(hours=self.season_hours)
        )
        return pd.DataFrame({"y_hat": y.values}, index=future.index)
