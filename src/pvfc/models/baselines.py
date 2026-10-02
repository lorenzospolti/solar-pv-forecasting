"""Rung-0 baselines — mandatory references that define the skill score.

- Persistence: naive same-hour-yesterday power. Easy to beat; meaningless bar.
- SmartPersistence: persist *yesterday's clear-sky index* onto today's clear-sky
  template. This is the real bar (plan §Risk-2). y_hat(t) = k(t-1d) * p_clear(t).
- SeasonalNaive: same-hour from `season` periods ago (24h or 168h).
- Climatology: the training window's mean daytime clear-sky index on today's
  template. y_hat(t) = k_bar * p_clear(t).
- CLIPER: the optimal convex combination of climatology and persistence (Yang
  2019). y_hat(t) = [rho k(t-1d) + (1 - rho) k_bar] * p_clear(t), rho = the
  lag-24 h autocorrelation of the clear-sky index.

Why the last two: at a 24-h lag smart persistence is a weak bar on this data —
plain persistence scores skill -0.001 against it over the panel — because
yesterday's cloudiness says little about today's (rho ~ 0.45-0.5). In theory
CLIPER's MSE is never above climatology's or persistence's, and Yang et al.
(2020) strongly recommend the RMSE skill score over CLIPER in all solar
forecasting studies, so skill over CLIPER is the stricter, comparable number.

All operate on the (history, future) protocol on an hourly, tz-aware frame:
the target (``power`` or ``k``) is read only from ``train`` / ``history``, and
nothing but ``p_clear`` from ``future``.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .base import Forecaster

logger = logging.getLogger("pvfc.models")


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


def _daytime_k(frame: pd.DataFrame, k_col: str, day_col: str) -> pd.Series:
    """The clear-sky index on daytime hours, NaN elsewhere (night, twilight, gaps)."""
    return frame[k_col].where(frame[day_col].astype(bool))


def _lag_correlation(k: pd.Series, lag: pd.Timedelta) -> tuple[float, int]:
    """Pearson correlation of (k(t), k(t - lag)) over the pairs where both are
    finite, and the number of pairs.

    Pairs are matched by timestamp (``reindex``), never by row position: a gap in
    the index would otherwise pair hours that are not ``lag`` apart.
    """
    now = k.to_numpy(dtype=float)
    then = k.reindex(k.index - lag).to_numpy(dtype=float)
    ok = np.isfinite(now) & np.isfinite(then)
    n = int(ok.sum())
    if n < 2:
        raise ValueError(f"only {n} pairs of finite values {lag} apart: rho is undefined")
    x, y = now[ok] - now[ok].mean(), then[ok] - then[ok].mean()
    scale = np.sqrt(np.mean(x * x) * np.mean(y * y))
    # a constant series carries no persistence signal: weight 0 (pure climatology)
    return (float(np.mean(x * y) / scale) if scale > 0 else 0.0), n


class Climatology(Forecaster):
    """Single-valued climatology of the clear-sky index: y_hat(t) = k_bar * p_clear(t).

    ``k_bar`` is the mean clear-sky index over the training window's daytime hours
    (``is_day``: solar zenith < 85 deg, the filter Yang et al. (2020) prescribe
    for the climatology mean) — one value for every hour, Yang's single-valued
    climatology. A mean per hour of day ("multiple-valued" in Murphy's terms) is
    deliberately not used: Yang et al. (2020) leave it out of the standard because
    its many possible conditionings defeat standardisation, and on the panel it
    moves CLIPER's mean skill by 0.0001.

    Yang et al. (2020) take the mean over the verified sample itself ("internal"
    climatology); here it comes from each fold's training window, so the reference
    passes through the same leak-free fit/predict protocol as every model.
    """

    name = "climatology"

    def __init__(self, k_col: str = "k", p_clear_col: str = "p_clear",
                 day_col: str = "is_day"):
        self.k_col = k_col
        self.p_clear_col = p_clear_col
        self.day_col = day_col
        self.k_bar_: float | None = None

    def fit(self, train: pd.DataFrame) -> "Climatology":
        k = _daytime_k(train, self.k_col, self.day_col)
        if not np.isfinite(k).any():
            raise ValueError(f"{self.name}: no finite daytime {self.k_col!r} in the training frame")
        self.k_bar_ = float(k.mean())
        return self

    def _fitted_k_bar(self) -> float:
        if self.k_bar_ is None:
            raise RuntimeError(f"{self.name}: call fit() before predict()")
        return self.k_bar_

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        y = self._fitted_k_bar() * future[self.p_clear_col].clip(lower=0.0).to_numpy()
        return pd.DataFrame({"y_hat": y}, index=future.index)


class CLIPER(Climatology):
    """CLImatology-PERsistence: their optimal convex combination (Yang 2019).

    y_hat(t) = [rho * k(t - 1 d) + (1 - rho) * k_bar] * p_clear(t)

    For a clear-sky index with variance sigma^2 and lag-h autocorrelation rho_h,
    climatology has MSE sigma^2, h-step persistence 2 (1 - rho_h) sigma^2, and
    w * persistence + (1 - w) * climatology is MSE-optimal at w = rho_h, with MSE
    (1 - rho_h^2) sigma^2 <= both (Yang 2019; Yang et al. 2020, Eqs. 15-18).

    rho is the Pearson correlation of k(t) and k(t - 24 h) over the training
    window's pairs in which BOTH hours are daytime with a finite k, matched by
    timestamp. Yang et al. (2020) compute the autocorrelation on unfiltered data
    "to preserve the temporal spacing"; matching by timestamp preserves it
    without letting night hours in, whose k = 0 by construction would inflate
    rho (~0.47 -> ~0.75 on CYT (Station 2)). A negative estimate is clipped to 0
    (pure climatology), which keeps the combination convex. Like ``k_bar``, rho
    is estimated on the training window only, not on the verified sample as Yang
    et al. (2020) define it; in a month far from the climatology (the overcast
    March 2023 fold) CLIPER is therefore less strict than its in-sample version.

    The persistence term is SmartPersistence's (same ``_lookup_same_hour`` gap
    fallback), so CLIPER with rho = 1 IS smart persistence and with rho = 0 IS
    Climatology.
    """

    name = "cliper"

    def __init__(self, k_col: str = "k", p_clear_col: str = "p_clear",
                 day_col: str = "is_day"):
        super().__init__(k_col, p_clear_col, day_col)
        self.rho_: float | None = None

    def fit(self, train: pd.DataFrame) -> "CLIPER":
        super().fit(train)
        rho, n_pairs = _lag_correlation(_daytime_k(train, self.k_col, self.day_col),
                                        pd.Timedelta(days=1))
        self.rho_ = float(np.clip(rho, 0.0, 1.0))
        logger.info("cliper: rho = %.3f (lag-24 h autocorrelation of the daytime clear-sky "
                    "index, %d training pairs), k_bar = %.3f", self.rho_, n_pairs, self.k_bar_)
        return self

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        k_bar = self._fitted_k_bar()
        if self.rho_ is None:
            raise RuntimeError(f"{self.name}: call fit() before predict()")
        k_lag = _lookup_same_hour(history[self.k_col], future.index, pd.Timedelta(days=1))
        k_hat = self.rho_ * k_lag.to_numpy() + (1.0 - self.rho_) * k_bar
        y = k_hat * future[self.p_clear_col].clip(lower=0.0).to_numpy()
        return pd.DataFrame({"y_hat": y}, index=future.index)
