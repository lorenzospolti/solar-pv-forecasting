"""Feature engineering for the tabular (LightGBM) and deep models.

Calendar + solar-geometry features are always known at issue time (safe as
day-ahead covariates). Lag/rolling features of the *target* must only use the
past — built per fold inside the harness to avoid leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclical encodings of hour-of-day and day-of-year (known in advance)."""
    hour = index.hour + index.minute / 60.0
    doy = index.dayofyear
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=index,
    )


def lag_features(series: pd.Series, lags: list[int]) -> pd.DataFrame:
    """Past lags of a series (hours). Causal: only shifts forward in time."""
    return pd.DataFrame({f"lag_{h}h": series.shift(h) for h in lags}, index=series.index)


def rolling_features(series: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Causal rolling means/stds (shifted by 1 so the current value is excluded)."""
    out = {}
    s = series.shift(1)
    for w in windows:
        out[f"rollmean_{w}h"] = s.rolling(w, min_periods=1).mean()
        out[f"rollstd_{w}h"] = s.rolling(w, min_periods=1).std().fillna(0.0)
    return pd.DataFrame(out, index=series.index)
