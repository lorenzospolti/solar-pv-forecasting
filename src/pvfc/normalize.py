"""Normalization fit on TRAIN ONLY, per fold.

Two complementary transforms:
- clear-sky-index space: the model works on k = power / p_clear instead of raw
  power, so it learns weather-driven variability rather than the sunrise. This
  is the standard solar-forecasting move (plan §Evaluation, correction #5).
- standardization of features/target, with statistics fit only on the training
  portion of each fold to prevent leakage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EPS = 1e-6

#: How many days back ``fill_index_gaps`` looks for the same hour of an earlier day; the
#: persistence references look back as far (``baselines._lookup_same_hour``).
GAP_FILL_BACK_DAYS = 7


def clearsky_index(power: pd.Series, p_clear: pd.Series, clip: float = 1.3) -> pd.Series:
    """k = power / p_clear on the daytime mask; 0 at night; clip cloud-enhancement.

    p_clear<=EPS (night / very low sun) -> k = 0. Daytime hours where ``power``
    is missing stay NaN (NOT 0): a data gap is not "no generation", and training
    on fake zeros would bias every model. Models that need a series without gaps
    fill them (``fill_index_gaps``); the evaluator drops non-finite truth.
    """
    k = power / p_clear.where(p_clear > EPS)
    k = k.clip(lower=0.0, upper=clip)
    return k.where(p_clear > EPS, 0.0)  # night -> 0; daytime gaps stay NaN


def fill_index_gaps(k: pd.Series, is_day: pd.Series | None = None,
                    back_days: int = GAP_FILL_BACK_DAYS) -> pd.Series:
    """Fill the gaps of a clear-sky index series for a model that needs none.

    A missing hour takes the index of the same hour on the most recent earlier day
    that has one, at most ``back_days`` back: the rule the persistence references
    follow. Hours still missing (the first day of a series, or a gap longer than
    ``back_days``) take the series' mean daytime index on daytime hours (``is_day``;
    every hour when it is not given) and 0 at other hours. A forward fill would carry
    the last valid hour, almost always a night-time 0, through a missing day, so the
    model would see every outage day as a day without output, in training and in its
    context (panel v2 was started with that fill).
    """
    out = k.copy()
    for back in range(1, back_days + 1):
        if not out.isna().any():
            break
        out = out.fillna(k.shift(freq=pd.Timedelta(days=back)).reindex(k.index))
    if out.isna().any():
        day = (pd.Series(True, index=k.index) if is_day is None
               else is_day.reindex(k.index, fill_value=False).astype(bool))
        k_bar = float(k[day].mean()) if k[day].notna().any() else 0.0
        out = out.fillna(pd.Series(np.where(day, k_bar, 0.0), index=k.index))
    return out


def reconstruct_power(k: pd.Series, p_clear: pd.Series) -> pd.Series:
    """Inverse transform: P_hat = k_hat * p_clear (auto-imposes night zeros)."""
    return (k.clip(lower=0.0) * p_clear.clip(lower=0.0)).rename("y_hat")


@dataclass
class StandardScaler:
    """Minimal, leakage-safe standard scaler fit on train only."""

    mean_: dict[str, float] = field(default_factory=dict)
    std_: dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "StandardScaler":
        self.mean_ = df.mean().to_dict()
        self.std_ = df.std(ddof=0).replace(0, 1.0).to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return (df - pd.Series(self.mean_)) / pd.Series(self.std_)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df * pd.Series(self.std_) + pd.Series(self.mean_)
