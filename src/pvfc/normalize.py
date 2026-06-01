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


def clearsky_index(power: pd.Series, p_clear: pd.Series, clip: float = 1.3) -> pd.Series:
    """k = power / p_clear on the daytime mask; 0 at night; clip cloud-enhancement.

    p_clear<=EPS (night / very low sun) -> k = 0 (and these are masked out of
    fitting). The clip caps physically-implausible k>clip from cloud enhancement.
    """
    k = power / p_clear.where(p_clear > EPS)
    k = k.clip(lower=0.0, upper=clip)
    return k.fillna(0.0)


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
