"""Time-aware, causal cross-validation.

Expanding-window blocked walk-forward on *whole-day blocks*, chronological,
with an embargo gap of at least the forecast horizon (1 day for day-ahead).
No shuffling, ever. The *same* Splitter instance is reused by every model so
all models see identical folds (see plan §Evaluation, §Risk-3 leakage).

Day blocks are defined in ``day_tz`` (the site's local timezone) when given:
for an UTC+8 site, UTC-normalized "days" would run 08:00->07:59 local, so the
issue time would sit mid-morning and forecasts would straddle two solar days.
Setting ``day_tz="Asia/Hong_Kong"`` makes blocks (and therefore the runner's
issue boundaries) true local calendar days. HK has no DST -> unambiguous.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass
class Fold:
    index: int
    train_idx: pd.DatetimeIndex
    test_idx: pd.DatetimeIndex


@dataclass
class ExpandingWindowSplitter:
    """Yield (train, test) folds over day blocks.

    Parameters
    ----------
    initial_train_days : days in the first training window.
    test_days          : days per test fold.
    embargo_days       : gap between train end and test start (>= horizon).
    step_days          : how far the origin advances each fold (default = test_days).
    max_folds          : optional cap.
    day_tz             : timezone defining "a day" (None -> the index's own tz).
    """

    initial_train_days: int = 365
    test_days: int = 30
    embargo_days: int = 1
    step_days: int | None = None
    max_folds: int | None = None
    day_tz: str | None = None

    def _day_index(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Each timestamp's calendar day (midnight) in the block timezone."""
        return (index.tz_convert(self.day_tz) if self.day_tz else index).normalize()

    def split(self, index: pd.DatetimeIndex) -> Iterator[Fold]:
        if index.tz is None:
            raise ValueError("Index must be tz-aware (UTC). See plan §ETL.")
        norm = self._day_index(index)
        days = np.array(sorted(pd.Series(norm).unique()))
        step = self.step_days or self.test_days
        fold = 0
        train_end = self.initial_train_days
        while True:
            test_start = train_end + self.embargo_days
            test_end = test_start + self.test_days
            if test_start >= len(days) or test_end > len(days):
                break
            train_days = set(days[:train_end])
            test_day_set = set(days[test_start:test_end])
            train_idx = index[norm.isin(train_days)]
            test_idx = index[norm.isin(test_day_set)]
            if len(train_idx) and len(test_idx):
                yield Fold(fold, train_idx, test_idx)
                fold += 1
                if self.max_folds and fold >= self.max_folds:
                    break
            train_end += step

    def n_splits(self, index: pd.DatetimeIndex) -> int:
        return sum(1 for _ in self.split(index))
