"""The Forecaster contract — the spine of the whole harness.

Every model in the thesis (persistence, smart persistence, SARIMA, LightGBM,
LSTM, and each foundation model) is a Forecaster. The Runner only ever talks to
this interface, which is what guarantees identical treatment across models.

Prediction protocol (day-ahead): given ``history`` (everything observed up to
the forecast issue time) and ``future`` (the target timestamps plus the
covariates *known at issue time* — clear-sky template, calendar features, and
NWP forecast weather), return one prediction per future timestamp.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Forecaster(ABC):
    #: human-readable model name, used as the results-row key
    name: str = "forecaster"
    #: whether predict() can populate quantile columns (q10..q90)
    supports_quantiles: bool = False

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> "Forecaster":
        """Fit on the training frame. Baselines may be no-ops."""

    @abstractmethod
    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        """Return a frame indexed by ``future.index`` with column ``y_hat``
        (and optionally ``q0.1`` .. ``q0.9`` when ``supports_quantiles``).

        ``future`` carries only information available at issue time; models
        MUST NOT read the realized target from it (leakage guard).
        """

    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} name={self.name!r}>"


def empty_prediction(future: pd.DataFrame, quantiles: list[float] | None = None) -> pd.DataFrame:
    """Helper: a correctly-shaped, zero-filled prediction frame."""
    cols = {"y_hat": 0.0}
    if quantiles:
        cols.update({f"q{q}": 0.0 for q in quantiles})
    return pd.DataFrame(cols, index=future.index)
