"""Adapters wrapping darts models in the Forecaster interface.

darts (v0.44+) is the unified spine: it exposes baselines, classical
(ARIMA/SARIMAX, Prophet), ML (LightGBM/XGBoost), deep (LSTM/TFT/TiDE/TSMixer/
PatchTST) and foundation models (Chronos2Model, TimesFM2p5Model, TiRexModel)
behind one fit()/predict() API. We wrap them so the harness stays model-agnostic.

This is intentionally a thin stub for Week 1 — it imports darts lazily and is
fleshed out in Weeks 5-8. The mapping below records the intended darts class per
config name so the registry is self-documenting.
"""
from __future__ import annotations

import pandas as pd

from .base import Forecaster

# config name -> (darts module path, class name). Verify exact availability
# against the installed darts version (PatchTSTFMModel may be unreleased).
DARTS_CLASSES = {
    "sarima": ("darts.models", "AutoARIMA"),
    "lightgbm": ("darts.models", "LightGBMModel"),
    "lstm": ("darts.models", "RNNModel"),          # model="LSTM", unidirectional on MPS
    "tft": ("darts.models", "TFTModel"),
    "tide": ("darts.models", "TiDEModel"),
    "chronos2": ("darts.models", "Chronos2Model"),
    "timesfm": ("darts.models", "TimesFM2p5Model"),
    "tirex": ("darts.models", "TiRexModel"),
}


class DartsForecaster(Forecaster):
    """Skeleton adapter — to be completed in Weeks 5-8.

    Plan for implementation: convert the (history, future) frames to darts
    TimeSeries (target = clear-sky index k, future_covariates = p_clear + calendar
    + forecast weather), call .fit(train) / .predict(n=horizon), then reconstruct
    power via k_hat * p_clear. Quantile output (where supported) populates q* cols.
    """

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self._model = None

    def fit(self, train: pd.DataFrame) -> "DartsForecaster":  # pragma: no cover
        raise NotImplementedError(
            f"DartsForecaster '{self.name}' not implemented yet (Weeks 5-8). "
            "See module docstring for the conversion recipe."
        )

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError


def build_darts_model(name: str, **kwargs) -> Forecaster:
    if name not in DARTS_CLASSES:
        raise KeyError(f"No darts mapping for '{name}'. Known: {sorted(DARTS_CLASSES)}")
    return DartsForecaster(name, **kwargs)
