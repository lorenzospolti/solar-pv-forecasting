"""Regression tests for the darts adapters (need darts; skipped without it).

LightGBM day-ahead wiring (1 Oct 2026): darts takes every lag relative to the
first step of the output chunk. With forecasts issued at local midnight and a
24-step chunk, the v0.5 setting ``lags_future_covariates=[0]`` showed every
horizon only the midnight covariates. On a real CYT fold that meant skill -0.08
against smart persistence instead of +0.26.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("darts")

from pvfc.models import build  # noqa: E402


def test_lightgbm_sees_the_covariate_of_the_hour_it_predicts():
    """k equals an i.i.d. 'weather' covariate at the same hour, so a correctly wired
    model predicts tomorrow almost exactly; one that sees only the first output
    step's covariate cannot do better than the mean (error ~ the target's std)."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-01", periods=24 * 90, freq="1h", tz="UTC")
    ghi = rng.uniform(0.0, 1.0, len(idx))
    frame = pd.DataFrame({"k": ghi, "ghi": ghi, "p_clear": 1.0, "power": ghi}, index=idx)
    train, history = frame.iloc[:-48], frame.iloc[:-24]
    future = frame.iloc[-24:].drop(columns=["power", "k"])

    model = build("lightgbm", lags=24, quantile=False)
    model.fit(train)
    pred = model.predict(history, future)["y_hat"].to_numpy()
    rmse = float(np.sqrt(np.mean((pred - frame["power"].iloc[-24:].to_numpy()) ** 2)))
    assert rmse < 0.05, f"rmse {rmse:.3f} vs target std {ghi.std():.3f}: covariates misaligned"
