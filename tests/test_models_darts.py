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


def _frame(days: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=24 * days, freq="1h", tz="UTC")
    day = (np.sin(2 * np.pi * (idx.hour - 6) / 24) > 0).astype(float)
    k = np.clip(0.7 + 0.2 * rng.standard_normal(len(idx)), 0, 1.3) * day
    return pd.DataFrame({"k": k, "p_clear": 5.0 * day, "power": 5.0 * k}, index=idx)


def test_each_fit_starts_from_a_fresh_model():
    """Panel v1 bug: the runner fits the same forecaster once per fold, and darts' torch
    models continued training from the previous fold's weights. Two fits on the same
    data must now give identical forecasts from two distinct darts objects."""
    frame = _frame()
    train, history = frame.iloc[:-48], frame.iloc[:-24]
    future = frame.iloc[-24:].drop(columns=["power", "k"])
    model = build("lstm", input_chunk_length=48, hidden_dim=4, n_epochs=2)
    model.fit(train)
    first_obj, first = model._model, model.predict(history, future)["y_hat"].to_numpy()
    model.fit(train)
    second = model.predict(history, future)["y_hat"].to_numpy()
    assert model._model is not first_obj
    np.testing.assert_allclose(second, first, atol=1e-6)


def test_zero_shot_models_keep_their_checkpoint(monkeypatch):
    """Zero-shot foundation models never train: their loaded model is kept across folds
    (no reload billed to every fold's fit time), unlike fine-tuned ones."""
    from pvfc.models import darts_wrapper as dw

    zero_shot, tuned = build("tirex"), build("chronos2_ft", arch="chronos2", finetune=True)
    sentinel = object()
    for m in (zero_shot, tuned):
        m._model = sentinel
        monkeypatch.setattr(m, "_build", lambda m=m: None)
        monkeypatch.setattr(m, "_fit_model", lambda target, cov: None)
        m.fit(_frame(days=20))
    assert zero_shot._model is sentinel
    assert tuned._model is None and isinstance(tuned, dw.DartsForecaster)
