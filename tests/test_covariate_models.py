"""Weather covariates for TiRex and TimesFM (10 Oct 2026).

* ``pvfc.models.xreg``: the in-context ridge regression ("XReg") that lets TiRex and
  TimesFM-2.5, which read no covariates, use the weather forecast;
* the ``xreg`` path of ``DartsForecaster``, with a stand-in for the darts model, so no
  checkpoint is needed;
* the TimesFM-3 and TiRex-2 wrappers (``pvfc.models.covariate_foundation``), with stand-ins
  at the dependency boundary: the covariates must reach the model aligned with the hours it
  predicts, and the TimesFM-3 licence gate must hold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from pvfc.models import build
from pvfc.models.covariate_foundation import (
    COVARIATE_FOUNDATION_ARCHES,
    TimesFM3Forecaster,
    TiRex2Forecaster,
)
from pvfc.models.xreg import ClearSkyGHICache, design_matrix, fit_ridge, nwp_clearsky_index

WEATHER = ["temp_air", "ghi", "wind_speed", "rh"]
SITE = (22.337, 114.263, 60.0)


def _frame(days: int = 30, seed: int = 0) -> pd.DataFrame:
    """Hourly UTC frame whose clear-sky index is an exact linear function of forecast GHI."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-03-01", periods=24 * days, freq="1h", tz="UTC")
    local = (idx.hour + 8) % 24
    day = (local >= 7) & (local <= 17)
    ghi = np.where(day, rng.uniform(50.0, 900.0, len(idx)), 0.0)
    k = np.where(day, 0.2 + 0.001 * ghi, 0.0)
    p_clear = np.where(day, 5.0, 0.0)
    return pd.DataFrame({"power": k * p_clear, "p_clear": p_clear, "k": k, "is_day": day,
                         "ghi": ghi, "temp_air": rng.normal(25, 3, len(idx)),
                         "wind_speed": rng.uniform(0, 8, len(idx)),
                         "rh": rng.uniform(40, 95, len(idx))}, index=idx)


def _split(frame: pd.DataFrame):
    history = frame.iloc[:-24]
    future = frame.iloc[-24:].drop(columns=["power", "k"])
    return history, future


# ------------------------------------------------------------------ regression ---
def test_fit_ridge_recovers_a_linear_relation():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    y = 0.3 + 0.5 * X["a"] - 0.2 * X["b"]
    np.testing.assert_allclose(fit_ridge(y, X, ridge=0.0).predict(X), y, atol=1e-8)


def test_fit_ridge_uses_only_masked_rows_and_never_shrinks_the_intercept():
    X = pd.DataFrame({"a": np.arange(100.0)})
    y = pd.Series(np.r_[np.full(50, 1.0), np.full(50, 5.0)])
    mask = np.r_[np.zeros(50, bool), np.ones(50, bool)]
    np.testing.assert_allclose(fit_ridge(y, X, mask, ridge=1e12).predict(X), 5.0, atol=1e-6)


def test_fit_ridge_with_too_few_rows_predicts_the_mean():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    fit = fit_ridge(pd.Series([1.0, 2.0, 3.0]), X)
    np.testing.assert_allclose(fit.predict(X), 2.0)


def test_design_matrix_holds_the_chronos2_covariates_and_kc():
    f = _frame(days=3)
    X = design_matrix(f, WEATHER, "covariates")
    assert list(X.columns) == ["p_clear", *WEATHER, "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    Xk = design_matrix(f, WEATHER, "covariates+kc", ClearSkyGHICache(*SITE))
    assert list(Xk.columns) == [*X.columns, "kc_nwp"]
    assert Xk["kc_nwp"].between(0.0, 1.5).all()
    # no forecast GHI (weather setting "none"): kc cannot be formed, covariates only
    bare = f[["p_clear", "k", "is_day"]]
    assert "kc_nwp" not in design_matrix(bare, [], "covariates+kc", ClearSkyGHICache(*SITE))


def test_nwp_clearsky_index_is_one_under_clear_sky_and_zero_at_night():
    idx = pd.date_range("2022-03-01", periods=48, freq="1h", tz="UTC")
    clear = ClearSkyGHICache(*SITE).get(idx)
    kc = nwp_clearsky_index(pd.DataFrame({"ghi": clear}, index=idx), clear)
    assert (kc[clear >= 20.0] - 1.0).abs().max() < 1e-9
    assert (kc[clear < 20.0] == 0.0).all()


# --------------------------------------------------------- darts wrapper, xreg ---
def test_xreg_path_adds_the_regression_to_the_residual_forecast(monkeypatch):
    """k is exactly linear in forecast GHI, and the stand-in model forecasts a zero
    residual: the forecast must then equal the truth hour by hour (alignment check)."""
    pytest.importorskip("darts")
    from darts import TimeSeries

    from pvfc.models.darts_wrapper import DartsForecaster

    class ZeroResidual:
        def predict(self, n, series, predict_likelihood_parameters=False, **kw):
            start = series.end_time() + pd.Timedelta(hours=1)
            idx = pd.date_range(start, periods=n, freq="h")
            cols = {f"k_q{q:.2f}": np.zeros(n) for q in np.arange(1, 10) / 10}
            return TimeSeries.from_dataframe(pd.DataFrame(cols, index=idx))

    def fake_build(self):
        self._model, self._mode, self._likelihood = ZeroResidual(), "global", "quantile"

    monkeypatch.setattr(DartsForecaster, "_build", fake_build)
    monkeypatch.setattr(DartsForecaster, "_fit_model", lambda self, target, cov: None)
    frame = _frame()
    history, future = _split(frame)
    model = build("tirex_xreg", arch="tirex", input_chunk_length=24 * 20,
                  output_chunk_length=24, quantile=True, accept_license=True,
                  xreg={"features": "covariates", "ridge": 0.0})
    model.fit(history)
    pred = model.predict(history, future)
    truth = frame["power"].iloc[-24:].to_numpy()
    np.testing.assert_allclose(pred["y_hat"].to_numpy(), truth, atol=1e-6)
    np.testing.assert_allclose(pred["q0.9"].to_numpy(), truth, atol=1e-6)


def test_xreg_is_refused_for_models_that_are_not_foundation_models():
    pytest.importorskip("darts")
    with pytest.raises(ValueError, match="xreg"):
        build("lightgbm", xreg=True)


# ------------------------------------------------------ TimesFM-3 / TiRex-2 ---
def test_covariate_arches_registered_and_built():
    assert COVARIATE_FOUNDATION_ARCHES == frozenset({"timesfm3", "tirex2"})
    assert isinstance(build("timesfm3"), TimesFM3Forecaster)
    t = build("tirex2_nwp", arch="tirex2", input_chunk_length=4096)
    assert isinstance(t, TiRex2Forecaster) and t.input_chunk_length == 2048  # its maximum


def test_timesfm3_refuses_to_load_without_the_licence_flag():
    with pytest.raises(PermissionError, match="non-commercial"):
        build("timesfm3", quantile=True).fit(_frame(days=3))


def test_fine_tuning_is_refused():
    with pytest.raises(NotImplementedError):
        build("timesfm3", finetune=True, accept_license=True).fit(_frame(days=3))


def _ghi_row(model) -> int:
    return 1 + model._cov_cols.index("ghi")  # row 0 is p_clear


def test_timesfm3_sees_the_covariates_of_the_hours_it_predicts(monkeypatch):
    seen = {}

    @dataclass
    class Out:
        quantiles: np.ndarray

    class FakeTimesFM3:
        class config:  # mimics the package's `config` attribute
            quantiles = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

        def predict(self, context, horizon, past_future_covariates, return_quantiles):
            seen.update(T=len(context), cov=past_future_covariates.shape, h=horizon)
            k = 0.2 + 0.001 * past_future_covariates[seen["ghi"], -horizon:]
            return Out(np.repeat(k[:, None], 9, axis=1))

    model = build("timesfm3", input_chunk_length=24 * 10, quantile=True, accept_license=True)
    monkeypatch.setattr(TimesFM3Forecaster, "_load_model", lambda self: FakeTimesFM3())
    frame = _frame()
    history, future = _split(frame)
    model.fit(history)
    seen["ghi"] = _ghi_row(model)
    pred = model.predict(history, future)
    assert seen["T"] == 240 and seen["cov"] == (9, 240 + 24) and seen["h"] == 24
    truth = frame["power"].iloc[-24:].to_numpy()
    np.testing.assert_allclose(pred["y_hat"].to_numpy(), truth, atol=1e-4)
    assert {f"q{q}" for q in (0.1, 0.5, 0.9)} <= set(pred.columns)


def test_tirex2_sees_the_covariates_of_the_hours_it_predicts(monkeypatch):
    torch = pytest.importorskip("torch")

    @dataclass
    class Series:
        target: object
        past_covariates: object
        future_covariates: object

    class FakeTiRex2:
        quantiles = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        def forecast(self, series, prediction_length, output_type):
            ts = series[0]
            assert ts.target.shape == (1, 240) and ts.past_covariates is None
            assert tuple(ts.future_covariates.shape) == (9, 240 + prediction_length)
            ghi = ts.future_covariates[FakeTiRex2.ghi, -prediction_length:].numpy()
            k = 0.2 + 0.001 * ghi
            return [np.repeat(k[None, None, :], 9, axis=1)]  # (1 target, 9 quantiles, n)

    model = build("tirex2", input_chunk_length=24 * 10, quantile=True)
    monkeypatch.setattr(TiRex2Forecaster, "_load_model", lambda self: FakeTiRex2())
    monkeypatch.setattr(TiRex2Forecaster, "_series_type", staticmethod(lambda: Series))
    frame = _frame()
    history, future = _split(frame)
    model.fit(history)
    FakeTiRex2.ghi = _ghi_row(model)
    pred = model.predict(history, future)
    np.testing.assert_allclose(pred["y_hat"].to_numpy(), frame["power"].iloc[-24:].to_numpy(),
                               atol=1e-4)
