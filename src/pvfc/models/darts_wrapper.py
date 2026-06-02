"""Adapters wrapping darts models in the Forecaster interface.

darts (v0.41, pinned) is the unified spine: baselines, classical (ARIMA/SARIMAX),
ML (LightGBM/XGBoost), deep (LSTM/TFT/TiDE/…) and foundation models behind one
fit()/predict() API. We wrap them so the harness stays model-agnostic.

The recipe (plan §Software-architecture): model the clear-sky index ``k`` as the
target; pass ``p_clear`` + calendar + (forecast) weather as *future covariates*
known at issue time; reconstruct power as ``k_hat * p_clear``; populate quantile
columns where supported.

Runner contract this fits: ``fit(train)`` once per fold, then ``predict(history,
future)`` per test-day with a fresh, growing ``history``. Both model families used
here accept a fresh series at predict time:
  * LightGBM is a *global* model — ``predict(n, series=history, future_covariates=…)``.
  * ARIMA is a *transferable local* model — same call refits on the given series.
Local models are capped to a recent ``context_days`` window so the per-day refit
stays tractable on multi-year hourly data.

Week 5 wires SARIMA + LightGBM; deep/foundation models (lstm, tft, chronos2, …)
follow in Weeks 6-8 via the same conversion helpers.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..data import WEATHER_COLS
from ..evaluation import DEFAULT_QUANTILES, enforce_monotone_quantiles
from ..features import calendar_features
from ..normalize import reconstruct_power
from .base import Forecaster

logger = logging.getLogger("pvfc.models")

# config name -> (darts module path, class name). SARIMA uses the statsmodels-backed
# ARIMA (no statsforecast dependency); the rest land in Weeks 6-8.
DARTS_CLASSES = {
    "sarima": ("darts.models", "ARIMA"),           # statsmodels SARIMAX, transferable
    "lightgbm": ("darts.models", "LightGBMModel"),  # global, native quantile likelihood
    "lstm": ("darts.models", "RNNModel"),           # model="LSTM", unidirectional on MPS
    "tft": ("darts.models", "TFTModel"),
    "tide": ("darts.models", "TiDEModel"),
    "chronos2": ("darts.models", "Chronos2Model"),
    "timesfm": ("darts.models", "TimesFM2p5Model"),
}

_WIRED = {"sarima", "lightgbm", "lstm"}  # implemented so far


def _target_series(df: pd.DataFrame, target_col: str = "k"):
    """Clear-sky index history -> a regular-hourly float32 darts TimeSeries."""
    from darts import TimeSeries

    s = df[target_col].astype("float32").asfreq("h")
    return TimeSeries.from_series(s.ffill().fillna(0.0))


def _covariate_series(history: pd.DataFrame, future: pd.DataFrame, weather_cols: list[str]):
    """Future-covariate TimeSeries over history ∪ future: p_clear + calendar + weather.

    All components are known at issue time (deterministic clear-sky template,
    calendar, and the chosen weather feed), so using them over the horizon is leak-free.
    """
    from darts import TimeSeries

    cols = ["p_clear"] + list(weather_cols)
    span = pd.concat([history[cols], future[cols]])
    span = span[~span.index.duplicated(keep="first")].sort_index().asfreq("h")
    cov = pd.concat([span, calendar_features(span.index)], axis=1)
    cov = cov.astype("float32").ffill().bfill()
    return TimeSeries.from_dataframe(cov)


class DartsForecaster(Forecaster):
    """Adapter turning a darts model into a harness Forecaster (target = k)."""

    def __init__(self, name: str, **params):
        self.name = name
        self.params = dict(params)
        self.supports_quantiles = bool(params.get("quantile", False))
        self.quantiles = [float(q) for q in DEFAULT_QUANTILES] if self.supports_quantiles else None
        self.context_days = params.get("context_days", 60)  # cap for local refit cost
        self._weather_cols = params.get("weather_cols")      # None -> auto-detect available
        self._model = None
        self._mode = None        # "global" | "local"
        self._likelihood = None  # "quantile" (lightgbm/QR) | "sample" (arima) | None
        self._fit_cols: list[str] = []
        self._scale = False      # torch models (LSTM) need train-fit input scaling
        self._tscaler = None
        self._cscaler = None

    # ------------------------------------------------------------------ #
    def _resolve_weather_cols(self, df: pd.DataFrame) -> list[str]:
        wanted = self._weather_cols if self._weather_cols is not None else WEATHER_COLS
        return [c for c in wanted if c in df.columns]

    def _build(self) -> None:
        if self._model is not None:
            return
        seed = int(self.params.get("seed", 0))
        if self.name == "lightgbm":
            from darts.models import LightGBMModel

            self._mode, self._likelihood = "global", ("quantile" if self.supports_quantiles else None)
            # config gives positive look-back hours; darts wants strictly-negative lags
            lags = [-abs(int(h)) for h in self.params.get("lags", [24, 48, 168])]
            self._model = LightGBMModel(
                lags=lags,
                lags_future_covariates=self.params.get("lags_future_covariates", [0]),
                output_chunk_length=self.params.get("output_chunk_length", 24),
                likelihood=self._likelihood,
                quantiles=self.quantiles,
                random_state=seed,
                verbose=-1,
            )
        elif self.name == "sarima":
            self._mode = "local"
            season = int(self.params.get("seasonal_m", 24))
            try:  # preferred: statsforecast AutoARIMA (automatic order search, transferable)
                from darts.models import AutoARIMA

                self._model = AutoARIMA(
                    season_length=season, quantiles=self.quantiles, random_state=seed,
                    max_p=int(self.params.get("max_p", 3)), max_q=int(self.params.get("max_q", 3)),
                    stepwise=True, approximation=True,
                )
                self._likelihood = "quantile" if self.supports_quantiles else None
            except ImportError:  # fallback: fixed-order statsmodels SARIMAX (no statsforecast)
                from darts.models import ARIMA

                order = tuple(self.params.get("order", (2, 0, 1)))
                seasonal = tuple(self.params.get("seasonal_order", (1, 0, 1, season)))
                self._model = ARIMA(p=order[0], d=order[1], q=order[2],
                                    seasonal_order=seasonal, random_state=seed)
                self._likelihood = "sample" if self.supports_quantiles else None
        elif self.name == "lstm":
            from darts.models import RNNModel

            from ..utils import get_device

            self._mode, self._scale = "global", True
            if self.supports_quantiles:  # quantile LSTM (via a Scaler pipeline) is future work
                logger.warning("lstm: quantile output not wired yet; producing point forecasts")
                self.supports_quantiles, self.quantiles = False, None
            icl = int(self.params.get("input_chunk_length", 168))
            device = get_device()  # MPS with CPU fallback (handles the Tahoe bug)
            self._model = RNNModel(
                model="LSTM",
                input_chunk_length=icl,
                training_length=int(self.params.get("training_length", icl + 24)),
                hidden_dim=int(self.params.get("hidden_dim", 32)),
                n_rnn_layers=int(self.params.get("n_rnn_layers", 1)),
                dropout=float(self.params.get("dropout", 0.0)),
                n_epochs=int(self.params.get("n_epochs", 20)),
                random_state=seed,
                force_reset=True,
                pl_trainer_kwargs={
                    "accelerator": "mps" if device == "mps" else "cpu",
                    "enable_progress_bar": False, "enable_model_summary": False, "logger": False,
                },
            )
        else:
            raise NotImplementedError(
                f"darts model '{self.name}' is not wired yet (Weeks 6-8). "
                f"Wired now: {sorted(_WIRED)}."
            )

    def fit(self, train: pd.DataFrame) -> "DartsForecaster":
        self._build()
        self._fit_cols = self._resolve_weather_cols(train)
        target = _target_series(train, "k")
        cov = _covariate_series(train, train.iloc[0:0], self._fit_cols)
        if self._scale:  # train-only fit, applied identically at predict (leakage-safe)
            from darts.dataprocessing.transformers import Scaler

            self._tscaler, self._cscaler = Scaler().fit(target), Scaler().fit(cov)
            target, cov = self._tscaler.transform(target), self._cscaler.transform(cov)
        self._model.fit(target, future_covariates=cov)
        return self

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        n = len(future)
        hist = history
        if self._mode == "local" and self.context_days:
            cutoff = history.index.max() - pd.Timedelta(days=self.context_days)
            hist = history.loc[history.index >= cutoff]
        target = _target_series(hist, "k")
        cov = _covariate_series(hist, future, self._fit_cols)
        p_clear = future["p_clear"]

        if self.supports_quantiles:
            kq = enforce_monotone_quantiles(self._predict_quantiles(target, cov, n))
            k_point = kq[0.5]
        elif self._scale:
            fc = self._model.predict(n=n, series=self._tscaler.transform(target),
                                     future_covariates=self._cscaler.transform(cov))
            k_point = np.asarray(self._tscaler.inverse_transform(fc).values()).reshape(-1)[:n]
            kq = None
        else:
            fc = self._model.predict(n=n, series=target, future_covariates=cov)
            k_point = np.asarray(fc.values()).reshape(-1)[:n]
            kq = None

        out = pd.DataFrame(index=future.index)
        out["y_hat"] = reconstruct_power(pd.Series(k_point, index=future.index), p_clear).to_numpy()
        if kq is not None:
            for tau, arr in kq.items():
                out[f"q{tau}"] = reconstruct_power(
                    pd.Series(arr, index=future.index), p_clear).to_numpy()
        return out

    def _predict_quantiles(self, target, cov, n: int) -> dict[float, np.ndarray]:
        """Return {tau: k-array(n)} from whichever probabilistic path the model uses."""
        if self._likelihood == "quantile":  # LightGBM quantile likelihood
            fc = self._model.predict(n=n, series=target, future_covariates=cov,
                                     predict_likelihood_parameters=True)
            df = fc.to_dataframe()
            parsed = {round(float(c.rsplit("_q", 1)[1]), 2): df[c].to_numpy()[:n] for c in df.columns}
            return {q: parsed[q] for q in self.quantiles if q in parsed}
        # sampling path (ARIMA): Monte-Carlo draws -> empirical quantiles
        fc = self._model.predict(n=n, series=target, future_covariates=cov, num_samples=200)
        vals = fc.all_values()  # (time, component, samples)
        return {q: np.quantile(vals[:, 0, :], q, axis=-1)[:n] for q in self.quantiles}


def build_darts_model(name: str, **kwargs) -> Forecaster:
    if name not in DARTS_CLASSES:
        raise KeyError(f"No darts mapping for '{name}'. Known: {sorted(DARTS_CLASSES)}")
    return DartsForecaster(name, **kwargs)
