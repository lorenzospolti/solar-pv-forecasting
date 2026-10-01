"""Adapters wrapping darts models in the Forecaster interface.

darts (>=0.41) is the unified spine: baselines, classical (ARIMA/SARIMAX), ML
(LightGBM), deep (LSTM/TFT/TiDE) and **foundation** models (Chronos-2,
TimesFM-2.5, TiRex) behind one fit()/predict() API. We wrap them so the harness
stays model-agnostic.

The recipe (plan §Software-architecture): model the clear-sky index ``k`` as the
target; pass ``p_clear`` + calendar + (forecast) weather as *future covariates*
known at issue time; reconstruct power as ``k_hat * p_clear``; populate quantile
columns where supported.

Architecture vs. label
----------------------
A model spec has a config ``name`` (the results-row label, e.g. ``chronos2_ft``)
and an ``arch`` (which darts class to build, e.g. ``chronos2``). ``arch`` defaults
to ``name``. This lets zero-shot and fine-tuned variants of the same foundation
model coexist as distinct, separately-scored rows.

Runner contract: ``fit(train)`` once per fold, then ``predict(history, future)``
per test-day with a fresh, growing ``history``.
  * LightGBM / foundation models are *global* — ``predict(n, series=history, …)``.
  * ARIMA is a *transferable local* model — refits on the given (capped) series.
  * Foundation models require ``fit()`` even for **zero-shot** (it is training-free
    unless ``finetune`` is set); per test-day ``predict`` does in-context inference
    on that day's history.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from ..data import WEATHER_COLS
from ..evaluation import DEFAULT_QUANTILES, enforce_monotone_quantiles
from ..features import calendar_features
from ..normalize import reconstruct_power
from .base import Forecaster

logger = logging.getLogger("pvfc.models")

# config arch -> (darts module path, class name).
DARTS_CLASSES = {
    "sarima": ("darts.models", "ARIMA"),            # statsmodels SARIMAX / AutoARIMA, transferable
    "lightgbm": ("darts.models", "LightGBMModel"),  # global, native quantile likelihood
    "lstm": ("darts.models", "RNNModel"),           # model="LSTM", unidirectional on MPS
    "tft": ("darts.models", "TFTModel"),
    "tide": ("darts.models", "TiDEModel"),
    # foundation models (darts native): Chronos-2 (>=0.40), TimesFM-2.5 (>=0.41), TiRex (>=0.44)
    "chronos2": ("darts.models", "Chronos2Model"),
    "timesfm": ("darts.models", "TimesFM2p5Model"),
    "tirex": ("darts.models", "TiRexModel"),
}

#: foundation arches — global, training-free unless ``finetune`` is set
_FOUNDATION = {"chronos2", "timesfm", "tirex"}

#: arches with a working fit/predict path in this wrapper
_WIRED = {"sarima", "lightgbm", "lstm", "chronos2", "timesfm", "tirex"}

#: arches that *used* to fail loudly because no darts class exists. As of
#: v0.5 ``moirai2`` and ``ttm`` are wired via the external-foundation wrapper
#: (see :mod:`pvfc.models.external_foundation`), so the registry only retains
#: legacy aliases that should redirect a user who typed the v1 name.
_EXTERNAL = {
    "moirai": "Salesforce Moirai v1 is not wired. Use `arch: moirai2` "
              "(uni2ts; new external-foundation wrapper) or substitute darts "
              "`TiRexModel` (arch: tirex).",
}

# Foundation models that, by default, accept future covariates in darts.
_FOUNDATION_COV_DEFAULT = {"chronos2": True, "timesfm": False, "tirex": False}


def _darts_at_least(min_version: str) -> bool:
    """True if the installed darts is >= ``min_version`` (e.g. '0.42')."""
    import darts

    def parse(v: str) -> tuple[int, ...]:
        return tuple(int(re.sub(r"\D.*$", "", p) or 0) for p in v.split(".")[:3])

    return parse(getattr(darts, "__version__", "0")) >= parse(min_version)


def _quantile_regression(quantiles):
    """Import QuantileRegression across darts layouts (moved to `.torch` in 0.44)."""
    try:
        from darts.utils.likelihood_models.torch import QuantileRegression
    except ImportError:  # older darts
        from darts.utils.likelihood_models import QuantileRegression
    return QuantileRegression(list(quantiles))


def _naive_utc(obj):
    """darts drops tz info (with a warning per call): hand it naive UTC directly.
    Predictions are mapped back to the caller's index by position, not by label."""
    if getattr(obj.index, "tz", None) is not None:
        obj = obj.copy()
        obj.index = obj.index.tz_convert("UTC").tz_localize(None)
    return obj


def _target_series(df: pd.DataFrame, target_col: str = "k"):
    """Clear-sky index history -> a regular-hourly float32 darts TimeSeries."""
    from darts import TimeSeries

    s = _naive_utc(df[target_col].astype("float32")).asfreq("h")
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
    cov = _naive_utc(cov.astype("float32").ffill().bfill())
    return TimeSeries.from_dataframe(cov)


class DartsForecaster(Forecaster):
    """Adapter turning a darts model into a harness Forecaster (target = k)."""

    def __init__(self, name: str, arch: str | None = None, **params):
        self.name = name
        self.arch = arch or name
        self.params = dict(params)
        self.supports_quantiles = bool(params.get("quantile", False))
        self.quantiles = [float(q) for q in DEFAULT_QUANTILES] if self.supports_quantiles else None
        self.context_days = params.get("context_days", 60)  # cap for local/foundation refit cost
        self.finetune = params.get("finetune", False)        # foundation: bool | dict
        self._is_foundation = self.arch in _FOUNDATION
        self._weather_cols = params.get("weather_cols")      # None -> auto-detect available
        # foundation covariate use is per-arch (some checkpoints are univariate)
        self._use_cov = bool(params.get(
            "use_future_covariates", _FOUNDATION_COV_DEFAULT.get(self.arch, True)
        )) if self._is_foundation else True
        self._model = None
        self._mode = None        # "global" | "local"
        self._likelihood = None  # "quantile" | "sample" | None
        self._fit_cols: list[str] = []
        self._scale = False      # torch RNN (LSTM) needs train-fit input scaling
        self._tscaler = None
        self._cscaler = None

    # ------------------------------------------------------------------ #
    def _resolve_weather_cols(self, df: pd.DataFrame) -> list[str]:
        wanted = self._weather_cols if self._weather_cols is not None else WEATHER_COLS
        return [c for c in wanted if c in df.columns]

    def _device_trainer_kwargs(self) -> dict:
        from ..utils import get_device

        device = get_device()  # MPS with CPU fallback (handles the Tahoe bug)
        return {
            "accelerator": "mps" if device == "mps" else "cpu",
            "enable_progress_bar": False, "enable_model_summary": False, "logger": False,
        }

    def _build(self) -> None:
        if self._model is not None:
            return
        if self.arch in _EXTERNAL:
            raise NotImplementedError(f"arch '{self.arch}': {_EXTERNAL[self.arch]}")
        seed = int(self.params.get("seed", 0))
        if self.arch == "lightgbm":
            from darts.models import LightGBMModel

            # darts takes EVERY lag relative to the first step of the output chunk, and
            # the runner issues at local midnight with a 24-step chunk. So the horizon-h
            # model needs future-covariate lags 0..23 (lag h = the hour it predicts) and
            # target lags -1..-24 (yesterday's whole profile). The v0.5 setting
            # (lags_future_covariates=[0], lags 24/48/168) showed every horizon only the
            # midnight covariates (ghi = 0) and night-time target lags (k = 0): skill
            # -0.08 vs +0.26 on a real CYT fold. tests/test_models_darts.py guards this.
            ocl = int(self.params.get("output_chunk_length", 24))
            self._mode, self._likelihood = "global", ("quantile" if self.supports_quantiles else None)
            lags = self.params.get("lags", ocl)
            lags = (list(range(-int(lags), 0)) if isinstance(lags, int)
                    else [-abs(int(h)) for h in lags])
            self._model = LightGBMModel(
                lags=lags,
                lags_future_covariates=self.params.get("lags_future_covariates", list(range(ocl))),
                output_chunk_length=ocl,
                likelihood=self._likelihood,
                quantiles=self.quantiles,
                random_state=seed,
                verbose=-1,
            )
        elif self.arch == "sarima":
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
        elif self.arch == "lstm":
            from darts.models import RNNModel

            self._mode, self._scale = "global", True
            if self.supports_quantiles:  # quantile LSTM (via a Scaler pipeline) is future work
                logger.warning("lstm: quantile output not wired yet; producing point forecasts")
                self.supports_quantiles, self.quantiles = False, None
            icl = int(self.params.get("input_chunk_length", 168))
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
                pl_trainer_kwargs=self._device_trainer_kwargs(),
            )
        elif self._is_foundation:
            self._build_foundation(seed)
        else:
            raise NotImplementedError(
                f"darts arch '{self.arch}' is not wired. Wired: {sorted(_WIRED)}."
            )

    def _build_foundation(self, seed: int) -> None:
        """Construct a darts foundation model (zero-shot, or fine-tuned if requested)."""
        self._mode = "global"
        module, cls_name = DARTS_CLASSES[self.arch]
        import importlib

        cls = getattr(importlib.import_module(module), cls_name)

        kwargs: dict = {
            "input_chunk_length": int(self.params.get("input_chunk_length", 336)),
            "output_chunk_length": int(self.params.get("output_chunk_length", 24)),
            "random_state": seed,
            "pl_trainer_kwargs": self._device_trainer_kwargs(),
        }
        if self.supports_quantiles:
            kwargs["likelihood"] = _quantile_regression(self.quantiles)
            self._likelihood = "quantile"
        if self.arch == "chronos2" and self.params.get("hub_model_name"):
            kwargs["hub_model_name"] = self.params["hub_model_name"]
        if self.params.get("use_longer_projection_head") and self.arch == "timesfm":
            kwargs["use_longer_projection_head"] = True
        if self.params.get("accept_license"):
            # licence-gated checkpoints (TiRex: NXAI Community License). The user
            # accepts the terms by setting this in the config, never the code.
            kwargs["accept_license"] = True
        if self.finetune:
            if not _darts_at_least("0.42"):
                import darts
                raise RuntimeError(
                    f"fine-tuning foundation models needs darts>=0.42; installed "
                    f"{getattr(darts, '__version__', '?')}. Upgrade (python -m pip install -U "
                    "'darts[torch]>=0.44'; the u8darts name stopped at 0.41) "
                    "or run this model zero-shot (drop `finetune: true`)."
                )
            kwargs["enable_finetuning"] = self.finetune  # True (full) or {"unfreeze"|"freeze": [...]}
            kwargs["n_epochs"] = int(self.params.get("n_epochs", 5))
            if self.params.get("optimizer_kwargs"):  # darts default Adam lr=1e-3 is high for FT
                kwargs["optimizer_kwargs"] = dict(self.params["optimizer_kwargs"])
            if self.params.get("batch_size"):
                kwargs["batch_size"] = int(self.params["batch_size"])
        try:
            self._model = cls(**kwargs)
        except TypeError as exc:  # a checkpoint that rejects likelihood/cov kwargs
            logger.warning("%s: retrying construction without optional kwargs (%s)", self.name, exc)
            for k in ("likelihood", "use_longer_projection_head"):
                kwargs.pop(k, None)
            self.supports_quantiles, self.quantiles, self._likelihood = False, None, None
            self._model = cls(**kwargs)

    # ------------------------------------------------------------------ #
    def fit(self, train: pd.DataFrame) -> "DartsForecaster":
        self._build()
        fit_days = self.params.get("fit_days")
        if fit_days:  # cap the fit window (e.g. SARIMA, a local model, needs recent data only)
            train = train.loc[train.index >= train.index.max() - pd.Timedelta(days=int(fit_days))]
        self._fit_cols = self._resolve_weather_cols(train) if self._use_cov else []
        target = _target_series(train, "k")
        cov = _covariate_series(train, train.iloc[0:0], self._fit_cols) if self._use_cov else None
        if self._scale:  # train-only fit, applied identically at predict (leakage-safe)
            from darts.dataprocessing.transformers import Scaler

            self._tscaler = Scaler().fit(target)
            target = self._tscaler.transform(target)
            if cov is not None:
                self._cscaler = Scaler().fit(cov)
                cov = self._cscaler.transform(cov)
        self._fit_model(target, cov)
        return self

    def _fit_model(self, target, cov) -> None:
        """Fit, tolerating foundation checkpoints that reject future covariates.

        ``fit_stride`` (params) samples every n-th training window. darts applies the
        stride from the END of the series, and fold training data ends at 23:00 local,
        so ``fit_stride: 24`` trains only on windows whose forecast origin is local
        midnight — the day-ahead task itself — at 1/24 of the cost.
        """
        kw = {"stride": int(self.params["fit_stride"])} if self.params.get("fit_stride") else {}
        try:
            if cov is not None:
                self._model.fit(target, future_covariates=cov, **kw)
            else:
                self._model.fit(target, **kw)
        except (TypeError, ValueError) as exc:
            if cov is not None and self._is_foundation:
                logger.warning("%s: fitting without future covariates (%s)", self.name, exc)
                self._use_cov = False
                self._model.fit(target, **kw)
            else:
                raise

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        n = len(future)
        hist = history
        if self.context_days and (self._mode == "local" or self._is_foundation):
            cutoff = history.index.max() - pd.Timedelta(days=self.context_days)
            hist = history.loc[history.index >= cutoff]
        target = _target_series(hist, "k")
        cov = _covariate_series(hist, future, self._fit_cols) if self._use_cov else None
        p_clear = future["p_clear"]

        if self.supports_quantiles:
            kq = enforce_monotone_quantiles(self._predict_quantiles(target, cov, n))
            k_point = kq[0.5]
        elif self._scale:
            cov_t = self._cscaler.transform(cov) if (cov is not None and self._cscaler) else None
            fc = self._model.predict(n=n, series=self._tscaler.transform(target),
                                     future_covariates=cov_t)
            k_point = np.asarray(self._tscaler.inverse_transform(fc).values()).reshape(-1)[:n]
            kq = None
        else:
            fc = self._predict_point(target, cov, n)
            k_point = np.asarray(fc.values()).reshape(-1)[:n]
            kq = None

        out = pd.DataFrame(index=future.index)
        out["y_hat"] = reconstruct_power(pd.Series(k_point, index=future.index), p_clear).to_numpy()
        if kq is not None:
            for tau, arr in kq.items():
                out[f"q{tau}"] = reconstruct_power(
                    pd.Series(arr, index=future.index), p_clear).to_numpy()
        return out

    def _predict_point(self, target, cov, n):
        """Deterministic predict, tolerating foundation models without covariates."""
        if cov is not None:
            try:
                return self._model.predict(n=n, series=target, future_covariates=cov)
            except (TypeError, ValueError) as exc:
                if self._is_foundation:
                    logger.warning("%s: predicting without future covariates (%s)", self.name, exc)
                    self._use_cov = False
                    return self._model.predict(n=n, series=target)
                raise
        return self._model.predict(n=n, series=target)

    def _predict_quantiles(self, target, cov, n: int) -> dict[float, np.ndarray]:
        """Return {tau: k-array(n)} from whichever probabilistic path the model uses."""
        if self._likelihood == "quantile":  # LightGBM / foundation quantile likelihood
            kw = dict(n=n, series=target, predict_likelihood_parameters=True)
            if cov is not None:
                kw["future_covariates"] = cov
            try:
                fc = self._model.predict(**kw)
            except (TypeError, ValueError) as exc:
                if cov is not None and self._is_foundation:
                    logger.warning("%s: quantile predict without covariates (%s)", self.name, exc)
                    self._use_cov = False
                    kw.pop("future_covariates", None)
                    fc = self._model.predict(**kw)
                else:
                    raise
            df = fc.to_dataframe()
            parsed = {round(float(c.rsplit("_q", 1)[1]), 2): df[c].to_numpy()[:n]
                      for c in df.columns if "_q" in c}
            return {q: parsed[q] for q in self.quantiles if q in parsed}
        # sampling path (ARIMA): Monte-Carlo draws -> empirical quantiles
        kw = dict(n=n, series=target, num_samples=200)
        if cov is not None:
            kw["future_covariates"] = cov
        fc = self._model.predict(**kw)
        vals = fc.all_values()  # (time, component, samples)
        return {q: np.quantile(vals[:, 0, :], q, axis=-1)[:n] for q in self.quantiles}


def build_darts_model(name: str, **kwargs) -> Forecaster:
    """Build a DartsForecaster. ``name`` is the results label; ``arch`` (kwarg,
    default ``name``) selects the darts class — so e.g. a fine-tuned variant can
    use ``name: chronos2_ft, arch: chronos2``."""
    arch = kwargs.pop("arch", name)
    if arch not in DARTS_CLASSES and arch not in _EXTERNAL:
        known = sorted(set(DARTS_CLASSES) | set(_EXTERNAL))
        raise KeyError(f"No darts mapping for arch '{arch}' (from model '{name}'). Known: {known}")
    return DartsForecaster(name, arch=arch, **kwargs)
