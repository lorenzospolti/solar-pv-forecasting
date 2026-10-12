"""Foundation models that read the weather forecast natively (known-future covariates).

Two models released after darts 0.47 fixed its roster (4 Sep 2026), so they get their own
glue, like Moirai-2 and TTM in ``external_foundation.py``:

* **TimesFM-3** (Google, Aug 2026): ``python -m pip install 'timesfm[torch]>=3.0.2'``,
  checkpoint ``google/timesfm-3.0-pytorch`` (about 330M parameters, context up to 15,360
  steps, past-only and past-and-future covariates). Torch backend on MPS/CPU, or Apple's
  MLX backend (``backend: mlx``; ``python -m pip install 'timesfm[mlx]>=3.0.2'``). The
  weights are under the TimesFM non-commercial licence v1.0, which allows academic
  research but not commercial or production use: set ``accept_license: true`` in the
  config once you have reviewed it (as for TiRex).
* **TiRex-2** (NXAI, Jul 2026): ``python -m pip install 'tirex-2>=0.3.1'``, checkpoint
  ``NX-AI/TiRex-2`` (38.4M parameters, plus 44.1M when more than one series is read;
  context 2,048 steps; past and future-known covariates). Apache-2.0, but the Hugging Face
  repo is gated: accept its terms on the model page and log in (``hf auth login``) once.

Both models receive the covariates Chronos-2 receives (``p_clear``, the weather columns and
the four calendar terms; ``darts_wrapper._covariate_series``), as covariates known over the
context and the horizon. ``covariate_features: covariates+kc`` adds the forecast's clear-sky
index (``pvfc.models.xreg.nwp_clearsky_index``). Zero-shot only: ``finetune`` raises.

Test mocking: ``_load_model`` (dependency + checkpoint) and ``_forecast`` (one window of
inference) are isolated, as in ``external_foundation.py``.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from ..data import WEATHER_COLS
from ..evaluation import DEFAULT_QUANTILES, enforce_monotone_quantiles
from ..normalize import fill_index_gaps, reconstruct_power
from .base import Forecaster
from .xreg import ClearSkyGHICache, day_mask, design_matrix

logger = logging.getLogger("pvfc.models.covariates")

_INSTALL = {
    "timesfm3": ("TimesFM-3 needs the `timesfm` package >= 3.0.2:\n"
                 "    python -m pip install 'timesfm[torch]>=3.0.2'   # or [mlx] for backend: mlx"),
    "tirex2": ("TiRex-2 needs the `tirex-2` package:\n"
               "    python -m pip install 'tirex-2>=0.3.1'\n"
               "and access to the gated Hugging Face repo NX-AI/TiRex-2 (accept the terms on the "
               "model page, then `hf auth login`)."),
}

#: arches handled by this module
COVARIATE_FOUNDATION_ARCHES: frozenset[str] = frozenset({"timesfm3", "tirex2"})


def _device(params: dict) -> str:
    from ..utils import get_device

    return str(params.get("device") or get_device())


def _levels(model: Any, n_cols: int) -> list[float]:
    """The quantile levels a checkpoint emits (its config, else the nine deciles)."""
    for obj in (getattr(model, "config", None), model):
        q = getattr(obj, "quantiles", None) if obj is not None else None
        if q is not None:
            q = q.tolist() if hasattr(q, "tolist") else list(q)
            if len(q) == n_cols:
                return [round(float(x), 2) for x in q]
    if n_cols == len(DEFAULT_QUANTILES):
        return [round(float(x), 2) for x in DEFAULT_QUANTILES]
    raise ValueError(f"cannot map {n_cols} quantile columns to levels")


class _NativeCovariateForecaster(Forecaster):
    """Shared glue: context window, covariate matrix, power reconstruction."""

    arch: str = ""
    max_context: int = 2048

    def __init__(self, name: str, arch: str | None = None, **params: Any) -> None:
        self.name = name
        self.arch = arch or type(self).arch or name
        self.params = dict(params)
        self.supports_quantiles = bool(params.get("quantile", False))
        self.quantiles: list[float] | None = (
            [float(q) for q in DEFAULT_QUANTILES] if self.supports_quantiles else None)
        self.input_chunk_length = min(int(params.get("input_chunk_length", self.max_context)),
                                      self.max_context)
        if int(params.get("input_chunk_length", self.max_context)) > self.max_context:
            logger.warning("%s: input_chunk_length capped at the model's maximum %d",
                           name, self.max_context)
        # history handed to the model: enough days to fill the context (+1 for the day cut)
        self.context_days = int(params.get("context_days",
                                           math.ceil(self.input_chunk_length / 24) + 1))
        self.output_chunk_length = int(params.get("output_chunk_length", 24))
        self.finetune = params.get("finetune", False)
        self.use_cov = bool(params.get("use_future_covariates", True))
        self.features = str(params.get("covariate_features", "covariates"))
        self._weather_cols = params.get("weather_cols")
        self._cov_cols: list[str] = []
        self._ghi_cache: ClearSkyGHICache | None = None
        self._model: Any = None

    # -- dependency boundary (tests monkeypatch these) ---------------------------------
    def _load_model(self) -> Any:  # pragma: no cover - subclasses implement
        raise NotImplementedError

    def _forecast(self, context: np.ndarray, covariates: np.ndarray | None,
                  n: int) -> dict[float, np.ndarray]:  # pragma: no cover
        """One window: ``context`` (T,), ``covariates`` (V, T+n) or None -> {tau: (n,)}."""
        raise NotImplementedError

    # -- Forecaster contract -----------------------------------------------------------
    def fit(self, train: pd.DataFrame) -> "_NativeCovariateForecaster":
        if self.finetune:
            raise NotImplementedError(
                f"{self.name}: no fine-tuning loop for arch '{self.arch}'; run it zero-shot.")
        if self._model is None:  # zero-shot: the checkpoint is loaded once, as in darts_wrapper
            self._model = self._load_model()
        wanted = self._weather_cols if self._weather_cols is not None else WEATHER_COLS
        self._cov_cols = [c for c in wanted if c in train.columns] if self.use_cov else []
        return self

    def _inputs(self, history: pd.DataFrame, future: pd.DataFrame
                ) -> tuple[np.ndarray, np.ndarray | None]:
        """Context k (gap-filled, ending the hour before ``future``) and the covariates
        over context + horizon, shape (V, T + n), on one regular hourly grid."""
        hist = history
        if self.context_days:
            hist = history.loc[history.index >= history.index.max()
                               - pd.Timedelta(days=self.context_days)]
        start = future.index[0]
        n_ctx = min(self.input_chunk_length,
                    int((start - hist.index[0]) / pd.Timedelta(hours=1)))
        ctx_idx = pd.date_range(end=start - pd.Timedelta(hours=1), periods=n_ctx, freq="h")
        hist = hist.reindex(ctx_idx)
        if "is_day" in hist:
            hist["is_day"] = hist["is_day"].astype("boolean").fillna(False).astype(bool)
        k = fill_index_gaps(hist["k"].astype(float), day_mask(hist) if "is_day" in hist else None)
        if not self.use_cov:
            return k.to_numpy(dtype=np.float32), None
        if self._ghi_cache is None and self.features != "covariates":
            from ..data import HKUST_ALT, HKUST_LAT, HKUST_LON

            self._ghi_cache = ClearSkyGHICache(HKUST_LAT, HKUST_LON, HKUST_ALT)
        span = pd.concat([hist.drop(columns=["k", "power"], errors="ignore"), future])
        X = design_matrix(span, self._cov_cols, self.features, self._ghi_cache)
        return k.to_numpy(dtype=np.float32), X.to_numpy(dtype=np.float32).T

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError(f"{self.name}: predict() called before fit()")
        n = len(future)
        context, cov = self._inputs(history, future)
        raw = self._forecast(context, cov, n)
        kq = {round(float(t), 2): np.asarray(v, dtype=float)[:n] for t, v in raw.items()}
        if 0.5 not in kq:
            raise ValueError(f"{self.name}: the model returned no median")
        p_clear = future["p_clear"]
        out = pd.DataFrame(index=future.index)
        if self.supports_quantiles:
            kq = enforce_monotone_quantiles({t: kq[t] for t in self.quantiles if t in kq})
        out["y_hat"] = reconstruct_power(pd.Series(kq[0.5], index=future.index), p_clear).to_numpy()
        if self.supports_quantiles:
            for tau, arr in kq.items():
                out[f"q{tau}"] = reconstruct_power(pd.Series(arr, index=future.index),
                                                   p_clear).to_numpy()
        return out


class TimesFM3Forecaster(_NativeCovariateForecaster):
    """Google TimesFM-3 (zero-shot) with the weather forecast as known-future covariates."""

    arch = "timesfm3"
    max_context = 15_360

    def _load_model(self) -> Any:
        if not self.params.get("accept_license"):
            raise PermissionError(
                f"{self.name}: the TimesFM-3 weights are under the TimesFM non-commercial "
                "licence v1.0 (academic research allowed; no commercial or production use). "
                "Review it on the model page and set `accept_license: true` in the config.")
        ckpt = self.params.get("checkpoint", "google/timesfm-3.0-pytorch")
        try:
            if self.params.get("backend", "torch") == "mlx":
                from timesfm3.mlx import TimesFM3Forecaster as Impl

                return Impl.from_pretrained(ckpt)
            from timesfm3 import TimesFM3Forecaster as Impl
        except ImportError as exc:
            raise ImportError(_INSTALL["timesfm3"]) from exc
        return Impl.from_pretrained(ckpt, device=_device(self.params))

    def _forecast(self, context, covariates, n):
        out = self._model.predict(context, horizon=n, past_future_covariates=covariates,
                                  return_quantiles=True)
        q = np.asarray(out.quantiles, dtype=float)  # (n, Q)
        return {t: q[:n, i] for i, t in enumerate(_levels(self._model, q.shape[-1]))}


class TiRex2Forecaster(_NativeCovariateForecaster):
    """NXAI TiRex-2 (zero-shot) with the weather forecast as known-future covariates."""

    arch = "tirex2"
    max_context = 2048

    def _load_model(self) -> Any:
        try:
            from tirex2 import load_model
        except ImportError as exc:
            raise ImportError(_INSTALL["tirex2"]) from exc
        return load_model(self.params.get("checkpoint", "NX-AI/TiRex-2"),
                          device=_device(self.params))

    @staticmethod
    def _series_type():
        from tirex2 import TimeseriesType

        return TimeseriesType

    def _forecast(self, context, covariates, n):
        import torch

        ts = self._series_type()(
            target=torch.as_tensor(context, dtype=torch.float32)[None, :],
            past_covariates=None,
            future_covariates=(None if covariates is None
                               else torch.as_tensor(covariates, dtype=torch.float32)),
        )
        fc = np.asarray(self._model.forecast([ts], prediction_length=n, output_type="numpy")[0],
                        dtype=float)  # (n_targets=1, Q, n)
        return {t: fc[0, i, :n] for i, t in enumerate(_levels(self._model, fc.shape[1]))}


_FACTORIES: dict[str, type[_NativeCovariateForecaster]] = {
    "timesfm3": TimesFM3Forecaster,
    "tirex2": TiRex2Forecaster,
}


def build_covariate_foundation(name: str, **kwargs: Any) -> Forecaster:
    """Build a TimesFM-3 or TiRex-2 wrapper; ``arch`` (default ``name``) picks the class."""
    arch = kwargs.pop("arch", name)
    if arch not in _FACTORIES:
        raise KeyError(f"No covariate-foundation wrapper for arch '{arch}' (model '{name}'). "
                       f"Known: {sorted(_FACTORIES)}.")
    return _FACTORIES[arch](name, arch=arch, **kwargs)


__all__ = ["COVARIATE_FOUNDATION_ARCHES", "TiRex2Forecaster", "TimesFM3Forecaster",
           "build_covariate_foundation"]
