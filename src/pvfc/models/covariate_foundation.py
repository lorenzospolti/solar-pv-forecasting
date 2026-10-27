"""Foundation models that read the weather forecast natively (known-future covariates).

Two models released after darts 0.47 fixed its roster (4 Sep 2026), so they get their own
glue, like Moirai-2 and TTM in ``external_foundation.py``:

* **TimesFM-3** (Google, Aug 2026): ``python -m pip install 'timesfm[torch]==3.0.2'``,
  checkpoint ``google/timesfm-3.0-pytorch`` (331M parameters, all used; context up to
  15,360 steps; past-only and past-and-future covariates). Torch backend only, on MPS or
  CPU: timesfm 3.0.2's MLX backend differs from torch, so ``backend: mlx`` raises. The
  ``predict()`` flags are stated in ``TimesFM3Forecaster.PREDICT_FLAGS`` (the package
  defaults). The weights are under the TimesFM non-commercial licence v1.0, which allows
  academic research but not commercial or production use: set ``accept_license: true`` in
  the config once you have reviewed it. Not gated.
* **TiRex-2** (NXAI, Jul 2026): ``python -m pip install 'tirex-2==0.3.1'``, checkpoint
  ``NX-AI/TiRex-2`` (82.5M parameters, all used when covariates are read; context 2,048
  steps; past and future-known covariates). Apache-2.0, not gated. It forecasts eagerly
  (``torch.compiler.set_stance("force_eager")``): tirex-2 compiles its ``ResidualBlock``,
  and inductor cannot build the Metal kernels with the conda-forge torch 2.10.

``revision`` (a commit sha) pins a checkpoint; the definitive run pins both in
``configs/experiment.yaml``. They are downloaded by hash, with no ``refs/main``, so an
offline load needs it. On MPS the peak driver memory is logged once per 30 forecasts, and
TimesFM-3 releases the MPS allocator cache after every forecast. Before the load, ``fit()``
releases the cache earlier models in the process left, so the lines count this model only.

Both models receive the covariates Chronos-2 receives (``p_clear``, the weather columns and
the four calendar terms; ``darts_wrapper._covariate_series``), as covariates known over the
context and the horizon. ``covariate_features: covariates+kc`` adds the forecast's clear-sky
index (``pvfc.models.xreg.nwp_clearsky_index``). Zero-shot only: ``finetune`` raises.

Test mocking: ``_load_model`` (dependency + checkpoint) and ``_forecast`` (one window of
inference) are isolated, as in ``external_foundation.py``; ``fit`` and ``_after_forecast``
touch ``torch.mps`` only when the model runs on MPS.
"""
from __future__ import annotations

import gc
import logging
import math
import os
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
    "timesfm3": ("TimesFM-3 needs the `timesfm` package 3.0.2 (torch backend):\n"
                 "    python -m pip install 'timesfm[torch]==3.0.2'"),
    "tirex2": ("TiRex-2 needs the `tirex-2` package 0.3.1:\n"
               "    python -m pip install 'tirex-2==0.3.1'"),
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
    #: release the MPS allocator cache after every forecast. TimesFM-3 rounds each context
    #: up to a multiple of 32 (timesfm3_forecaster.py:654-661), so most test days bring a
    #: new shape, and the allocator keeps ~0.2-0.3 GB per shape that nothing frees (its
    #: try_gc, :119-133, empties only the CUDA cache). DEFINITIVE_RUN.md section 4.1.
    release_mps_cache: bool = False
    #: forecasts per memory log line: one 30-day test window
    MEMORY_LOG_EVERY = 30

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
        self._on_mps = False
        self._mps_peak = 0
        self._n_forecasts = 0

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
        self._on_mps = _device(self.params).split(":")[0] == "mps"  # "mps" or "mps:0"
        if self._model is None:  # zero-shot: the checkpoint is loaded once, as in darts_wrapper
            if self._on_mps:
                # the driver memory is process-wide: free the cache earlier models in this
                # process left (run.py --all-models) before the weights can land in it, so
                # the memory lines count this model only
                import torch

                gc.collect()
                torch.mps.empty_cache()
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

    def _after_forecast(self) -> None:
        """On MPS: track the driver memory, release the cache if asked, log every 30."""
        if not self._on_mps:
            return
        import torch

        self._mps_peak = max(self._mps_peak, torch.mps.driver_allocated_memory())
        if self.release_mps_cache:
            torch.mps.empty_cache()
        self._n_forecasts += 1
        if self._n_forecasts % self.MEMORY_LOG_EVERY == 0:
            logger.info("%s: MPS driver memory, peak over the last %d forecasts: %.2f GB",
                        self.name, self.MEMORY_LOG_EVERY, self._mps_peak / 2**30)
            self._mps_peak = 0

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError(f"{self.name}: predict() called before fit()")
        n = len(future)
        context, cov = self._inputs(history, future)
        raw = self._forecast(context, cov, n)
        self._after_forecast()
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
    max_context = 15_360  # timesfm3/torch/timesfm3_forecaster.py:32 (_MAX_CONTEXT_LENGTH)
    release_mps_cache = True
    #: timesfm 3.0.2 predict() defaults (:433-446), passed explicitly so the run states them.
    #: Google's benchmark evaluator turns on use_symmetric_averaging and make_positive
    #: (timesfm3/torch/evaluator.py:46-58); neither is used here (DEFINITIVE_RUN.md section 4.1).
    PREDICT_FLAGS: dict[str, Any] = {"use_symmetric_averaging": False, "make_positive": False,
                                     "sort_quantiles": True, "use_znorm": False,
                                     "padding_mode": "none"}

    def _load_model(self) -> Any:
        if not self.params.get("accept_license"):
            raise PermissionError(
                f"{self.name}: the TimesFM-3 weights are under the TimesFM non-commercial "
                "licence v1.0 (academic research allowed; no commercial or production use). "
                "Review it on the model page and set `accept_license: true` in the config.")
        if self.params.get("backend", "torch") != "torch":
            raise ValueError(f"{self.name}: only the torch backend is supported (timesfm 3.0.2's "
                             "MLX backend differs from torch: google-research/timesfm#506)")
        ckpt = str(self.params.get("checkpoint", "google/timesfm-3.0-pytorch"))
        if os.path.isfile(os.path.expanduser(ckpt)):
            raise ValueError(f"{self.name}: pass a repo id or a snapshot directory, not a weights "
                             "file (a file path builds the model from package defaults, e.g. "
                             "use_rope_var=True; the checkpoint's config.json has false)")
        try:
            from timesfm3 import TimesFM3Forecaster as Impl
        except ImportError as exc:
            raise ImportError(_INSTALL["timesfm3"]) from exc
        return Impl.from_pretrained(ckpt, device=_device(self.params),
                                    revision=self.params.get("revision"))

    def _forecast(self, context, covariates, n):
        out = self._model.predict(context, horizon=n, past_future_covariates=covariates,
                                  return_quantiles=True, **self.PREDICT_FLAGS)
        q = np.asarray(out.quantiles, dtype=float)  # (n, 9): quantiles only, no mean column
        if q.ndim != 2 or q.shape[0] < n:
            raise ValueError(f"{self.name}: unexpected quantile output shape {q.shape}")
        return {t: q[:n, i] for i, t in enumerate(_levels(self._model, q.shape[-1]))}


class TiRex2Forecaster(_NativeCovariateForecaster):
    """NXAI TiRex-2 (zero-shot) with the weather forecast as known-future covariates."""

    arch = "tirex2"
    max_context = 2048  # context_len in the checkpoint's model-config.yaml

    def _load_model(self) -> Any:
        try:
            from tirex2 import load_model
        except ImportError as exc:
            raise ImportError(_INSTALL["tirex2"]) from exc
        rev = self.params.get("revision")
        return load_model(self.params.get("checkpoint", "NX-AI/TiRex-2"),
                          device=_device(self.params),
                          hf_kwargs={"revision": rev} if rev else None)

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
        # tirex-2 0.3.1 compiles ResidualBlock (@torch.compile, residual_block.py:12); with the
        # conda-forge torch 2.10 build, inductor cannot build Metal kernels on MPS
        # ('c10/metal/utils.h' not found). Eager on every device: one code path.
        with torch.compiler.set_stance("force_eager"):
            fc = self._model.forecast([ts], prediction_length=n, output_type="numpy")[0]
        fc = np.asarray(fc, dtype=float)  # (n_targets=1, Q, n)
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
