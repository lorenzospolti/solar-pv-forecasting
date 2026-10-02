"""External foundation-model wrappers (Moirai-2, TTM / TinyTimeMixer).

These foundation models are **not** in darts (even 0.44), so they need their
own glue. Each wrapper lives behind a *lazy* import of its optional dep:
``uni2ts`` for Moirai-2 (Salesforce) and ``granite-tsfm`` for TTM (IBM). The
existing darts-native roster (Chronos-2, TimesFM-2.5, TiRex) keeps working
without these deps installed.

Architecture vs. label (mirror of `darts_wrapper.py`)
----------------------------------------------------
A model spec has a config ``name`` (the results-row label) and an ``arch``
(which external class to build, e.g. ``ttm``). ``arch`` defaults to ``name``,
so two configurations of one arch score as distinct rows.

Zero-shot only. Neither wrapper has a fine-tuning loop, so ``finetune: true``
raises at ``fit`` instead of producing a zero-shot forecast under a fine-tuned
label (v0.5 shipped ``moirai2_ft`` / ``ttm_ft`` entries that did exactly that).
Moirai-2 is out of the thesis scope; TTM is an optional zero-shot stretch.

Runner contract: ``fit(train)`` once per fold (loads the checkpoint);
``predict(history, future)`` rolls a fresh, growing ``history`` per test day,
as for the darts foundation models.

Test mocking
------------
The two methods ``_load_model`` (resolves the optional dep + checkpoint) and
``_forecast`` (one window of in-context inference) are isolated so tests can
monkeypatch them without installing the real libraries.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ..evaluation import DEFAULT_QUANTILES, enforce_monotone_quantiles
from ..normalize import fill_index_gaps, reconstruct_power
from .base import Forecaster

logger = logging.getLogger("pvfc.models.external")

#: install-instructions shown when the lazy import fails
_INSTALL: dict[str, str] = {
    "moirai2": (
        "Moirai-2 (Salesforce) is not a darts class. It needs the optional "
        "`uni2ts` library. Install with:\n"
        "    pip install 'uni2ts>=1.2'\n"
        "then re-run the entry with `name: moirai2` (zero-shot only). HF "
        "checkpoint defaults to `Salesforce/moirai-2.0-R-small` and is "
        "downloaded on first use."
    ),
    "ttm": (
        "TTM / TinyTimeMixer (IBM) is not a darts class. It needs the optional "
        "`granite-tsfm` library. Install with:\n"
        "    pip install 'granite-tsfm[notebooks]>=0.2.16'\n"
        "then re-run the entry with `name: ttm` (zero-shot only). HF checkpoint "
        "defaults to `ibm-granite/granite-timeseries-ttm-r2` and is downloaded "
        "on first use."
    ),
}

#: arches handled by this module (mirror of darts_wrapper._FOUNDATION)
EXTERNAL_FOUNDATION_ARCHES: frozenset[str] = frozenset({"moirai2", "ttm"})


class _ExternalFoundationForecaster(Forecaster):
    """Base class for the external foundation wrappers.

    Both Moirai-2 and TTM follow the same shape:
      1. lazy-import their dep in ``_load_model`` (raise with install hint on miss);
      2. ``fit`` only loads the checkpoint (zero-shot); ``finetune=True`` raises
         through ``_finetune`` until a subclass implements a real loop;
      3. ``predict`` builds a clear-sky-index context window and asks
         ``_forecast`` for (point, quantiles).

    Subclasses implement ``_load_model``, ``_forecast``, and optionally
    ``_finetune``.
    """

    arch: str = ""

    def __init__(self, name: str, arch: str | None = None, **params: Any) -> None:
        self.name = name
        self.arch = arch or type(self).arch or name
        self.params = dict(params)
        self.supports_quantiles = bool(params.get("quantile", False))
        self.quantiles: list[float] | None = (
            [float(q) for q in DEFAULT_QUANTILES] if self.supports_quantiles else None
        )
        # cap how far back we feed the model at predict time (keeps in-context
        # inference cheap and mirrors the darts-foundation path)
        self.context_days: int = int(params.get("context_days", 60))
        self.finetune: bool | dict = params.get("finetune", False)
        self.input_chunk_length: int = int(params.get("input_chunk_length", 512))
        self.output_chunk_length: int = int(params.get("output_chunk_length", 24))
        self._model: Any = None
        self._fitted: bool = False

    # ------------------------------------------------------------------ #
    # Lazy import + load. Tests monkeypatch this to dodge the real dep.  #
    # ------------------------------------------------------------------ #
    def _load_model(self) -> Any:  # pragma: no cover - subclasses implement
        raise NotImplementedError

    def _finetune(self, train: pd.DataFrame) -> None:
        """No fine-tuning loop is implemented: refuse rather than mislabel.

        A results row labelled ``*_ft`` must come from a fine-tuned model; silently
        running zero-shot under that label would corrupt the comparison.
        """
        raise NotImplementedError(
            f"{self.name}: fine-tuning is not implemented for arch '{self.arch}'. "
            f"Run it zero-shot (drop `finetune: true`) or implement "
            f"{type(self).__name__}._finetune()."
        )

    # ------------------------------------------------------------------ #
    # Forecaster contract                                                #
    # ------------------------------------------------------------------ #
    def fit(self, train: pd.DataFrame) -> "Forecaster":
        # Building the model triggers the lazy import; a missing dep raises
        # ImportError with install instructions here (before any predict call).
        self._model = self._load_model()
        if self.finetune:
            self._finetune(train)
        self._fitted = True
        return self

    def _context_window(self, history: pd.DataFrame) -> np.ndarray:
        """Last ``context_days`` of clear-sky-index history as a float32 array,
        gaps filled by ``fill_index_gaps`` and clipped to ``input_chunk_length``."""
        hist = history
        if self.context_days:
            cutoff = history.index.max() - pd.Timedelta(days=self.context_days)
            hist = history.loc[history.index >= cutoff]
        s = hist["k"].astype("float32").asfreq("h")
        day = hist["is_day"].astype(bool) if "is_day" in hist else None
        arr = fill_index_gaps(s, day).to_numpy(dtype="float32")
        if self.input_chunk_length and len(arr) > self.input_chunk_length:
            arr = arr[-self.input_chunk_length:]
        return arr

    def _forecast(
        self, context: np.ndarray, n: int,
    ) -> tuple[np.ndarray, dict[float, np.ndarray] | None]:
        """One window of in-context inference.

        Returns
        -------
        point : (n,) float array — median / mean prediction in k-space.
        quantiles : {tau: (n,) array} or None — populated only when
            ``supports_quantiles``.
        """
        raise NotImplementedError  # pragma: no cover - subclasses implement

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: predict() called before fit()")
        n = len(future)
        context = self._context_window(history)
        k_point, k_q = self._forecast(context, n)
        if len(k_point) < n:
            k_point = np.pad(k_point, (0, n - len(k_point)), mode="edge")
        k_point = np.asarray(k_point[:n], dtype="float32")

        p_clear = future["p_clear"]
        out = pd.DataFrame(index=future.index)
        out["y_hat"] = reconstruct_power(
            pd.Series(k_point, index=future.index), p_clear,
        ).to_numpy()

        if self.supports_quantiles and k_q:
            # right-trim / pad each quantile band, then enforce monotonicity
            bands: dict[float, np.ndarray] = {}
            for tau, arr in k_q.items():
                arr = np.asarray(arr, dtype="float32")
                if len(arr) < n:
                    arr = np.pad(arr, (0, n - len(arr)), mode="edge")
                bands[float(round(tau, 2))] = arr[:n]
            bands = enforce_monotone_quantiles(bands)
            for tau, arr in bands.items():
                out[f"q{tau}"] = reconstruct_power(
                    pd.Series(arr, index=future.index), p_clear,
                ).to_numpy()
        return out


# --------------------------------------------------------------------- #
# Moirai-2 (Salesforce, via uni2ts)                                     #
# --------------------------------------------------------------------- #
class Moirai2Forecaster(_ExternalFoundationForecaster):
    """Salesforce Moirai-2, zero-shot (uni2ts >= 1.2). Out of the thesis scope.

    Notes
    -----
    Moirai-2 is a probabilistic foundation model — quantiles come from sampling
    its output distribution and taking empirical quantiles. The HF checkpoint
    defaults to ``Salesforce/moirai-2.0-R-small`` (small/base/large variants;
    `small` fits 16 GB unified memory on an M4).
    """

    arch = "moirai2"

    @staticmethod
    def _import_deps() -> tuple[Any, Any]:
        """Isolate the actual `from uni2ts...` import so tests can patch this
        method to simulate a missing dep."""
        # uni2ts >= 1.2 exposes a v2 namespace; fall back to v1 names if the
        # user has an older drop.
        try:
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        except ImportError:
            from uni2ts.model.moirai import (  # type: ignore[no-redef]
                MoiraiForecast as Moirai2Forecast,
                MoiraiModule as Moirai2Module,
            )
        return Moirai2Forecast, Moirai2Module

    def _load_model(self) -> Any:
        try:
            Moirai2Forecast, Moirai2Module = self._import_deps()
        except ImportError as exc:
            raise ImportError(_INSTALL["moirai2"]) from exc

        size = self.params.get("size", "small")
        ckpt = self.params.get("checkpoint", f"Salesforce/moirai-2.0-R-{size}")
        module = Moirai2Module.from_pretrained(ckpt)
        return Moirai2Forecast(
            module=module,
            prediction_length=self.output_chunk_length,
            context_length=self.input_chunk_length,
            patch_size=self.params.get("patch_size", "auto"),
            num_samples=int(self.params.get("num_samples", 100)),
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

    def _forecast(
        self, context: np.ndarray, n: int,
    ) -> tuple[np.ndarray, dict[float, np.ndarray] | None]:
        import torch

        past = torch.as_tensor(context, dtype=torch.float32).reshape(1, -1, 1)
        with torch.no_grad():
            # uni2ts forecast returns (B, num_samples, prediction_length, target_dim)
            samples = self._model(past_target=past).cpu().numpy()
        samples = samples[0, :, :n, 0]  # (S, n)
        point = samples.mean(axis=0)
        q: dict[float, np.ndarray] | None = None
        if self.supports_quantiles and self.quantiles:
            q = {float(round(tau, 2)): np.quantile(samples, tau, axis=0)
                 for tau in self.quantiles}
        return point.astype("float32"), q


# --------------------------------------------------------------------- #
# TTM / TinyTimeMixer (IBM, via granite-tsfm)                           #
# --------------------------------------------------------------------- #
class TTMForecaster(_ExternalFoundationForecaster):
    """IBM TinyTimeMixer (TTM-R2), zero-shot via granite-tsfm (optional stretch).

    TTM is tiny (1-5 M params) and CPU-friendly, which is the whole reason it
    earns a slot in a consumer-hardware comparison: it slots in next to the
    much larger transformer foundation models (Chronos-2, TimesFM-2.5,
    Moirai-2) without needing a GPU. Default checkpoint:
    ``ibm-granite/granite-timeseries-ttm-r2``.

    TTM is a *deterministic* model. When ``quantile=True`` is requested we
    return a Gaussian-bracketed band of constant width ``residual_std`` around
    the point forecast. It is NOT a calibrated probabilistic forecast, so keep
    ``quantile: false`` for TTM in the thesis runs (point metrics only).
    """

    arch = "ttm"

    def __init__(self, name: str, arch: str | None = None, **params: Any) -> None:
        super().__init__(name, arch=arch, **params)
        # default context length tuned to TTM-R2 (512 is one of the released CLs)
        self.input_chunk_length = int(params.get("input_chunk_length", 512))
        self.output_chunk_length = int(params.get("output_chunk_length", 24))
        self._residual_std: float = float(params.get("residual_std", 0.1))

    @staticmethod
    def _import_deps() -> Any:
        """Isolate the actual `from tsfm_public...` import so tests can patch
        this to simulate a missing dep. Returns a callable
        ``loader(checkpoint, context_length, prediction_length)``."""
        try:
            from tsfm_public.toolkit.get_model import get_model

            def loader(checkpoint, context_length, prediction_length):
                return get_model(
                    model_path=checkpoint,
                    context_length=context_length,
                    prediction_length=prediction_length,
                )
            return loader
        except ImportError:
            # older granite-tsfm exposes the class directly
            from tsfm_public.models.tinytimemixer import (  # type: ignore[import-not-found]
                TinyTimeMixerForPrediction,
            )

            def loader(checkpoint, context_length, prediction_length):
                return TinyTimeMixerForPrediction.from_pretrained(
                    checkpoint,
                    context_length=context_length,
                    prediction_length=prediction_length,
                )
            return loader

    def _load_model(self) -> Any:
        try:
            loader = self._import_deps()
        except ImportError as exc:
            raise ImportError(_INSTALL["ttm"]) from exc
        checkpoint = self.params.get(
            "checkpoint", "ibm-granite/granite-timeseries-ttm-r2",
        )
        return loader(
            checkpoint=checkpoint,
            context_length=self.input_chunk_length,
            prediction_length=self.output_chunk_length,
        )

    def _forecast(
        self, context: np.ndarray, n: int,
    ) -> tuple[np.ndarray, dict[float, np.ndarray] | None]:
        import torch

        # TTM expects (batch, context_length, channels); univariate -> 1 channel
        past = torch.as_tensor(context, dtype=torch.float32).reshape(1, -1, 1)
        if past.shape[1] < self.input_chunk_length:
            pad = self.input_chunk_length - past.shape[1]
            past = torch.nn.functional.pad(past, (0, 0, pad, 0), mode="replicate")
        with torch.no_grad():
            out = self._model(past_values=past)
        # granite-tsfm returns an object with `.prediction_outputs` (B, P, C)
        preds = getattr(out, "prediction_outputs", out)
        if hasattr(preds, "cpu"):
            preds = preds.cpu().numpy()
        preds = np.asarray(preds).reshape(-1)[:n]
        point = preds.astype("float32")

        q: dict[float, np.ndarray] | None = None
        if self.supports_quantiles and self.quantiles:
            # Symmetric Gaussian band around the point; widens with horizon.
            from math import sqrt
            sigma = max(self._residual_std, 1e-3)
            z = {0.1: -1.2816, 0.2: -0.8416, 0.3: -0.5244, 0.4: -0.2533,
                 0.5: 0.0, 0.6: 0.2533, 0.7: 0.5244, 0.8: 0.8416, 0.9: 1.2816}
            horizon_growth = np.sqrt(np.arange(1, n + 1, dtype="float32"))
            q = {}
            for tau in self.quantiles:
                tau_r = round(float(tau), 2)
                z_tau = z.get(tau_r, 0.0)
                q[tau_r] = point + z_tau * sigma * horizon_growth / sqrt(n)
        return point, q


# --------------------------------------------------------------------- #
# Registry — consulted by build_darts_model() / models.build()          #
# --------------------------------------------------------------------- #
_EXTERNAL_FACTORIES: dict[str, type[_ExternalFoundationForecaster]] = {
    "moirai2": Moirai2Forecaster,
    "ttm": TTMForecaster,
}


def build_external_foundation(name: str, **kwargs: Any) -> Forecaster:
    """Build a Moirai-2 or TTM wrapper from a config entry.

    ``name`` is the results-row label; ``arch`` (kwarg, default ``name``)
    selects the wrapper class — so two configurations of one arch can coexist
    as separately-scored rows (e.g. ``name: ttm_ctx1024, arch: ttm``).
    """
    arch = kwargs.pop("arch", name)
    if arch not in _EXTERNAL_FACTORIES:
        raise KeyError(
            f"No external-foundation wrapper for arch '{arch}' (from model "
            f"'{name}'). Known: {sorted(_EXTERNAL_FACTORIES)}."
        )
    cls = _EXTERNAL_FACTORIES[arch]
    return cls(name, arch=arch, **kwargs)


__all__ = [
    "EXTERNAL_FOUNDATION_ARCHES",
    "Moirai2Forecaster",
    "TTMForecaster",
    "build_external_foundation",
]
