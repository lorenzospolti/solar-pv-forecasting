"""In-context covariate regression ("XReg") for foundation models without covariates.

TiRex (tirex-ts 1.4) and TimesFM-2.5 (darts 0.47) forecast a single series and cannot
read the weather forecast. Google's answer for TimesFM-2.5 is XReg (timesfm package,
Oct 2025; ``forecast_with_covariates``): fit a linear model of the target on the
covariates over the context window, let the foundation model forecast what the
covariates leave unexplained, and add the two ("xreg + timesfm" mode). This module
re-implements that recipe in numpy, so it works for any univariate foundation model and
needs no JAX:

    1. fit   k_t = x_t' beta + r_t   by ridge regression on the daytime hours of the context;
    2. the foundation model forecasts the residual series r (quantiles r_q);
    3. k_q = x' beta + r_q           over the horizon (x' beta := 0 at night).

Deliberate differences from ``timesfm.utils.xreg_lib`` (stated in the thesis):

* the regression is fit on daytime hours only (``is_day``, a deterministic sun-position
  mask known in advance). At night k = 0 by construction, and a linear model cannot
  reproduce exact zeros, so night hours would only bend the fit;
* the intercept is not penalised (xreg_lib shrinks it with the slopes);
* optional feature ``kc_nwp``: the weather forecast's own clear-sky index, forecast GHI
  over the interval-mean clear-sky GHI of pvlib (the geometry behind ``p_clear``). The
  PV clear-sky index k is close to proportional to it, while k is far from linear in raw
  GHI: 400 W/m2 is an overcast noon but a clear early morning. It is a transformation
  of inputs every model already receives (forecast GHI and sun geometry), not new data.

Everything the regression reads is known at issue time: the observed k of the context
and the archived forecasts for the context and the horizon.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features import calendar_features

logger = logging.getLogger("pvfc.models.xreg")

#: values of the config key ``xreg.features``
FEATURE_SETS = ("covariates", "covariates+kc", "kc")

#: defaults for the config dict ``xreg: {...}``
XREG_DEFAULTS = {
    "features": "covariates+kc",  # chosen on the pre-test check of 10 Oct 2026 (COVARIATES.md)
    "ridge": 1.0,                 # penalty on the standardised slopes (not the intercept)
    "fit_hours": None,            # None: the foundation model's own context (input_chunk_length)
}

KC_CLIP = 1.5             # cloud enhancement rarely lifts hourly GHI above 1.5 x clear sky
KC_MIN_GHI_CLEAR = 20.0   # W/m2: below this (sunrise, sunset) the ratio is noise -> 0
MIN_ROWS_PER_FEATURE = 5  # fewer daytime rows than this per slope: no regression (x'b = mean)


def day_mask(frame: pd.DataFrame) -> pd.Series:
    """Daytime hours: ``is_day`` when the frame has it, else ``p_clear > 0``."""
    if "is_day" in frame.columns:
        return frame["is_day"].fillna(False).astype(bool)
    return (frame["p_clear"] > 1e-6).rename("is_day")


class ClearSkyGHICache:
    """Interval-mean clear-sky GHI per timestamp, computed once (pvlib is not free)."""

    def __init__(self, lat: float, lon: float, altitude: float = 0.0) -> None:
        self.lat, self.lon, self.altitude = lat, lon, altitude
        self._s = pd.Series(dtype=float)

    def get(self, index: pd.DatetimeIndex) -> pd.Series:
        from ..clearsky import clearsky_ghi_interval

        if index.tz is None:
            raise ValueError("kc_nwp needs a tz-aware index (the frames are UTC)")
        missing = index.difference(self._s.index)
        if len(missing):
            new = clearsky_ghi_interval(pd.DatetimeIndex(missing), self.lat, self.lon,
                                        self.altitude)
            self._s = pd.concat([self._s, new.astype(float)]).sort_index()
        return self._s.reindex(index)


def nwp_clearsky_index(frame: pd.DataFrame, ghi_clear: pd.Series) -> pd.Series:
    """Forecast GHI / clear-sky GHI, clipped to [0, KC_CLIP]; 0 when the sun is too low."""
    clear = ghi_clear.reindex(frame.index)
    kc = frame["ghi"] / clear.where(clear >= KC_MIN_GHI_CLEAR)
    return kc.clip(lower=0.0, upper=KC_CLIP).fillna(0.0).rename("kc_nwp")


def design_matrix(span: pd.DataFrame, weather_cols: list[str], features: str = "covariates",
                  ghi_cache: ClearSkyGHICache | None = None) -> pd.DataFrame:
    """Regressors over ``span`` (context + horizon rows), without the intercept.

    ``covariates``: p_clear + the weather columns + the four calendar terms, i.e. the
    future covariates Chronos-2 receives (darts_wrapper._covariate_series).
    ``covariates+kc``: the same plus ``kc_nwp``. ``kc``: ``kc_nwp`` alone.
    Without a ``ghi`` column (weather setting "none") ``kc_nwp`` cannot be formed and
    the set falls back to ``covariates``.
    """
    if features not in FEATURE_SETS:
        raise ValueError(f"xreg.features must be one of {FEATURE_SETS}, got {features!r}")
    use_kc = features in ("covariates+kc", "kc") and "ghi" in span.columns
    if features != "covariates" and not use_kc:
        logger.info("xreg: no forecast GHI in this frame; kc_nwp dropped (covariates only)")
        features = "covariates"
    parts = []
    if features in ("covariates", "covariates+kc"):
        cols = ["p_clear"] + [c for c in weather_cols if c in span.columns]
        parts += [span[cols], calendar_features(span.index)]
    if use_kc:
        if ghi_cache is None:
            raise ValueError("kc_nwp needs a ClearSkyGHICache")
        parts.append(nwp_clearsky_index(span, ghi_cache.get(span.index)).to_frame())
    X = pd.concat(parts, axis=1).astype(float)
    return X.ffill().bfill().fillna(0.0)


@dataclass
class RidgeFit:
    """A fitted ridge regression on standardised regressors (intercept unpenalised)."""

    columns: list[str]
    mean_: np.ndarray
    scale_: np.ndarray
    coef_: np.ndarray
    intercept_: float
    n_rows: int

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Z = (X[self.columns].to_numpy(dtype=float) - self.mean_) / self.scale_
        return Z @ self.coef_ + self.intercept_


def fit_ridge(y: pd.Series, X: pd.DataFrame, mask: pd.Series | np.ndarray | None = None,
              ridge: float = 1.0) -> RidgeFit:
    """Ridge regression of ``y`` on ``X`` over the rows in ``mask`` with finite values.

    The slopes are fit on z-scored regressors with penalty ``ridge``; the intercept is the
    mean of ``y`` (centred design), so it is not shrunk. With too few rows for a stable fit
    the slopes are zero and the fit predicts the mean (the residual is then the target).
    """
    yv = y.to_numpy(dtype=float)
    Xv = X.to_numpy(dtype=float)
    rows = np.isfinite(yv) & np.isfinite(Xv).all(axis=1)
    if mask is not None:
        rows &= np.asarray(mask, dtype=bool)
    p = Xv.shape[1]
    if rows.sum() < max(MIN_ROWS_PER_FEATURE * p, 2):
        ybar = float(np.nanmean(yv[rows])) if rows.any() else 0.0
        return RidgeFit(list(X.columns), np.zeros(p), np.ones(p), np.zeros(p), ybar,
                        int(rows.sum()))
    Xf, yf = Xv[rows], yv[rows]
    mean = Xf.mean(axis=0)
    scale = Xf.std(axis=0)
    scale[scale < 1e-8] = 1.0  # a constant column contributes nothing
    Z = (Xf - mean) / scale
    ybar = float(yf.mean())
    A = Z.T @ Z + float(ridge) * np.eye(p)
    coef = np.linalg.lstsq(A, Z.T @ (yf - ybar), rcond=None)[0]
    return RidgeFit(list(X.columns), mean, scale, coef, ybar, int(rows.sum()))


__all__ = ["FEATURE_SETS", "XREG_DEFAULTS", "ClearSkyGHICache", "RidgeFit", "day_mask",
           "design_matrix", "fit_ridge", "nwp_clearsky_index"]
