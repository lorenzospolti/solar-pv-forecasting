"""Clear-sky PV template via pvlib.

We need a clear-sky *power* envelope ``p_clear`` in the same units as measured
power so we can form the clear-sky index k = power / p_clear. Two routes (plan
correction #5): an empirical clear-day fit, or a physical pvlib chain. This
module provides the physical route plus a robust empirical fallback.

Key facts confirmed in research:
- ``pvlib.location.Location.get_clearsky(model='ineichen')`` returns clear-sky
  GHI/DNI/DHI and clips to 0 at night for all three models.
- ``pvlib.irradiance.clearsky_index`` is for *irradiance*, not PV power — so for
  PV power we scale a clear-sky irradiance proxy to the site's observed clear
  ceiling (empirical) or run a full ModelChain (physical).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def clearsky_ghi(index: pd.DatetimeIndex, lat: float, lon: float, altitude: float = 0.0,
                 model: str = "ineichen") -> pd.Series:
    """Clear-sky GHI (W/m^2) from pvlib. Index must be tz-aware."""
    import pvlib

    loc = pvlib.location.Location(lat, lon, altitude=altitude, tz=str(index.tz))
    cs = loc.get_clearsky(index, model=model)
    return cs["ghi"].rename("ghi_clear")


def solar_zenith(index: pd.DatetimeIndex, lat: float, lon: float, altitude: float = 0.0) -> pd.Series:
    import pvlib

    loc = pvlib.location.Location(lat, lon, altitude=altitude, tz=str(index.tz))
    return loc.get_solarposition(index)["apparent_zenith"].rename("zenith")


def daytime_mask(index: pd.DatetimeIndex, lat: float, lon: float, max_zenith: float = 85.0,
                 altitude: float = 0.0) -> pd.Series:
    """Boolean daytime mask (apparent zenith below ``max_zenith``)."""
    return (solar_zenith(index, lat, lon, altitude) < max_zenith).rename("is_day")


def empirical_clearsky_power(
    power: pd.Series, ghi_clear: pd.Series, capacity: float | None = None,
    quantile: float = 0.95, fit_mask: pd.Series | None = None,
) -> pd.Series:
    """Scale clear-sky GHI to a clear-sky *power* ceiling per (month, hour).

    For each calendar (month, hour) bucket we take a high quantile of the ratio
    power/ghi_clear on sunny samples as the effective performance ratio, then
    p_clear = ratio * ghi_clear. This absorbs azimuth/tilt/efficiency without a
    full physical model and stays robust to outliers. A pvlib ModelChain can
    replace this later for a physics-based comparison.

    ``fit_mask`` (boolean, aligned with ``power``) restricts the rows used to FIT
    the performance-ratio table; the template is still APPLIED to the full index.
    Pass the initial training window so the target transform never sees test-period
    data (leakage guard) — the template is then identical for every fold.
    """
    df = pd.DataFrame({"power": power, "ghi_clear": ghi_clear})
    if fit_mask is not None:
        df = df[fit_mask.reindex(df.index).fillna(False).astype(bool)]
    df = df[df["ghi_clear"] > 1.0]
    ratio = (df["power"] / df["ghi_clear"]).clip(lower=0)
    key = [df.index.month, df.index.hour]
    perf = ratio.groupby(key).quantile(quantile)
    idx_key = list(zip(power.index.month, power.index.hour))
    pr = pd.Series([perf.get(k, np.nan) for k in idx_key], index=power.index)
    pr = pr.ffill().bfill().fillna(ratio.median() if len(ratio) else 0.0)
    p_clear = (pr * ghi_clear).clip(lower=0.0)
    if capacity:
        p_clear = p_clear.clip(upper=capacity)
    return p_clear.rename("p_clear")
