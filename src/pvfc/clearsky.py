"""Clear-sky PV template via pvlib.

We need a clear-sky *power* envelope ``p_clear`` in the same units as measured
power so we can form the clear-sky index k = power / p_clear. Two routes (plan
correction #5): an empirical clear-day fit, or a physical pvlib chain. This
module provides the empirical route: the Brick metadata do list a tilt and azimuth
per system, but several arrays face two directions or follow curved roofs
("Mixed"), and the empirical envelope also absorbs shading and losses.

Interval convention (ETL v2): station frames are hourly with interval-START
labels — the value stamped HH:00 is the mean over [HH:00, HH+1:00). Every solar
quantity is therefore evaluated for that interval: clear-sky GHI as the interval
*mean* (``clearsky_ghi_interval``) and sun position at the interval midpoint
(HH:30), never at the instant HH:00.

Key facts confirmed in research:
- ``pvlib.location.Location.get_clearsky(model='ineichen')`` returns clear-sky
  GHI/DNI/DHI and clips to 0 at night for all three models.
- ``pvlib.irradiance.clearsky_index`` is for *irradiance*, not PV power — so for
  PV power we scale a clear-sky irradiance proxy to the site's observed clear
  ceiling (empirical) or run a full ModelChain (physical).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def clearsky_ghi(index: pd.DatetimeIndex, lat: float, lon: float, altitude: float = 0.0,
                 model: str = "ineichen") -> pd.Series:
    """Instantaneous clear-sky GHI (W/m^2) from pvlib. Index must be tz-aware."""
    import pvlib

    loc = pvlib.location.Location(lat, lon, altitude=altitude, tz=str(index.tz))
    cs = loc.get_clearsky(index, model=model)
    return cs["ghi"].rename("ghi_clear")


def clearsky_ghi_interval(index: pd.DatetimeIndex, lat: float, lon: float,
                          altitude: float = 0.0, interval: str = "1h", n_sub: int = 12,
                          model: str = "ineichen") -> pd.Series:
    """Mean clear-sky GHI over each interval ``[t, t + interval)`` (interval-start labels).

    Averages ``n_sub`` evenly spaced instants, which is what an interval-mean power
    meter integrates. A single instant mis-states the sunrise and sunset hours,
    when the sun is up for only part of the interval.
    """
    width = pd.Timedelta(interval).value  # ns
    offsets = ((np.arange(n_sub) + 0.5) / n_sub * width).astype("int64")
    fine = pd.to_datetime((index.asi8[:, None] + offsets[None, :]).ravel(), utc=True)
    ghi = clearsky_ghi(pd.DatetimeIndex(fine).tz_convert(index.tz), lat, lon, altitude, model)
    return pd.Series(ghi.to_numpy().reshape(len(index), n_sub).mean(axis=1),
                     index=index, name="ghi_clear")


def solar_zenith(index: pd.DatetimeIndex, lat: float, lon: float, altitude: float = 0.0) -> pd.Series:
    import pvlib

    loc = pvlib.location.Location(lat, lon, altitude=altitude, tz=str(index.tz))
    return loc.get_solarposition(index)["apparent_zenith"].rename("zenith")


def daytime_mask(index: pd.DatetimeIndex, lat: float, lon: float, max_zenith: float = 85.0,
                 altitude: float = 0.0) -> pd.Series:
    """Boolean daytime mask (apparent zenith below ``max_zenith``) at the given instants.

    Callers with interval-start frames pass the interval midpoints.
    """
    return (solar_zenith(index, lat, lon, altitude) < max_zenith).rename("is_day")


def sun_coordinates(index: pd.DatetimeIndex, lon: float) -> tuple[np.ndarray, np.ndarray]:
    """Solar declination and hour angle (degrees) at ``index`` (tz-aware).

    Together they fix the sun's position for a given site. Unlike the calendar
    month, they repeat symmetrically around the solstices: early February and
    early November share a declination, so do mid-April and late August.
    """
    import pvlib

    doy = index.dayofyear.to_numpy()
    decl = np.degrees(pvlib.solarposition.declination_spencer71(doy))
    eot = pvlib.solarposition.equation_of_time_spencer71(doy)
    ha = np.asarray(pvlib.solarposition.hour_angle(index, lon, eot), dtype=float)
    # pvlib does not wrap: on a UTC index, 05:00-07:59 Hong Kong time came out at
    # 248-278 deg instead of -112..-82 (ETL v2 put every early-morning sample in the
    # template's post-sunset bin). Wrap to [-180, 180): morning < 0 < afternoon.
    ha = (ha + 180.0) % 360.0 - 180.0
    return np.asarray(decl, dtype=float), ha


def _fill_from_neighbours(grid: np.ndarray, max_iter: int = 500) -> np.ndarray:
    """Fill NaN cells with the mean of their finite 8-neighbours, repeatedly.

    Filling spreads outward from the fitted cells, so an empty cell takes the
    value of the nearest fitted sun positions, never of the previous timestamp.
    """
    g = grid.copy()
    for _ in range(max_iter):
        nan = ~np.isfinite(g)
        if not nan.any():
            break
        pad = np.pad(g, 1, constant_values=np.nan)
        rows, cols = g.shape
        neigh = np.stack([pad[1 + dy:1 + dy + rows, 1 + dx:1 + dx + cols]
                          for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN neighbourhoods
            mean = np.nanmean(neigh, axis=0)
        fill = nan & np.isfinite(mean)
        if not fill.any():
            break
        g[fill] = mean[fill]
    return g


def empirical_clearsky_power(
    power: pd.Series, ghi_clear: pd.Series, declination: np.ndarray, hour_angle: np.ndarray,
    capacity: float | None = None, quantile: float = 0.95, fit_mask: pd.Series | None = None,
    decl_step: float = 4.0, ha_step: float = 15.0, min_count: int = 8, min_ghi: float = 50.0,
) -> pd.Series:
    """Clear-sky *power* envelope ``p_clear = PR(sun position) * ghi_clear``.

    ``PR = power / ghi_clear`` is an effective performance ratio that absorbs the
    array's tilt, azimuth, efficiency and fixed shading without a physical model.
    It is binned by SUN POSITION (``declination`` x ``hour_angle``, see
    ``sun_coordinates``), not by calendar month:

    * a month missing from the fit window borrows from its mirror month (same
      declination on the other side of the solstice), so the template has no holes
      (ETL v1 binned by month: no February data -> p_clear = 0 all February);
    * each bin pools two separate times of year, so one unusually cloudy month no
      longer drags its whole template down;
    * per bin we take a high ``quantile`` of PR (the clear-sky envelope) when the bin
      holds at least ``min_count`` samples; empty bins are filled from neighbouring
      bins, the grid is lightly smoothed (3x3) and PR is interpolated bilinearly at
      each timestamp, so ``p_clear`` varies smoothly in time (no month-boundary steps).

    ``fit_mask`` (boolean, aligned with ``power``) restricts the rows used to FIT
    the PR grid; the template is still APPLIED to the full index. Pass the initial
    training window so the target transform never sees test-period data (leakage
    guard). Samples with ``ghi_clear < min_ghi`` (very low sun) are not fitted.
    """
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import uniform_filter

    idx = power.index
    p = power.to_numpy(dtype=float)
    g = np.asarray(ghi_clear, dtype=float)
    decl = np.asarray(declination, dtype=float)
    ha = np.asarray(hour_angle, dtype=float)

    fit = np.isfinite(p) & (g >= min_ghi)
    if fit_mask is not None:
        fit &= fit_mask.reindex(idx).fillna(False).to_numpy(dtype=bool)

    d_edges = np.arange(-24.0, 24.0 + decl_step, decl_step)  # declination stays within +-23.45
    h_edges = np.arange(-120.0, 120.0 + ha_step, ha_step)    # daylight at 22 N spans ~ +-105
    shape = (len(d_edges) - 1, len(h_edges) - 1)
    di = np.clip(np.digitize(decl, d_edges) - 1, 0, shape[0] - 1)
    hi = np.clip(np.digitize(ha, h_edges) - 1, 0, shape[1] - 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.clip(p / g, 0.0, None)
    grid = np.full(shape, np.nan)
    if fit.any():
        grouped = pd.Series(ratio[fit]).groupby([di[fit], hi[fit]])
        q, n = grouped.quantile(quantile), grouped.size()
        for (d, h), val in q[n >= min_count].items():
            grid[d, h] = val
    if not np.isfinite(grid).any():  # too little data for any bin: one global ratio
        grid[:] = np.nanquantile(ratio[fit], quantile) if fit.any() else 0.0
    grid = uniform_filter(_fill_from_neighbours(grid), size=3, mode="nearest")

    d_c = (d_edges[:-1] + d_edges[1:]) / 2
    h_c = (h_edges[:-1] + h_edges[1:]) / 2
    pts = np.column_stack([np.clip(decl, d_c[0], d_c[-1]), np.clip(ha, h_c[0], h_c[-1])])
    pr = RegularGridInterpolator((d_c, h_c), grid)(pts)
    p_clear = np.clip(pr, 0.0, None) * np.clip(g, 0.0, None)
    if capacity:
        p_clear = np.minimum(p_clear, capacity)
    return pd.Series(p_clear, index=idx, name="p_clear")
