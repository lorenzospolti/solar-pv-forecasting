"""NWP forecast-weather feed for the realistic day-ahead experiment.

The HKUST campus-tower weather is *measured* (perfect foresight). For an honest
day-ahead setup the covariates a model sees must be what was *forecast* at issue
time. We source those from the Open-Meteo **Historical Forecast API** for the
HKUST coordinates and cache them locally (one parquet per year).

Honesty caveat (documented in the thesis plan): the Historical Forecast API is
stitched from the first hours of successive model runs — a best-available
near-analysis, not a frozen 24-h-lead forecast. The truly frozen feed (Previous
Runs API) only archives back to 2024, which does not overlap the 2021-2023 PV
window. We therefore use this feed as a "best-available archived forecast" that
sits between the measured oracle (upper bound) and a true day-ahead forecast, and
label it as such. Hold it fixed across all models.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("pvfc.weather")

OPEN_METEO_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
EXTERNAL_DIR = Path(__file__).resolve().parents[2] / "data" / "external"

# Open-Meteo hourly variable -> canonical frame column. The first four cover the
# canonical WEATHER_COLS; cloud_cover/direct/diffuse are kept as extra covariates.
_OM_RENAME = {
    "shortwave_radiation": "ghi",
    "temperature_2m": "temp_air",
    "wind_speed_10m": "wind_speed",
    "relative_humidity_2m": "rh",
    "cloud_cover": "cloud_cover",
    "direct_radiation": "ghi_direct",
    "diffuse_radiation": "ghi_diffuse",
}
NWP_VARS = list(_OM_RENAME)

# Open-Meteo convention: radiation variables are *backward-averaged* — the value
# stamped HH is the mean over [HH-1, HH] — while our frames are hour-beginning
# (resampled mean over [HH, HH+1]). These columns must therefore be shifted -1 h
# to align (verified on CYT: daytime corr(measured, forecast GHI) 0.66 -> 0.77,
# and the diurnal forecast peak moves from 13:00 back to 12:00 local).
# Instantaneous variables (temp/wind/rh/cloud_cover) are valid AT HH and stay put.
_BACKWARD_AVERAGED = ("ghi", "ghi_direct", "ghi_diffuse")


def _align_om_conventions(df: pd.DataFrame) -> pd.DataFrame:
    """Shift backward-averaged Open-Meteo columns onto hour-beginning labels."""
    out = df.copy()
    for col in _BACKWARD_AVERAGED:
        if col in out.columns:
            out[col] = out[col].shift(-1)
    return out


def _cache_path(lat: float, lon: float, year: int, cache_dir: Path) -> Path:
    return cache_dir / f"openmeteo_{lat:.4f}_{lon:.4f}_{year}.parquet"


def _fetch_year(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """One Open-Meteo request -> hourly UTC frame with canonical-renamed columns."""
    import requests

    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": ",".join(NWP_VARS), "timezone": "UTC",
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=60)
    r.raise_for_status()
    hourly = r.json()["hourly"]
    idx = pd.DatetimeIndex(pd.to_datetime(hourly["time"]), tz="UTC")
    data = {_OM_RENAME[v]: hourly[v] for v in NWP_VARS if v in hourly}
    return pd.DataFrame(data, index=idx).sort_index()


def fetch_open_meteo(lat: float, lon: float, start: str, end: str,
                     cache_dir: Path = EXTERNAL_DIR) -> pd.DataFrame:
    """Hourly archived-forecast weather for [start, end] (YYYY-MM-DD), tz-aware UTC.

    Caches one parquet per calendar year in ``cache_dir``, always holding the WHOLE
    year: a year file is fetched for 1 Jan-31 Dec and re-fetched if it covers less.
    (Up to ETL v2 a year file held only the span first requested: the 2021 file began
    on 4 May 2021, so SQ15, whose data start on 23 March, had no weather for 993
    hours.) Returns the concatenated frame sliced to [start, end].
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    y0, y1 = int(start[:4]), int(end[:4])
    parts = []
    for year in range(y0, y1 + 1):
        cp = _cache_path(lat, lon, year, cache_dir)
        y_start, y_end = f"{year}-01-01", f"{year}-12-31"
        if cp.exists():
            cached = pd.read_parquet(cp)
            if (len(cached) and cached.index.min() <= pd.Timestamp(y_start, tz="UTC")
                    and cached.index.max() >= pd.Timestamp(f"{y_end} 23:00", tz="UTC")):
                parts.append(cached)
                continue
            logger.info("Open-Meteo cache %s covers %s..%s only: re-fetching the year",
                        cp.name, cached.index.min(), cached.index.max())
        logger.info("Open-Meteo fetch %s %s..%s", year, y_start, y_end)
        df = _fetch_year(lat, lon, y_start, y_end)
        df.to_parquet(cp)
        parts.append(df)
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    # cache files keep the raw Open-Meteo stamps; alignment happens at load time
    return _align_om_conventions(out).loc[start:end]


def attach_nwp_forecast(frame: pd.DataFrame, lat: float, lon: float,
                        cache_dir: Path = EXTERNAL_DIR) -> pd.DataFrame:
    """Replace ``frame``'s weather columns with issue-time-aligned archived forecasts.

    ``power``/``p_clear``/``k``/``is_day`` are preserved (they derive from measured
    power and deterministic pvlib geometry, legitimately known ahead). EVERY other
    (measured) column is dropped — including tower rainfall/slp/visibility — so the
    nwp frame cannot leak measured future weather no matter which covariate list a
    model is configured with. The forecast columns are reindexed onto the frame's
    hourly UTC index.
    """
    start = frame.index.min().strftime("%Y-%m-%d")
    end = frame.index.max().strftime("%Y-%m-%d")
    nwp = fetch_open_meteo(lat, lon, start, end, cache_dir=cache_dir)
    nwp = nwp.reindex(frame.index)

    keep = ["power", "p_clear", "k", "is_day"]
    out = frame[[c for c in keep if c in frame.columns]].copy()
    for col in nwp.columns:
        out[col] = nwp[col].to_numpy()
    return out
