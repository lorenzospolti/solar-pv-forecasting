"""Data loading & ETL for the HKUST rooftop PV dataset (primary).

Dryad doi:10.5061/dryad.m37pvmd99 — 60 stations, 2021-2023, 5-min PV power +
1-min weather from one shared campus tower (CC0). The streams are NOT
pre-aligned (plan correction #2): we resample 5-min power -> hourly means,
align 1-min weather to the same hourly grid, keep everything tz-aware in UTC.

This module is the only place that knows the raw file layout. Until the data
is downloaded (Week 3), ``load_station`` raises a clear instruction, and
``synthetic_station`` provides a physically-plausible series so the harness and
tests run end-to-end without the download.

HKUST site constants (Sai Kung):
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("pvfc.data")

HKUST_LAT = 22.337
HKUST_LON = 114.263
HKUST_ALT = 60.0
HKUST_TZ = "Asia/Hong_Kong"  # UTC+8, no DST

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
METADATA_TTL = RAW_DIR / "Dataset" / "Metadata" / "PV generation system metadata.ttl"

# Canonical weather columns present in BOTH oracle and nwp modes, so a model's
# covariate config is identical regardless of weather source (only the values
# differ). Models intersect their requested covariates with what the frame has.
WEATHER_COLS = ["temp_air", "ghi", "wind_speed", "rh"]

# Raw weather column *base name* (unit suffix stripped + lowercased) -> canonical
# name. The real files name columns with units, e.g. "Irradiance (W/m2)", "Temp
# (Degree Celsius)", "RH (%)", "Wind Speed (m/s)", and store one variable per file
# under data/raw/.../Meteorological dataset/<Variable>/<Variable>_<year>.csv.
_WEATHER_BASE = {
    "irradiance": "ghi",         # W/m^2, global horizontal irradiance
    "temp": "temp_air",          # deg C
    "rh": "rh",                  # %
    "wind speed": "wind_speed",  # m/s
    "slp": "slp",                # hPa
    "vis": "visibility",         # km
    "rainfall": "rainfall",      # mm
}
# column base names that mark an inverter-level (5-min) file we skip for site forecasting
_INVERTER_MARKERS = ("totalactivepower", "l1_activepower", "dcvoltage")
# plausible timestamp column names; HKUST files use "Time"
_TS_CANDIDATES = ("time", "timestamp", "datetime", "date", "localtime", "local_time", "utc")


def _base(col: str) -> str:
    """Column base name: drop any '(unit)' suffix, lowercase, strip. 'power(W)' -> 'power'."""
    return str(col).split("(")[0].strip().lower()


@dataclass
class StationSeries:
    """Hourly, tz-aware (UTC) frame for one PV station with clear-sky context."""

    site_id: str
    frame: pd.DataFrame  # cols: power, p_clear, k, is_day, + weather/calendar
    capacity: float
    lat: float = HKUST_LAT
    lon: float = HKUST_LON
    altitude: float = HKUST_ALT


def _read_ts_csv(path: Path) -> pd.DataFrame:
    """Read a HKUST CSV and return it indexed by a tz-aware UTC DatetimeIndex.

    Raw timestamps are assumed Hong Kong local (UTC+8, no DST -> unambiguous);
    if they already carry a tz we just convert. The timestamp column is detected
    by name, else taken to be the leading column (HKUST files lead with it).
    """
    df = pd.read_csv(path)
    ts_col = next((c for c in df.columns if str(c).strip().lower() in _TS_CANDIDATES), df.columns[0])
    idx = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.drop(columns=[ts_col])
    df.index = idx
    df = df[df.index.notna()].sort_index()
    df.index = (df.index.tz_localize(HKUST_TZ) if df.index.tz is None else df.index).tz_convert("UTC")
    return df


def _classify(header: str) -> str:
    bases = {_base(c) for c in header.split(",")}
    if any(m in bases for m in _INVERTER_MARKERS):
        return "inverter"          # 5-min inverter-level — skipped for site forecasting
    if "power" in bases and "generation" in bases:
        return "pv_site"           # 60 site-level stations (1-h or 15-min)
    if bases & set(_WEATHER_BASE):
        return "weather"           # one campus-tower variable per file, per year
    return "other"


def _scan(raw_dir: Path) -> dict[str, list[Path]]:
    """Group raw CSVs by class from their header signature."""
    groups: dict[str, list[Path]] = {"pv_site": [], "inverter": [], "weather": [], "other": []}
    for p in sorted(raw_dir.rglob("*.csv")):
        with open(p, "r", errors="replace") as fh:
            groups[_classify(fh.readline().strip())].append(p)
    return groups


def list_stations(raw_dir: Path = RAW_DIR) -> dict[str, Path]:
    """Discover site-level PV stations as {site_id: csv_path}.

    site_id is the file stem (the parent folder is prefixed when stems collide).
    """
    if not raw_dir.exists() or not any(raw_dir.rglob("*.csv")):
        raise FileNotFoundError(
            f"No raw CSVs in {raw_dir}. Run `python experiments/get_data.py` first "
            "(that script documents the manual Dryad download). Use synthetic_station() "
            "for harness development without the download."
        )
    sites: dict[str, Path] = {}
    counts: dict[str, int] = {}
    for p in _scan(raw_dir)["pv_site"]:
        counts[p.stem] = counts.get(p.stem, 0) + 1
        sid = p.stem if counts[p.stem] == 1 else f"{p.parent.name}__{p.stem}"
        sites[sid] = p
    return sites


def _read_power(path: Path) -> pd.Series:
    """Site-level active power (W) -> hourly-mean kW, tz-aware UTC."""
    df = _read_ts_csv(path)
    col = next((c for c in df.columns if _base(c) in ("power", "totalactivepower")), None)
    if col is None:
        raise ValueError(f"No power column in {path.name}; columns={list(df.columns)}")
    p_kw = (pd.to_numeric(df[col], errors="coerce") / 1000.0).clip(lower=0.0)  # W -> kW
    return p_kw.resample("1h").mean()


def _read_weather(raw_dir: Path) -> pd.DataFrame:
    """Assemble the per-variable campus-tower weather files -> hourly-mean canonical frame.

    Each variable lives in its own folder with one file per year (e.g.
    Meteorological dataset/Temperature/Temperature_2023.csv), so we collect every
    matching value column across files, concatenate years, and resample to hourly.
    """
    by_col: dict[str, list[pd.Series]] = {}
    for p in _scan(raw_dir)["weather"]:
        df = _read_ts_csv(p)
        for col in df.columns:
            canon = _WEATHER_BASE.get(_base(col))
            if canon:
                by_col.setdefault(canon, []).append(pd.to_numeric(df[col], errors="coerce"))
    if not by_col:
        logger.warning("no weather files found under %s; weather columns will be absent", raw_dir)
        return pd.DataFrame()
    out = {}
    for canon, parts in by_col.items():
        s = pd.concat(parts).sort_index()
        out[canon] = s[~s.index.duplicated(keep="first")].resample("1h").mean()
    wx = pd.DataFrame(out)
    wx = wx[~wx.index.duplicated(keep="first")]
    return wx.resample("1h").mean()


def _norm_key(s: str) -> str:
    """Collapse a station name to a comparison key: lowercase, alphanumerics only."""
    import re

    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# CSV stem -> TTL entity spelling mismatches that survive _norm_key (Centre/Center, etc.).
_CAPACITY_ALIASES = {"indoorsportscentre": "indoorsportcenter"}

_CAPACITY_CACHE: dict[str, float] | None = None


def load_nameplate_capacities(ttl_path: Path = METADATA_TTL) -> dict[str, float]:
    """Parse per-system rated power (kW) from the Brick metadata .ttl.

    Each ``pvsystem:<name> a brick:PV_Generation_System`` block carries an
    ``ext:ratedPowerOutput [ brick:hasUnit unit:KW ; brick:value <X> ]``. Returns
    ``{_norm_key(name): kW}`` (regex-parsed; no rdflib dependency). Cached after
    first read. Returns ``{}`` if the file is absent.
    """
    global _CAPACITY_CACHE
    if _CAPACITY_CACHE is not None:
        return _CAPACITY_CACHE
    import re

    caps: dict[str, float] = {}
    if Path(ttl_path).exists():
        text = Path(ttl_path).read_text(encoding="utf-8", errors="replace")
        block = re.compile(
            r"pvsystem:(\S+)\s+a\s+brick:PV_Generation_System\s*;(.*?)"
            r"(?:\.\s*\n\s*\n|\.\s*\npvsystem:|\Z)",
            re.S,
        )
        rated = re.compile(
            r"ratedPowerOutput\s*\[\s*brick:hasUnit\s+unit:KW\s*;\s*brick:value\s+([0-9.]+)"
        )
        for m in block.finditer(text):
            rm = rated.search(m.group(2))
            if rm:
                caps[_norm_key(m.group(1))] = float(rm.group(1))
    else:
        logger.warning("metadata .ttl not found at %s; nameplate lookup disabled", ttl_path)
    _CAPACITY_CACHE = caps
    return caps


def station_capacity(site_id: str, ttl_path: Path = METADATA_TTL) -> float | None:
    """Nameplate kW for ``site_id`` from Brick metadata, or ``None`` if not found."""
    caps = load_nameplate_capacities(ttl_path)
    key = _norm_key(site_id)
    key = _CAPACITY_ALIASES.get(key, key)
    return caps.get(key)


def _derive_capacity(power: pd.Series, override: float | None, site_id: str | None = None) -> float:
    """Installed kW for metric normalization and the p_clear clip.

    Priority: explicit ``override`` -> Brick-metadata nameplate (``station_capacity``)
    -> robust 0.999-quantile of observed power (fallback proxy). The nameplate is
    the physically correct denominator; the proxy is only used when no metadata
    match exists (e.g. the synthetic station or an unmatched name).
    """
    if override:
        return float(override)
    if site_id is not None:
        nameplate = station_capacity(site_id)
        if nameplate:
            logger.info("capacity: %r -> %.2f kW nameplate (Brick metadata)", site_id, nameplate)
            return float(nameplate)
    cap = float(power.quantile(0.999))
    logger.warning("capacity: no nameplate for %r; using 0.999-quantile of power = %.2f kW",
                   site_id, cap)
    return cap


def load_station(site_id: str, raw_dir: Path = RAW_DIR, weather_mode: str = "oracle",
                 capacity: float | None = None, cache: bool = True) -> StationSeries:
    """Load + ETL one HKUST station into an hourly, tz-aware (UTC) StationSeries.

    weather_mode:
      - "oracle": measured campus-tower weather (perfect-foresight upper bound).
      - "nwp":    archived day-ahead forecast weather (pvfc.weather / Open-Meteo).

    Steps: read site-level power (W) -> hourly kW; read 1-min weather -> hourly;
    join on the hourly UTC index; derive capacity; attach clear-sky power/index/
    daytime mask. The resulting frame schema matches synthetic_station().
    """
    if weather_mode not in ("oracle", "nwp"):
        raise ValueError(f"weather_mode must be 'oracle' or 'nwp', got {weather_mode!r}")

    cache_path = PROCESSED_DIR / f"{site_id}__{weather_mode}.parquet"
    cap_sidecar = cache_path.with_suffix(".cap.json")
    if cache and cache_path.exists() and cap_sidecar.exists():
        frame = pd.read_parquet(cache_path)
        cap = capacity or float(json.loads(cap_sidecar.read_text())["capacity"])
        return StationSeries(site_id=site_id, frame=frame, capacity=cap)

    sites = list_stations(raw_dir)
    if site_id not in sites:
        preview = ", ".join(sorted(sites)[:10]) + (" ..." if len(sites) > 10 else "")
        raise KeyError(f"Unknown site_id {site_id!r}. Available ({len(sites)}): {preview}")

    power = _read_power(sites[site_id])
    weather = _read_weather(raw_dir)
    frame = power.to_frame("power").join(weather, how="left") if not weather.empty \
        else power.to_frame("power")
    # short daytime gaps: interpolate; longer gaps stay NaN (guarded in the runner)
    frame["power"] = frame["power"].interpolate(limit=2).clip(lower=0.0)

    cap = _derive_capacity(frame["power"], capacity, site_id=site_id)
    frame = attach_clearsky(frame, capacity=cap)

    if weather_mode == "nwp":
        from . import weather as wx
        frame = wx.attach_nwp_forecast(frame, lat=HKUST_LAT, lon=HKUST_LON)

    if cache:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache_path)
        cap_sidecar.write_text(json.dumps({"capacity": cap}))
    return StationSeries(site_id=site_id, frame=frame, capacity=cap)


def attach_clearsky(frame: pd.DataFrame, capacity: float, lat: float = HKUST_LAT,
                    lon: float = HKUST_LON, altitude: float = HKUST_ALT,
                    fit_days: int | None = 365) -> pd.DataFrame:
    """Add p_clear, k and is_day columns to an hourly power frame.

    The empirical performance-ratio table behind ``p_clear`` is FIT on the first
    ``fit_days`` calendar days only (default 365 = the splitter's initial train
    window, which is inside every fold's training set by construction) and then
    applied to the whole index — so the clear-sky template, and therefore the
    target ``k``, never embeds test-period information. Residual drift
    (degradation/soiling after year 1) is a documented simplification.
    ``fit_days=None`` fits on everything (synthetic/diagnostic use only).
    """
    from . import clearsky as cs
    from .normalize import clearsky_index

    idx = frame.index
    ghi_clear = cs.clearsky_ghi(idx, lat, lon, altitude)
    fit_mask = None
    if fit_days is not None:
        fit_mask = pd.Series(idx < idx.min() + pd.Timedelta(days=fit_days), index=idx)
    p_clear = cs.empirical_clearsky_power(frame["power"], ghi_clear, capacity=capacity,
                                          fit_mask=fit_mask)
    out = frame.copy()
    out["p_clear"] = p_clear
    out["k"] = clearsky_index(out["power"], out["p_clear"])
    out["is_day"] = cs.daytime_mask(idx, lat, lon).values
    return out


def synthetic_station(
    site_id: str = "synthetic", days: int = 540, seed: int = 0, capacity: float = 5.0,
    lat: float = HKUST_LAT, lon: float = HKUST_LON, altitude: float = HKUST_ALT,
) -> StationSeries:
    """A physically-plausible hourly PV series for harness/tests (no download).

    power = clear-sky power * a smoothly-varying cloudiness factor in [0,1].
    Uses the same pvlib clear-sky envelope as the real pipeline so the clear-sky
    index machinery is exercised identically.
    """
    from . import clearsky as cs
    from .normalize import clearsky_index

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=days * 24, freq="1h", tz="UTC")
    ghi_clear = cs.clearsky_ghi(idx, lat, lon, altitude)
    p_clear = cs.empirical_clearsky_power(
        pd.Series(np.maximum(ghi_clear.values, 0) * capacity / max(ghi_clear.max(), 1), index=idx),
        ghi_clear, capacity=capacity,
    )
    # AR(1) cloudiness in clear-sky-index space, bounded to [0, 1]
    k_true = np.empty(len(idx))
    k_true[0] = 0.8
    for t in range(1, len(idx)):
        k_true[t] = np.clip(0.92 * k_true[t - 1] + 0.08 * rng.uniform(0.2, 1.0)
                            + rng.normal(0, 0.05), 0.05, 1.05)
    power = pd.Series(np.maximum(k_true * p_clear.values, 0.0), index=idx, name="power")
    frame = pd.DataFrame({"power": power, "p_clear": p_clear})
    frame["k"] = clearsky_index(frame["power"], frame["p_clear"])
    frame["is_day"] = cs.daytime_mask(idx, lat, lon).values
    # simple weather proxy correlated with cloudiness (stand-in for real columns)
    frame["temp_air"] = 20 + 8 * np.sin(2 * np.pi * idx.dayofyear / 365) + rng.normal(0, 1, len(idx))
    return StationSeries(site_id=site_id, frame=frame, capacity=capacity, lat=lat, lon=lon, altitude=altitude)
