"""Data loading & ETL for the HKUST rooftop PV dataset (primary).

Dryad doi:10.5061/dryad.m37pvmd99 — 60 stations, 2021-2023, site-level PV power
(37 stations at 15-min, 23 at 1-h resolution) + 1-min weather from one shared
campus tower (CC0). The streams are NOT pre-aligned (plan correction #2): we
resample power -> hourly means, align 1-min weather to the same hourly grid, and
keep everything tz-aware in UTC.

Timestamp convention (ETL v2+). Every hourly frame uses interval-START labels: the
value stamped HH:00 is the mean over [HH:00, HH+1:00). The 15-min files already
label the start of each interval, but the 23 hourly files label the END of the
hour: each of them correlates best with interval-start references (15-min
stations, 1-min tower irradiance) at a -1 h shift (e.g. CYT (Station 2): 0.985 at
-1 h vs 0.837 unshifted). ``_read_power`` therefore shifts hourly files back by
one hour.

Power source (ETL v4). Mean power is the ``power(W)`` column. In the hourly files it
equals ``generation(kWh)`` per hour exactly. In the 15-min files the two columns
disagree by about 3%: ``power`` integrates to within 0.3% of the AC power the
inverters log every 5 minutes (pooled ratio 1.002-1.003 at the four 15-min panel
stations) and is the column the dataset paper integrates to validate the site level
against the inverters (Library, 31 Aug 2022: 642.09 vs 641.55 kWh), while
``generation`` sits 3.1-3.3% above the inverters (2.8-3.0% above ``power``). Both
columns follow the quarter hour they label. ETL v3 used ``generation``, on a
misreading of the paper's variable tables.

This module is the only place that knows the raw file layout. Without the
download, ``load_station`` raises a clear instruction, and ``synthetic_station``
provides a physically-plausible series so the harness and tests run end-to-end.

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
QC_LOG = PROCESSED_DIR / "qc_log.csv"

#: Bump whenever the ETL changes what a processed frame contains: it is part of the
#: cache file name, so frames built by an older ETL are never silently reused.
#: v2 (Oct 2026): hourly files relabelled to interval start, clear-sky quantities
#: evaluated per interval, sun-position template, capacity QC, weather mode "none".
#: v3 (Oct 2026): power from the energy column; negative meter errors set missing;
#: only gaps of <= 2 h bridged; outage hours (pvfc.qc) set missing before the
#: clear-sky template fit; hour angle wrapped to [-180, 180) (pvfc.clearsky);
#: Open-Meteo cache years always complete (SQ15 lacked weather before 4 May 2021).
#: v4 (Oct 2026): power from the power column again: v3's energy column sits about 3%
#: above the inverters' AC records at the 15-min stations (module docstring).
ETL_VERSION = 4

#: weather feeds: measured campus tower (oracle), archived forecast (nwp), or no
#: weather covariates at all (none: p_clear + calendar only).
WEATHER_MODES = ("oracle", "nwp", "none")

#: QC: raw power above this multiple of nameplate is a meter error (Tower A reads
#: 31,755 kW against 40 kW) and is set to NaN before resampling.
QC_MAX_CAPACITY_RATIO = 1.1
#: QC: raw power below this multiple of nameplate is a meter error too (SQ15 reads
#: -111 x and -148 x nameplate) and is set to NaN; smaller negatives (standby
#: consumption at night) become 0.
QC_MIN_CAPACITY_RATIO = -0.05
#: gaps of at most this many hours, with a measured hour on both sides, are bridged
#: by linear interpolation; longer gaps stay missing in full
MAX_BRIDGED_GAP_HOURS = 2
#: QC: an hour whose midpoint has the sun more than 5 deg below the horizon cannot
#: generate; a reading above this share of nameplate then is a meter error (SQ15:
#: 8.9 kW of 33.3 kW at 02:00 on 29 Aug 2022, the only one in the panel)
QC_NIGHT_MAX_RATIO = 0.01
QC_NIGHT_ZENITH_DEG = 95.0

#: Stations kept out of the benchmark, with the reason (Data chapter material).
EXCLUDED_STATIONS = {
    "CYT (Station 1)": (
        "its timestamps sit 30-45 min off every other station (best match to an "
        "interval-start reference: corr 0.92 for a window shifted 30-45 min, vs 0.985 "
        "for CYT (Station 2) at -1 h), so no hourly relabel aligns it; it also has "
        "the second-worst coverage of the 60 (8.8% of daytime hours missing)"
    ),
    "Zone A2": (
        "its output changes regime in May 2023 (99th-percentile power 0.25 x nameplate "
        "through April 2023, 0.66 x after; part of the 249.5 kW array apparently came "
        "online then), so a clear-sky template fit on the first year pins k at its "
        "1.3 ceiling in 43% of 2023's daytime hours (every other eligible station: "
        "<= 2.9% in any year)"
    ),
}

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


def _read_power(path: Path) -> tuple[pd.Series, pd.Timedelta]:
    """Site-level mean power (kW) per native interval, tz-aware UTC, with
    interval-START labels; also returns the native step.

    Mean power is the ``power(W)`` column in kW (see the module docstring for why not
    ``generation(kWh)``). Hourly files label the END of each hour, so they are shifted
    back by one step; 15-min files already label the start. Negative readings are
    kept: ``_qc_power`` decides what they are.
    """
    df = _read_ts_csv(path)
    col = next((c for c in df.columns if _base(c) in ("power", "totalactivepower")), None)
    if col is None:
        raise ValueError(f"No power column in {path.name}; columns={list(df.columns)}")
    step = pd.Series(df.index.unique()).diff().median()
    p_kw = pd.to_numeric(df[col], errors="coerce") / 1000.0  # W -> kW
    if step >= pd.Timedelta(minutes=59):  # hourly meter: stamp HH covers (HH-1, HH]
        p_kw.index = p_kw.index - step
    return p_kw.rename("power"), step


def _qc_power(raw_kw: pd.Series, capacity: float) -> tuple[pd.Series, dict[str, int]]:
    """Meter errors at native resolution, before resampling (one bad reading cannot
    contaminate an hourly mean): readings above ``QC_MAX_CAPACITY_RATIO`` x nameplate
    and below ``QC_MIN_CAPACITY_RATIO`` x nameplate become NaN; the remaining negative
    readings (standby consumption) become 0. Returns the cleaned series and the counts.
    """
    over = raw_kw > QC_MAX_CAPACITY_RATIO * capacity
    under = raw_kw < QC_MIN_CAPACITY_RATIO * capacity
    standby = (raw_kw < 0) & ~under
    clean = raw_kw.mask(over | under).mask(standby, 0.0)
    return clean, {"over_capacity_removed": int(over.sum()),
                   "negative_removed": int(under.sum()), "negative_zeroed": int(standby.sum())}


def _qc_night(power: pd.Series, capacity: float) -> tuple[pd.Series, int]:
    """NaN out hourly readings above ``QC_NIGHT_MAX_RATIO`` x nameplate whose interval
    midpoint has the sun more than 5 deg below the horizon (``QC_NIGHT_ZENITH_DEG``):
    such an hour cannot generate. Returns the series and the number removed."""
    from . import clearsky as cs

    zenith = cs.solar_zenith(power.index + pd.Timedelta(minutes=30), HKUST_LAT, HKUST_LON,
                             HKUST_ALT).to_numpy()
    bad = (zenith > QC_NIGHT_ZENITH_DEG) & (power.to_numpy() > QC_NIGHT_MAX_RATIO * capacity)
    return power.mask(bad), int(bad.sum())


def _bridge_short_gaps(power: pd.Series,
                       max_hours: int = MAX_BRIDGED_GAP_HOURS) -> tuple[pd.Series, int]:
    """Linearly interpolate runs of at most ``max_hours`` missing hours that have a
    measured hour on both sides; longer gaps stay missing in full. Returns the series
    and the number of hours filled. Expects a regular hourly index.

    (pandas' ``interpolate(limit=2)``, used up to ETL v2, also fills the first two
    hours of every longer gap: it wrote interpolated zeros into the first two hours
    of the 10-hour outage days of CYT (Station 2) and SQ15.)
    """
    missing = power.isna()
    run_len = missing.groupby((~missing).cumsum()).transform("sum")
    bridge = missing & (run_len <= max_hours)
    out = power.where(~bridge, power.interpolate(limit_area="inside"))
    return out, int((bridge & out.notna()).sum())


def _write_qc_row(row: dict, path: Path = QC_LOG) -> None:
    """Upsert one station's QC counts into the QC log (one row per site_id)."""
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        old = old[old["site_id"] != row["site_id"]]
        if not old.empty:
            new = pd.concat([old, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.sort_values("site_id").to_csv(path, index=False)


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


def load_station(site_id: str, raw_dir: Path = RAW_DIR, weather_mode: str = "nwp",
                 capacity: float | None = None, cache: bool = True,
                 allow_excluded: bool = False) -> StationSeries:
    """Load + ETL one HKUST station into an hourly, tz-aware (UTC) StationSeries.

    weather_mode:
      - "nwp":    archived day-ahead forecast weather (pvfc.weather / Open-Meteo).
      - "none":   no weather columns; models get p_clear + calendar only.
      - "oracle": measured campus-tower weather. NOT a perfect-foresight bound: the
                  tower pyranometer drifts (midday means 404/422/632 W/m^2 in
                  2021/22/23 vs a flat ~590 in the archive). Diagnostics only.

    Steps: read site-level mean power (kW, from the power column) with interval-start
    labels; meter-error QC (``_qc_power``); resample to hourly means; bridge gaps of
    at most ``MAX_BRIDGED_GAP_HOURS``; attach clear-sky power/index/daytime mask; set
    outage hours (``pvfc.qc.outage_mask``) missing and refit the clear-sky template
    without them; attach the weather feed.
    Frames are cached under a name that carries ``ETL_VERSION``; building one
    also upserts the station's QC counts into ``data/processed/qc_log.csv``.
    The frame schema matches synthetic_station().
    """
    if weather_mode not in WEATHER_MODES:
        raise ValueError(f"weather_mode must be one of {WEATHER_MODES}, got {weather_mode!r}")
    if site_id in EXCLUDED_STATIONS and not allow_excluded:
        raise ValueError(f"{site_id!r} is excluded from the benchmark: "
                         f"{EXCLUDED_STATIONS[site_id]}. Pass allow_excluded=True for diagnostics.")

    cache_path = PROCESSED_DIR / f"{site_id}__{weather_mode}__etl{ETL_VERSION}.parquet"
    cap_sidecar = cache_path.with_suffix(".cap.json")
    if cache and cache_path.exists() and cap_sidecar.exists():
        frame = pd.read_parquet(cache_path)
        cap = capacity or float(json.loads(cap_sidecar.read_text())["capacity"])
        return StationSeries(site_id=site_id, frame=frame, capacity=cap)

    sites = list_stations(raw_dir)
    if site_id not in sites:
        preview = ", ".join(sorted(sites)[:10]) + (" ..." if len(sites) > 10 else "")
        raise KeyError(f"Unknown site_id {site_id!r}. Available ({len(sites)}): {preview}")

    from .qc import outage_mask

    raw, step = _read_power(sites[site_id])
    cap = _derive_capacity(raw, capacity, site_id=site_id)
    raw, qc = _qc_power(raw, cap)
    if qc["over_capacity_removed"] or qc["negative_removed"]:
        logger.warning("QC %r: %d raw readings > %.1f x and %d < %.2f x nameplate (%.1f kW) "
                       "set to NaN", site_id, qc["over_capacity_removed"], QC_MAX_CAPACITY_RATIO,
                       qc["negative_removed"], QC_MIN_CAPACITY_RATIO, cap)
    frame = raw.resample("1h").mean().to_frame("power")
    if weather_mode == "oracle":
        weather = _read_weather(raw_dir)
        if not weather.empty:
            frame = frame.join(weather, how="left")
    frame["power"], qc["night_removed"] = _qc_night(frame["power"], cap)
    # gaps of <= 2 h with data on both sides are bridged; longer gaps stay NaN
    frame["power"], qc["interpolated_hours"] = _bridge_short_gaps(frame["power"])
    frame = attach_clearsky(frame, capacity=cap)
    # outage hours (runs of zero output under a high sun) are missing data, not
    # measurements: drop them, then refit the template so its fit never sees them
    outage = outage_mask(frame.index, frame["power"], frame["p_clear"], frame["is_day"], cap)
    qc["outage_hours_removed"] = int(outage.sum())
    if outage.any():
        logger.warning("QC %r: %d outage hours set to NaN", site_id, int(outage.sum()))
        frame.loc[outage, "power"] = np.nan
        frame = attach_clearsky(frame.drop(columns=["p_clear", "k", "is_day"]), capacity=cap)

    if weather_mode == "nwp":
        from . import weather as wx
        frame = wx.attach_nwp_forecast(frame, lat=HKUST_LAT, lon=HKUST_LON)

    if cache:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache_path)
        cap_sidecar.write_text(json.dumps({"capacity": cap}))
        day = frame["is_day"].to_numpy(dtype=bool)
        _write_qc_row({
            "site_id": site_id, "etl_version": ETL_VERSION,
            "resolution_min": int(step / pd.Timedelta(minutes=1)),
            "relabelled_minus_1h": bool(step >= pd.Timedelta(minutes=59)),
            "capacity_kw": cap, "raw_readings": int(raw.size),
            "source_column": "power", **qc,
            "first_hour_utc": frame.index.min(), "last_hour_utc": frame.index.max(),
            "daytime_hours": int(day.sum()),
            "daytime_missing_pct": round(100 * float(frame.loc[day, "power"].isna().mean()), 2),
            "daytime_k_above_1_pct": round(100 * float((frame.loc[day, "k"] > 1).mean()), 2),
        })
    return StationSeries(site_id=site_id, frame=frame, capacity=cap)


def attach_clearsky(frame: pd.DataFrame, capacity: float, lat: float = HKUST_LAT,
                    lon: float = HKUST_LON, altitude: float = HKUST_ALT,
                    fit_days: int | None = 365, interval: str = "1h") -> pd.DataFrame:
    """Add p_clear, k and is_day columns to an interval-start power frame.

    Clear-sky GHI is the mean over each interval and the sun position (template
    coordinates, daytime mask) is taken at the interval midpoint, matching the
    interval-mean power labels.

    The sun-position performance-ratio grid behind ``p_clear`` is FIT on the
    first ``fit_days`` calendar days only (default 365 = the splitter's initial
    train window, which is inside every fold's training set by construction) and
    then applied to the whole index — so the clear-sky template, and therefore
    the target ``k``, never embeds test-period information. Residual drift
    (degradation/soiling after year 1) is a documented simplification.
    ``fit_days=None`` fits on everything (synthetic/diagnostic use only).
    """
    from . import clearsky as cs
    from .normalize import clearsky_index

    idx = frame.index
    mid = idx + pd.Timedelta(interval) / 2
    ghi_clear = cs.clearsky_ghi_interval(idx, lat, lon, altitude, interval=interval)
    decl, ha = cs.sun_coordinates(mid, lon)
    fit_mask = None
    if fit_days is not None:
        fit_mask = pd.Series(idx < idx.min() + pd.Timedelta(days=fit_days), index=idx)
    p_clear = cs.empirical_clearsky_power(frame["power"], ghi_clear, decl, ha,
                                          capacity=capacity, fit_mask=fit_mask)
    out = frame.copy()
    out["p_clear"] = p_clear
    out["k"] = clearsky_index(out["power"], out["p_clear"])
    out["is_day"] = cs.daytime_mask(mid, lat, lon).to_numpy()
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
    mid = idx + pd.Timedelta(minutes=30)
    ghi_clear = cs.clearsky_ghi_interval(idx, lat, lon, altitude)
    decl, ha = cs.sun_coordinates(mid, lon)
    p_clear = cs.empirical_clearsky_power(
        pd.Series(np.maximum(ghi_clear.values, 0) * capacity / max(ghi_clear.max(), 1), index=idx),
        ghi_clear, decl, ha, capacity=capacity,
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
    frame["is_day"] = cs.daytime_mask(mid, lat, lon).to_numpy()
    # simple weather proxy correlated with cloudiness (stand-in for real columns)
    frame["temp_air"] = 20 + 8 * np.sin(2 * np.pi * idx.dayofyear / 365) + rng.normal(0, 1, len(idx))
    return StationSeries(site_id=site_id, frame=frame, capacity=capacity, lat=lat, lon=lon, altitude=altitude)
