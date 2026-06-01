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

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HKUST_LAT = 22.337
HKUST_LON = 114.263
HKUST_ALT = 60.0
HKUST_TZ = "Asia/Hong_Kong"  # UTC+8, no DST

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


@dataclass
class StationSeries:
    """Hourly, tz-aware (UTC) frame for one PV station with clear-sky context."""

    site_id: str
    frame: pd.DataFrame  # cols: power, p_clear, k, is_day, + weather/calendar
    capacity: float
    lat: float = HKUST_LAT
    lon: float = HKUST_LON
    altitude: float = HKUST_ALT


def load_station(site_id: str, raw_dir: Path = RAW_DIR) -> StationSeries:
    """Load and ETL one station from the raw HK files into an hourly UTC frame.

    TODO (Week 3): implement against the real Dryad file layout once downloaded.
    Steps: read 5-min power CSV -> to_datetime(utc=True) -> resample('1h').mean();
    read 1-min weather -> resample('1h').mean(); join on the hourly UTC index;
    attach clear-sky power + index + daytime mask via pvfc.clearsky.
    """
    if not any(raw_dir.glob("*")):
        raise FileNotFoundError(
            f"No raw data in {raw_dir}. Download HKUST PV (Dryad "
            "doi:10.5061/dryad.m37pvmd99) into data/raw/, then implement the "
            "real layout here. Use synthetic_station() for harness development."
        )
    raise NotImplementedError("Implement real HK ETL in Week 3 (see docstring).")


def attach_clearsky(frame: pd.DataFrame, capacity: float, lat: float = HKUST_LAT,
                    lon: float = HKUST_LON, altitude: float = HKUST_ALT) -> pd.DataFrame:
    """Add p_clear, k and is_day columns to an hourly power frame."""
    from . import clearsky as cs
    from .normalize import clearsky_index

    idx = frame.index
    ghi_clear = cs.clearsky_ghi(idx, lat, lon, altitude)
    p_clear = cs.empirical_clearsky_power(frame["power"], ghi_clear, capacity=capacity)
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
