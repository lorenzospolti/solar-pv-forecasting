"""Real-data ETL smoke tests — skipped when the HKUST raw data is not present.

These exercise pvfc.data.load_station against the downloaded Dryad files. They
stay skipped (green) in any environment without the ~296 MB download, mirroring
how test_smoke.py uses the synthetic station.
"""
from __future__ import annotations

import numpy as np
import pytest

from pvfc import data as datamod
from pvfc.data import HKUST_TZ, RAW_DIR

_has_raw = RAW_DIR.exists() and any(RAW_DIR.rglob("*.csv"))
pytestmark = pytest.mark.skipif(not _has_raw, reason="HKUST raw data not downloaded")


@pytest.fixture(scope="module")
def first_station():
    site = sorted(datamod.list_stations())[0]
    return datamod.load_station(site, weather_mode="oracle")


def test_index_is_utc_hourly(first_station):
    idx = first_station.frame.index
    assert str(idx.tz) == "UTC"
    assert idx.is_monotonic_increasing


def test_power_peaks_at_local_noon(first_station):
    f = first_station.frame
    peak = int(f["power"].groupby(f.index.tz_convert(HKUST_TZ).hour).mean().idxmax())
    assert peak in (11, 12, 13)


def test_night_power_is_zero(first_station):
    f = first_station.frame
    night = ~f["is_day"].to_numpy().astype(bool)
    # deep night is exactly 0; twilight (zenith>85) trickles, so check the mean
    assert float(f.loc[night, "power"].mean()) < 0.01 * first_station.capacity


def test_clearsky_finite_on_daytime(first_station):
    f = first_station.frame
    day = f["is_day"].to_numpy().astype(bool)
    assert np.isfinite(f.loc[day, ["p_clear", "k"]].to_numpy()).all()
    assert 0.2 <= float(f.loc[day, "k"].median()) <= 1.0
