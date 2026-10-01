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


DEV_STATION = "CYT (Station 2)"  # the development station (configs/experiment.yaml)


@pytest.fixture(scope="module")
def first_station():
    return datamod.load_station(DEV_STATION, weather_mode="nwp")


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


def test_k_finite_wherever_power_is_finite(first_station):
    """Daytime gaps stay NaN in k by design (June fix B6), so finiteness is only
    required where power itself was measured."""
    f = first_station.frame
    day = f["is_day"].to_numpy().astype(bool)
    assert np.isfinite(f.loc[day, "p_clear"].to_numpy()).all()
    measured = day & np.isfinite(f["power"].to_numpy())
    assert np.isfinite(f.loc[measured, "k"].to_numpy()).all()
    assert 0.2 <= float(f.loc[day, "k"].median()) <= 1.0


def test_template_has_no_holes_and_no_steps(first_station):
    p = first_station.frame["p_clear"].tz_convert(HKUST_TZ)
    by_day = p.groupby(p.index.date)
    daily_max = by_day.max()[by_day.size() == 24]  # ignore partial first/last days
    assert (daily_max > 0.4 * daily_max.max()).all()
    assert float(daily_max.pct_change().abs().max()) < 0.05


def test_output_is_stable_relative_to_the_fit_year(first_station):
    """A capacity change after the clear-sky fit year pins k at its 1.3 ceiling for a
    large share of daytime hours (Zone A2: 43% in 2023, hence excluded; every other
    eligible station stays <= 2.9% in any year)."""
    f = first_station.frame.tz_convert(HKUST_TZ)
    k = f.loc[f["is_day"].astype(bool) & f["k"].notna(), "k"]
    at_ceiling = (k >= 1.3).groupby(k.index.year).mean()
    assert (at_ceiling < 0.05).all(), at_ceiling.to_dict()


def test_hours_are_labelled_by_interval_start(first_station):
    """With interval-start labels the first and last daylight hours of a day carry
    power of the same order (ETL v1, with end-of-hour stamps on hourly files,
    had ~0.5 kW at the first daytime stamp vs ~1.0 kW at the last)."""
    f = first_station.frame.tz_convert(HKUST_TZ)
    d = f[f["is_day"].astype(bool) & f["power"].notna()]
    by_day = d.groupby(d.index.date)["power"]
    ratio = by_day.first().mean() / by_day.last().mean()
    # 0.98 for this station with ETL v2; end-of-hour labels give ~0.5. (Orientation or
    # shading can skew it per station: SQ15 reads 0.67 although its -1 h relabel is
    # unambiguous, corr 0.970 vs 0.819 unshifted.)
    assert 0.75 < ratio < 1.33, f"first/last daytime power ratio {ratio:.2f}"


def test_power_respects_the_qc_bound(first_station):
    assert float(first_station.frame["power"].max()) <= \
        datamod.QC_MAX_CAPACITY_RATIO * first_station.capacity


def test_excluded_station_is_refused():
    with pytest.raises(ValueError, match="excluded from the benchmark"):
        datamod.load_station("CYT (Station 1)", cache=False)


def test_none_mode_has_no_weather_columns():
    f = datamod.load_station(DEV_STATION, weather_mode="none").frame
    assert set(f.columns) == {"power", "p_clear", "k", "is_day"}
