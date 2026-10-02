"""ETL v3 fixes (pvfc.data, pvfc.clearsky, pvfc.weather), one test per defect found in
the ETL v2 audit: the hour angle wrap, short-gap-only interpolation, negative meter
errors, power from the energy column, and complete Open-Meteo cache years."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pvfc import data as datamod
from pvfc import weather
from pvfc.clearsky import sun_coordinates

HKT = "Asia/Hong_Kong"


def test_hour_angle_is_wrapped_morning_negative_afternoon_positive():
    local = pd.date_range("2022-07-05 05:30", "2022-07-05 18:30", freq="1h", tz=HKT)
    _, ha = sun_coordinates(local.tz_convert("UTC"), lon=datamod.HKUST_LON)
    assert ((ha >= -180) & (ha < 180)).all()
    assert np.all(np.diff(ha) > 0)                       # monotone through the day
    assert -92 < ha[1] < -86                             # 06:30 HKT, about -89 deg
    assert abs(ha[np.argmin(np.abs(ha))]) < 10           # solar noon near 12:30 HKT


def test_only_short_gaps_are_bridged():
    idx = pd.date_range("2022-01-01", periods=30, freq="1h", tz="UTC")
    p = pd.Series(np.arange(30, dtype=float), index=idx)
    p.iloc[2] = np.nan                                   # 1 h gap -> bridged
    p.iloc[5:7] = np.nan                                 # 2 h gap -> bridged
    p.iloc[10:13] = np.nan                               # 3 h gap -> stays missing
    p.iloc[16:26] = np.nan                               # 10 h gap -> stays missing in full
    out, n = datamod._bridge_short_gaps(p)
    assert n == 3
    assert out.iloc[2] == 2.0 and out.iloc[5] == 5.0 and out.iloc[6] == 6.0
    assert out.iloc[10:13].isna().all() and out.iloc[16:26].isna().all()


def test_a_gap_at_the_start_is_never_bridged():
    p = pd.Series([np.nan, np.nan, 3.0, 4.0],
                  index=pd.date_range("2022-01-01", periods=4, freq="1h", tz="UTC"))
    out, n = datamod._bridge_short_gaps(p)
    assert n == 0 and out.iloc[:2].isna().all()


def test_negative_meter_errors_are_missing_standby_is_zero():
    raw = pd.Series([-1110.0, -0.2, -0.6, 4.0])          # capacity 10 kW: -0.5 kW is the line
    clean, counts = datamod._qc_power(raw, capacity=10.0)
    assert np.isnan(clean.iloc[0]) and np.isnan(clean.iloc[2])
    assert clean.iloc[1] == 0.0 and clean.iloc[3] == 4.0
    assert counts == {"over_capacity_removed": 0, "negative_removed": 2, "negative_zeroed": 1}


def test_power_comes_from_the_energy_column(tmp_path):
    """15-min files: power(W) is not the interval mean (2.8-2.9% low on the panel);
    mean power must be generation(kWh) / 0.25 h."""
    f = tmp_path / "q15.csv"
    f.write_text("Time,generation(kWh),power(W)\n2022-06-15 07:00:00,0.25,900\n"
                 "2022-06-15 07:15:00,0.5,1900\n2022-06-15 07:30:00,0.75,2900\n")
    p, step = datamod._read_power(f)
    assert step == pd.Timedelta(minutes=15)
    np.testing.assert_allclose(p.to_numpy(), [1.0, 2.0, 3.0])


def test_open_meteo_cache_years_are_complete(tmp_path, monkeypatch):
    """A cached year that starts late (the v2 2021 file began on 4 May) is re-fetched
    for the whole year; a complete one is reused."""
    calls = []

    def fake_fetch(lat, lon, start, end):
        calls.append((start, end))
        idx = pd.date_range(f"{start} 00:00", f"{end} 23:00", freq="1h", tz="UTC")
        return pd.DataFrame({"ghi": 1.0, "temp_air": 20.0}, index=idx)

    monkeypatch.setattr(weather, "_fetch_year", fake_fetch)
    late = pd.date_range("2021-05-04", "2021-12-31 23:00", freq="1h", tz="UTC")
    pd.DataFrame({"ghi": 1.0, "temp_air": 20.0}, index=late).to_parquet(
        weather._cache_path(22.0, 114.0, 2021, tmp_path))
    out = weather.fetch_open_meteo(22.0, 114.0, "2021-03-23", "2021-06-01", cache_dir=tmp_path)
    assert calls == [("2021-01-01", "2021-12-31")]
    assert out.index.min() == pd.Timestamp("2021-03-23", tz="UTC")
    weather.fetch_open_meteo(22.0, 114.0, "2021-03-23", "2021-06-01", cache_dir=tmp_path)
    assert len(calls) == 1                               # now complete: reused


def test_impossible_night_readings_are_missing():
    """SQ15 read 8.9 kW of 33.3 kW at 02:00 on 29 Aug 2022: no sun, so a meter error."""
    local = pd.DatetimeIndex(["2022-08-29 02:00", "2022-08-29 12:00", "2022-08-29 18:30"],
                             tz=HKT)
    p = pd.Series([8.9, 20.0, 0.2], index=local.tz_convert("UTC"))
    out, n = datamod._qc_night(p, capacity=33.3)
    assert n == 1 and np.isnan(out.iloc[0])
    assert out.iloc[1] == 20.0 and out.iloc[2] == 0.2    # midday and twilight untouched
