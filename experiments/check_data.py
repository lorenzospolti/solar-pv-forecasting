"""Pipeline smoke test for the real HKUST ETL (thesis plan, Week-3 verification).

Loads one station through pvfc.data.load_station and asserts the physical
invariants that prove the ETL is correct, printing PASS/FAIL for each. Requires
the raw data (`python experiments/get_data.py`). Exits non-zero on any failure.

Usage:
    python experiments/check_data.py                       # config site_id (dev station)
    python experiments/check_data.py --site <station_id>
    python experiments/check_data.py --oracle              # also check the tower-weather frame
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from pvfc import data as datamod
from pvfc.data import HKUST_TZ

ROOT = Path(__file__).resolve().parents[1]


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"))
    ap.add_argument("--site", default=None)
    ap.add_argument("--oracle", action="store_true",
                    help="also build the oracle (tower) frame; slow, diagnostics only")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    stations = datamod.list_stations()  # raises a clear error if no raw data
    site = args.site or cfg["data"].get("site_id")
    if site not in stations:
        raise SystemExit(f"unknown station {site!r}; discovered: {sorted(stations)}")
    print(f"Discovered {len(stations)} stations. Loading {site!r} (nwp, ETL v{datamod.ETL_VERSION}) ...")

    s = datamod.load_station(site, weather_mode="nwp", capacity=cfg["data"].get("capacity_kw"))
    f = s.frame
    day = f["is_day"].to_numpy().astype(bool)
    results = []

    results.append(_check(
        "index is tz-aware UTC", f.index.tz is not None and str(f.index.tz) == "UTC",
        str(f.index.tz)))

    local = f.tz_convert(HKUST_TZ)
    peak_hour = int(local["power"].groupby(local.index.hour).mean().idxmax())
    results.append(_check(
        "power peaks at local solar noon (11–13h)", peak_hour in (11, 12, 13),
        f"peak local hour = {peak_hour}"))

    night = f.loc[~day, "power"]
    night_mean = float(night.mean())  # deep night is 0; twilight (zenith>85) trickles a little
    results.append(_check(
        "night power ~ 0 (mean)", night_mean < 0.01 * s.capacity,
        f"mean {night_mean:.4f} kW, max {float(night.max()):.3f} kW (twilight), cap {s.capacity:.1f}"))

    nan_frac = float(f.loc[day, "power"].isna().mean())  # real PV has outages; runner drops NaNs
    results.append(_check(
        "daytime power mostly present", nan_frac < 0.20, f"daytime NaN fraction = {nan_frac:.3f}"))
    results.append(_check(
        "power within the QC bound",
        float(f["power"].max()) <= datamod.QC_MAX_CAPACITY_RATIO * s.capacity,
        f"max {float(f['power'].max()):.2f} kW vs {datamod.QC_MAX_CAPACITY_RATIO} x {s.capacity:.1f}"))

    # daytime gaps stay NaN in k by design, so k must be finite where power is measured
    measured = day & np.isfinite(f["power"].to_numpy())
    results.append(_check(
        "p_clear finite on daytime, k finite wherever power is measured",
        bool(np.isfinite(f.loc[day, "p_clear"]).all() and np.isfinite(f.loc[measured, "k"]).all())))

    by_day = local["p_clear"].groupby(local.index.date)
    daily_max = by_day.max()[by_day.size() == 24]  # complete days only
    results.append(_check(
        "clear-sky template: no holes (every daily max > 40% of the template peak)",
        bool((daily_max > 0.4 * daily_max.max()).all()),
        f"lowest daily max {float(daily_max.min()):.1f} kW, peak {float(daily_max.max()):.1f} kW"))
    results.append(_check(
        "clear-sky template: no steps (day-to-day change < 5%)",
        float(daily_max.pct_change().abs().max()) < 0.05,
        f"largest change {float(daily_max.pct_change().abs().max()):.1%}"))

    # hourly cloud enhancement rarely beats clear sky by 30%, so k pinned at its 1.3
    # ceiling means the template no longer matches the system (Zone A2: 43% in 2023)
    k_meas = local.loc[local["is_day"].astype(bool) & local["k"].notna(), "k"]
    at_ceiling = (k_meas >= 1.3).groupby(k_meas.index.year).mean()
    results.append(_check(
        "output stable vs the fit year (k at its 1.3 ceiling in < 5% of daytime hours)",
        bool((at_ceiling < 0.05).all()),
        " / ".join(f"{y}: {v:.1%}" for y, v in at_ceiling.items())))

    k_day = f.loc[day, "k"]
    results.append(_check(
        "daytime k median in [0.2, 1.0]", 0.2 <= float(k_day.median()) <= 1.0,
        f"median k = {k_day.median():.3f}, q90 = {k_day.quantile(0.9):.3f}, "
        f"k > 1 in {(k_day > 1).mean():.1%} of daytime hours"))

    d = local[local["is_day"].astype(bool) & local["power"].notna()]
    first_last = d.groupby(d.index.date)["power"]
    ratio = float(first_last.first().mean() / first_last.last().mean())
    print(f"  [INFO] first/last daytime power ratio = {ratio:.2f} (≈1 for a symmetric array; "
          "~0.5 was the signature of end-of-hour stamps; orientation/shading can skew it)")

    if "ghi" in f.columns:
        corr = float(f.loc[day, "ghi"].corr(f.loc[day, "p_clear"]))
        results.append(_check("nwp ghi correlates with p_clear", corr > 0.3, f"corr = {corr:.2f}"))

    print(f"Loading {site!r} (none) for parity ...")
    s0 = datamod.load_station(site, weather_mode="none", capacity=cfg["data"].get("capacity_kw"))
    results.append(_check("none frame shares the nwp index", s0.frame.index.equals(f.index)))
    results.append(_check("none frame has no weather columns",
                          set(s0.frame.columns) == {"power", "p_clear", "k", "is_day"},
                          str(sorted(s0.frame.columns))))

    if args.oracle:
        print(f"Loading {site!r} (oracle) ...")
        so = datamod.load_station(site, weather_mode="oracle",
                                  capacity=cfg["data"].get("capacity_kw"))
        results.append(_check("oracle frame shares the nwp index", so.frame.index.equals(f.index)))

    print(f"\n{site}: {len(f)} hours, {f.index.min().date()}..{f.index.max().date()}, "
          f"capacity {s.capacity:.2f} kW")
    if not all(results):
        sys.exit(1)
    print("All data smoke checks passed.")


if __name__ == "__main__":
    main()
