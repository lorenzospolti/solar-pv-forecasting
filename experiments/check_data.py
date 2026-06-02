"""Pipeline smoke test for the real HKUST ETL (thesis plan, Week-3 verification).

Loads one station through pvfc.data.load_station and asserts the physical
invariants that prove the ETL is correct, printing PASS/FAIL for each. Requires
the raw data (`python experiments/get_data.py`). Exits non-zero on any failure.

Usage:
    python experiments/check_data.py                       # uses config site_id (or first found)
    python experiments/check_data.py --site <station_id>
    python experiments/check_data.py --no-nwp              # skip the Open-Meteo parity check
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
    ap.add_argument("--no-nwp", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    stations = datamod.list_stations()  # raises a clear error if no raw data
    site = args.site or cfg["data"].get("site_id")
    if site not in stations:
        site = sorted(stations)[0]
        print(f"(config site_id not found; using first discovered station: {site})")
    print(f"Discovered {len(stations)} stations. Loading {site!r} (oracle) ...")

    s = datamod.load_station(site, weather_mode="oracle", capacity=cfg["data"].get("capacity_kw"))
    f = s.frame
    day = f["is_day"].to_numpy().astype(bool)
    results = []

    results.append(_check(
        "index is tz-aware UTC", f.index.tz is not None and str(f.index.tz) == "UTC",
        str(f.index.tz)))

    local_hour = f.index.tz_convert(HKUST_TZ).hour
    peak_hour = int(f["power"].groupby(local_hour).mean().idxmax())
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
        "p_clear & k finite on daytime",
        bool(np.isfinite(f.loc[day, ["p_clear", "k"]].to_numpy()).all())))

    k_med = float(f.loc[day, "k"].median())
    results.append(_check(
        "daytime k median in [0.2, 1.0]", 0.2 <= k_med <= 1.0,
        f"median k = {k_med:.3f}, q90 = {f.loc[day, 'k'].quantile(0.9):.3f}"))

    if not args.no_nwp:
        print(f"Loading {site!r} (nwp) for parity ...")
        s2 = datamod.load_station(site, weather_mode="nwp",
                                  capacity=cfg["data"].get("capacity_kw"))
        same_idx = s2.frame.index.equals(f.index)
        results.append(_check("nwp frame shares the oracle index", same_idx))
        if "ghi" in s2.frame.columns:
            corr = float(s2.frame.loc[day, "ghi"].corr(f.loc[day, "p_clear"]))
            results.append(_check("nwp ghi correlates with p_clear", corr > 0.3, f"corr = {corr:.2f}"))

    print(f"\n{site}: {len(f)} hours, {f.index.min().date()}..{f.index.max().date()}, "
          f"capacity {s.capacity:.2f} kW")
    if not all(results):
        sys.exit(1)
    print("All data smoke checks passed.")


if __name__ == "__main__":
    main()
