"""How often the ceiling of the clear-sky index binds, at 1.2 and at 1.3.

Reproduces the numbers on the ceiling in Chapter 3 (Section "Clear-sky index") and
Appendix B: for each of the six stations, the share of daytime hours whose ratio
power / p_clear lies above the ceiling, and the share of those hours that are the
first or last daytime hour of their local day. Reads the cached ETL frames
(``pvfc.data.load_station``, weather mode "none"; the power, template and daytime mask
do not depend on the weather mode).

A daytime hour is one whose apparent solar zenith at mid-hour is below 85 degrees (the
frame's ``is_day``). The ratio is taken before clipping, on daytime hours with measured
power and a positive template; these hours are the denominator of the shares. The column
``pct_above_of_all_day`` divides by all daytime hours instead, missing ones included.

Usage:
    python experiments/clip_ceiling.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pvfc import data as datamod
from pvfc.data import HKUST_TZ
from pvfc.normalize import EPS

ROOT = Path(__file__).resolve().parents[1]
CEILINGS = (1.2, 1.3)


def edge_hours(is_day: pd.Series) -> pd.Series:
    """True on the first and the last daytime hour of each local calendar day."""
    local = is_day.tz_convert(HKUST_TZ)
    day = local[local.astype(bool)]
    date = pd.Series(day.index.date, index=day.index)
    first = ~date.duplicated(keep="first")
    last = ~date.duplicated(keep="last")
    edge = pd.Series(False, index=local.index)
    edge.loc[day.index] = (first | last).to_numpy()
    return edge.tz_convert("UTC")


def station_rows(site: str) -> list[dict]:
    """One row per ceiling: daytime hours above it, and the share of those at the day's edges."""
    f = datamod.load_station(site, weather_mode="none").frame
    day = f["is_day"].astype(bool)
    edge = edge_hours(day)
    ok = day & f["power"].notna() & (f["p_clear"] > EPS)
    ratio = (f["power"] / f["p_clear"])[ok]
    rows = []
    for c in CEILINGS:
        above = ratio > c
        n_above = int(above.sum())
        rows.append({
            "station": site, "ceiling": c, "day_hours_measured": int(ok.sum()),
            "hours_above": n_above,
            "pct_above": 100 * n_above / int(ok.sum()),
            "pct_above_of_all_day": 100 * n_above / int(day.sum()),
            "pct_edge": (100 * float(edge[ratio.index][above].mean())
                         if n_above else np.nan),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"))
    args = ap.parse_args()
    stations = yaml.safe_load(open(args.config))["data"]["panel"]

    tab = pd.DataFrame([r for s in stations for r in station_rows(s)])
    pd.set_option("display.width", 120)
    print(f"Clear-sky index ceiling, ETL v{datamod.ETL_VERSION}: daytime hours (zenith < 85 deg "
          "at mid-hour) with measured power and a positive template")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print()

    print("Range over the six stations:")
    for c in CEILINGS:
        t = tab[tab["ceiling"] == c]
        lo_pct, hi_pct = t["pct_above"].min(), t["pct_above"].max()
        print(f"  ceiling {c}: above it in {lo_pct:.2f}-{hi_pct:.2f}% ({lo_pct:.1f}-{hi_pct:.1f}%) "
              f"of daytime hours with a measured ratio; {t['pct_edge'].min():.0f}-"
              f"{t['pct_edge'].max():.0f}% of those are the first or last daytime hour of their "
              f"local day [of all daytime hours: {t['pct_above_of_all_day'].min():.2f}-"
              f"{t['pct_above_of_all_day'].max():.2f}%]")
    lo, hi = CEILINGS
    ratio = (tab[tab["ceiling"] == lo].set_index("station")["hours_above"]
             / tab[tab["ceiling"] == hi].set_index("station")["hours_above"])
    print(f"  hours above {lo} / hours above {hi}: {ratio.min():.1f}-{ratio.max():.1f} times")


if __name__ == "__main__":
    main()
