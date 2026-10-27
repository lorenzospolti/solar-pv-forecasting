"""Covariate response on one development day (DEFINITIVE_RUN.md section 5, P4).

NWP vs No weather shows that a forecast changes when the covariate rows change; this shows
that it follows their content. The day's forecast GHI is multiplied by each factor (the
history and every other covariate unchanged), and the mean daytime median power must rise
with the factor. A failure starts a bug search; nothing is tuned on it.

Development days only: a ``--day`` on or after 4 Jun 2022 (the first inner window's embargo
day; the test days start 5 Jul 2022) is refused.

Usage:
    python experiments/check_covariate_response.py --config configs/dev_check.yaml \\
        --site SQ1 --day 2022-01-20 --models chronos2,timesfm3,tirex2 --factors 0.5,1,1.5
"""
from __future__ import annotations

import os

# as in run.py, before torch is imported: unsupported MPS ops fall back to the CPU
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from pvfc import data as datamod  # noqa: E402
from pvfc.models import build  # noqa: E402
from pvfc.runner import TARGET_COLS  # noqa: E402
from pvfc.utils import set_seed, setup_logging  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
#: the first day that is not a development day (configs/dev_check.yaml)
LAST_DAY = pd.Timestamp("2022-06-04")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(ROOT / "configs" / "dev_check.yaml"))
    ap.add_argument("--site", default="SQ1")
    ap.add_argument("--day", default="2022-01-20", help="local date, before 2022-06-04")
    ap.add_argument("--models", default="chronos2,timesfm3,tirex2")
    ap.add_argument("--factors", default="0.5,1,1.5", help="multipliers of the forecast GHI")
    args = ap.parse_args()

    day = pd.Timestamp(args.day)
    if day.normalize() >= LAST_DAY:
        raise SystemExit(f"--day {day.date()}: development days only (before {LAST_DAY.date()})")
    factors = sorted(float(f) for f in args.factors.split(","))
    names = [m.strip() for m in args.models.split(",") if m.strip()]
    cfg = yaml.safe_load(open(args.config))
    if cfg["data"].get("etl_version") != datamod.ETL_VERSION:
        raise SystemExit(f"{args.config}: data.etl_version is not {datamod.ETL_VERSION}")
    entries = {m["name"]: m.get("params", {}) for m in cfg["models"]}
    unknown = sorted(set(names) - set(entries))
    if unknown:
        raise SystemExit(f"--models: not in {args.config}: {unknown}")
    setup_logging()
    set_seed(cfg.get("seed", 0))

    series = datamod.load_station(args.site, weather_mode="nwp",
                                  capacity=cfg["data"].get("capacity_kw"))
    full = series.frame
    start = pd.Timestamp(str(day.date()), tz=cfg["splits"].get("day_tz") or datamod.HKUST_TZ)
    end = start + pd.Timedelta(days=1)
    history = full.loc[full.index < start]  # as the runner: only the past at issue time
    future = full.loc[(full.index >= start) & (full.index < end)].drop(
        columns=list(TARGET_COLS), errors="ignore")
    if len(future) != 24 or "ghi" not in future.columns:
        raise SystemExit(f"{args.site} {day.date()}: {len(future)} hours, columns "
                         f"{list(future.columns)}: need 24 hours with a forecast `ghi`")
    daytime = future["is_day"].to_numpy(bool)
    print(f"{args.site}, {day.date()} (local): {len(history) // 24} days of history, "
          f"{int(daytime.sum())} daytime hours; mean daytime forecast GHI "
          f"{future.loc[daytime, 'ghi'].mean():.0f} W/m2 at factor 1")

    failed = []
    for name in names:
        model = build(name, **entries[name])
        t0 = time.perf_counter()
        model.fit(history)
        fit_s = time.perf_counter() - t0
        means, times = [], []
        for factor in factors:
            fut = future.copy()
            fut["ghi"] = fut["ghi"] * factor
            t0 = time.perf_counter()
            pred = model.predict(history, fut)
            times.append(time.perf_counter() - t0)
            col = "q0.5" if "q0.5" in pred.columns else "y_hat"
            means.append(float(pred.loc[daytime, col].mean()))
        rises = bool(np.all(np.diff(means) > 0))
        if not rises:
            failed.append(name)
        shown = ", ".join(f"x{f:g}: {m:.3f}" for f, m in zip(factors, means))
        print(f"  {name:16s} mean daytime median kW  {shown}  "
              f"{'PASS' if rises else 'FAIL'}  (fit {fit_s:.1f} s, "
              f"{np.mean(times):.2f} s per forecast)", flush=True)
    print(f"Covariate response: {'PASS' if not failed else 'FAIL: ' + ', '.join(failed)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
