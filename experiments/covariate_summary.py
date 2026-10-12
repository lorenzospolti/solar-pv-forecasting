"""Quick look at the covariate runs (configs/covariates.yaml) next to the reported models.

Skill over CLIPER on daytime hours, as in the master table: each station pooled over its
test days, then the mean over stations; probabilistic models are scored on the mean of
their nine quantiles (pvfc.evaluation.point_forecast). No bootstrap: the thesis tables
come from experiments/report.py.

Usage:
    python experiments/covariate_summary.py
    python experiments/covariate_summary.py --compare lightgbm,chronos2_base_ctx2048,tirex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pvfc.evaluation import point_forecast

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def _capacity(site: str, mode: str) -> float:
    cap = next(PROCESSED.glob(f"{site}__{mode}__etl*.cap.json"), None)
    return float(json.loads(cap.read_text())["capacity"]) if cap else float("nan")


def _load(dirs: list[Path]) -> pd.DataFrame:
    """The newest prediction file per (site, weather, model) found in ``dirs``."""
    files = {}
    for d in dirs:
        for f in sorted(Path(d).glob("*.parquet"), key=lambda p: p.stat().st_mtime):
            df = pd.read_parquet(f, columns=["site", "model", "weather_mode"]).iloc[:1]
            if len(df):
                files[tuple(df.iloc[0])] = f
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default=str(ROOT / "results" / "covariates" / "predictions"))
    ap.add_argument("--ref-dirs", default=",".join(str(ROOT / "results" / d / "predictions")
                                                    for d in ("", "extra")).replace("//", "/"))
    ap.add_argument("--compare", default="lightgbm,chronos2_base_ctx2048,chronos2_ft,timesfm,tirex",
                    help="reported models shown alongside (from --ref-dirs)")
    args = ap.parse_args()

    ref_dirs = [Path(p) for p in args.ref_dirs.split(",") if Path(p).exists()]
    reported = _load(ref_dirs)
    new = _load([Path(args.pred_dir)])
    cliper = {(s, w): f for (s, m, w), f in reported.items() if m == "cliper"}
    wanted = {(s, m, w): f for (s, m, w), f in reported.items()
              if m in set(args.compare.split(","))}
    rows = []
    for (site, model, mode), f in {**wanted, **new}.items():
        ref_f = cliper.get((site, mode)) or cliper.get((site, "none"))
        if ref_f is None:
            continue
        p = pd.read_parquet(f)
        ref = pd.read_parquet(ref_f)["y_hat"].rename("cliper")
        p = p.join(ref, how="inner")
        m = p["is_day"].astype(bool) & np.isfinite(p["y_true"])
        q = {float(c[1:]): p.loc[m, c].to_numpy() for c in p.columns
             if c.startswith("q") and c[1:].replace(".", "").isdigit()} or None
        y = p.loc[m, "y_true"].to_numpy()
        yhat = point_forecast(p.loc[m, "y_hat"].to_numpy(), q)
        rmse = np.sqrt(np.mean((yhat - y) ** 2))
        cap = _capacity(site, mode)
        row = {"model": model, "weather": mode, "site": site, "nrmse": rmse / cap,
               "skill_cliper": 1 - rmse / np.sqrt(np.mean((p.loc[m, "cliper"].to_numpy() - y) ** 2))}
        if q:
            row["qs_c"] = np.mean([np.mean(np.maximum(t * (y - v), (t - 1) * (y - v)))
                                   for t, v in q.items()]) / cap
            if 0.1 in q and 0.9 in q:
                row["picp80"] = np.mean((y >= q[0.1]) & (y <= q[0.9]))
        rows.append(row)
    if not rows:
        raise SystemExit(f"no prediction files in {args.pred_dir}")
    df = pd.DataFrame(rows)
    out = (df.groupby(["weather", "model"])
             .agg(stations=("site", "nunique"), skill_cliper=("skill_cliper", "mean"),
                  nrmse=("nrmse", "mean"), qs_c=("qs_c", "mean"), picp80=("picp80", "mean"))
             .sort_values(["weather", "skill_cliper"], ascending=[True, False]))
    with pd.option_context("display.width", 120):
        print(out.round(4).to_string())


if __name__ == "__main__":
    main()
