"""Pre-test check of the covariate designs (COVARIATES.md): development days only.

Three 30-day windows (4 Jan, 5 Mar and 4 May 2022), all BEFORE the first test day
(5 Jul 2022), on the six panel stations, NWP setting. The day-ahead protocol is the
runner's (_day_ahead_predictions: issue at local midnight, history strictly before);
the score is skill over CLIPER on daytime hours, each station pooled over its 90 days,
then the mean over stations. This is how the xreg settings in configs/covariates.yaml
were chosen on 10 Oct 2026 without looking at a test day.

Usage:
    python experiments/precheck_covariates.py                     # every candidate
    python experiments/precheck_covariates.py --models tirex_xreg_kc_8192,tirex_ctx2048
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pvfc import data as datamod  # noqa: E402
from pvfc.evaluation import point_forecast  # noqa: E402
from pvfc.models import build  # noqa: E402
from pvfc.models.baselines import CLIPER  # noqa: E402
from pvfc.runner import _day_ahead_predictions  # noqa: E402
from pvfc.splits import ExpandingWindowSplitter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PANEL = ["LSK North", "SQ1", "SQ567", "UG Hall6", "CYT (Station 2)", "SQ15"]
TZ = "Asia/Hong_Kong"
COMMON = dict(output_chunk_length=24, quantile=True)
TIREX = dict(arch="tirex", accept_license=True, input_chunk_length=2048)
CANDIDATES = {
    "tirex_ctx2048": dict(**TIREX, context_days=90),
    "tirex_xreg_cov_2048": dict(**TIREX, context_days=90,
                                xreg={"features": "covariates", "fit_hours": 2048}),
    "tirex_xreg_kc_2048": dict(**TIREX, context_days=90,
                               xreg={"features": "covariates+kc", "fit_hours": 2048}),
    "tirex_xreg_cov_8192": dict(**TIREX, context_days=345,
                                xreg={"features": "covariates", "fit_hours": 8192}),
    "tirex_xreg_kc_8192": dict(**TIREX, context_days=345,
                               xreg={"features": "covariates+kc", "fit_hours": 8192}),
    "timesfm_ctx512": dict(arch="timesfm", input_chunk_length=512, context_days=60),
    "timesfm_ctx8192": dict(arch="timesfm", input_chunk_length=[512, 8192], context_days=345),
    "timesfm_xreg_cov_8192": dict(arch="timesfm", input_chunk_length=[512, 8192], context_days=345,
                                  xreg={"features": "covariates", "fit_hours": 8192}),
    "timesfm_xreg_kc_8192": dict(arch="timesfm", input_chunk_length=[512, 8192], context_days=345,
                                 xreg={"features": "covariates+kc", "fit_hours": 8192}),
    "chronos2_base_ctx2048": dict(arch="chronos2", hub_model_name="amazon/chronos-2",
                                  input_chunk_length=2048, context_days=90),
}


def dev_splitter() -> ExpandingWindowSplitter:
    return ExpandingWindowSplitter(initial_train_days=150, test_days=30, embargo_days=1,
                                   step_days=60, max_folds=3, day_tz=TZ,
                                   first_test_day="2022-01-04")


def predictions(model, frame: pd.DataFrame) -> pd.DataFrame:
    """The model's and CLIPER's day-ahead forecasts on the development windows."""
    parts, ref = [], CLIPER()
    for fold in dev_splitter().split(frame.index):
        train = frame.loc[fold.train_idx]
        model.fit(train)
        pred = _day_ahead_predictions(model, frame, fold.test_idx, TZ)
        ref.fit(train)
        pred["ref_hat"] = _day_ahead_predictions(ref, frame, fold.test_idx, TZ).loc[pred.index, "y_hat"]
        parts.append(pred)
    out = pd.concat(parts)
    out["y_true"] = frame.loc[out.index, "power"]
    out["is_day"] = frame.loc[out.index, "is_day"].astype(bool)
    return out


def score(p: pd.DataFrame, capacity: float) -> dict:
    m = p["is_day"].to_numpy() & np.isfinite(p["y_true"].to_numpy())
    q = {float(c[1:]): p.loc[m, c].to_numpy() for c in p.columns if c.startswith("q")} or None
    y = p.loc[m, "y_true"].to_numpy()
    rmse = float(np.sqrt(np.mean((point_forecast(p.loc[m, "y_hat"].to_numpy(), q) - y) ** 2)))
    out = {"nrmse": rmse / capacity,
           "skill_cliper": 1 - rmse / float(np.sqrt(np.mean((p.loc[m, "ref_hat"] - y) ** 2)))}
    if q:
        out["qs_c"] = float(np.mean([np.mean(np.maximum(t * (y - v), (t - 1) * (y - v)))
                                     for t, v in q.items()])) / capacity
        out["picp80"] = float(np.mean((y >= q[0.1]) & (y <= q[0.9])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(CANDIDATES))
    ap.add_argument("--sites", default=",".join(PANEL))
    ap.add_argument("--out", default=str(ROOT / "results" / "dev_covariates" / "precheck.csv"))
    args = ap.parse_args()
    logging.disable(logging.WARNING)
    rows = []
    for name in args.models.split(","):
        for site in args.sites.split(","):
            series = datamod.load_station(site, weather_mode="nwp")
            t0 = time.perf_counter()
            p = predictions(build(name, **COMMON, **CANDIDATES[name]), series.frame)
            rows.append({"model": name, "site": site, "seconds": time.perf_counter() - t0,
                         **score(p, series.capacity)})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.groupby("model")[["skill_cliper", "nrmse", "qs_c", "picp80"]].mean()
            .sort_values("skill_cliper").round(4).to_string())


if __name__ == "__main__":
    main()
