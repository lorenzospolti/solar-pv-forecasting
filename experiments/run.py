"""Single experiment runner (plan §Software-architecture).

Loads a composed YAML config, builds the dataset + splitter + models, runs each
model through the shared harness, and appends tidy rows to the parquet results
store keyed by config hash.

Usage:
    python experiments/run.py --config configs/experiment.yaml
    python experiments/run.py --config configs/experiment.yaml --synthetic   # no download needed
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from pvfc import data as datamod
from pvfc.models import build
from pvfc.runner import append_results, run_model
from pvfc.splits import ExpandingWindowSplitter
from pvfc.utils import set_seed, setup_logging

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"))
    ap.add_argument("--synthetic", action="store_true",
                    help="use a synthetic station instead of real HK data")
    ap.add_argument("--all-models", action="store_true",
                    help="with --synthetic, also run non-baseline models (needs darts)")
    ap.add_argument("--weather-mode", choices=["oracle", "nwp"], default=None,
                    help="override data.weather_mode (oracle=measured, nwp=archived forecast)")
    ap.add_argument("--site", default=None, help="override data.site_id (a discovered HKUST station)")
    ap.add_argument("--results", default=str(ROOT / "results" / "results.parquet"))
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    # CLI overrides flow into cfg["data"], so they are captured in the config hash
    # (oracle vs nwp and per-site runs become distinct, traceable result rows).
    if args.weather_mode is not None:
        cfg["data"]["weather_mode"] = args.weather_mode
    if args.site is not None:
        cfg["data"]["site_id"] = args.site
    set_seed(cfg.get("seed", 0))

    # --- data ---
    if args.synthetic:
        series = datamod.synthetic_station(days=cfg["data"].get("synthetic_days", 540))
    else:
        d = cfg["data"]
        series = datamod.load_station(
            d["site_id"],
            weather_mode=d.get("weather_mode", "oracle"),
            capacity=d.get("capacity_kw"),
        )

    # --- splitter ---
    sp = cfg["splits"]
    splitter = ExpandingWindowSplitter(
        initial_train_days=sp["initial_train_days"],
        test_days=sp["test_days"],
        embargo_days=sp["embargo_days"],
        step_days=sp.get("step_days"),
        max_folds=sp.get("max_folds"),
        day_tz=sp.get("day_tz"),
    )

    # --- models -> harness ---
    # Each model is fit ONCE per fold; the runner scores the same predictions under
    # both the daytime-only (headline) and all-hours masks, distinguished by the
    # `daytime_only` column. A fresh model is built per model entry so stateful
    # (darts) models never carry fit state across entries.
    # The synthetic path is the fast, dependency-light harness check (`make smoke`):
    # run baselines only unless --all-models is given, so it needs no darts/torch.
    models = cfg["models"]
    if args.synthetic and not args.all_models:
        _BASELINES = {"persistence", "smart_persistence", "seasonal_naive"}
        models = [m for m in models if m["name"] in _BASELINES]

    daytime_modes = (True, False) if cfg.get("evaluation", {}).get("daytime_only", True) \
        else (False,)
    all_rows: list[dict] = []
    for m in models:
        model = build(m["name"], **m.get("params", {}))
        run_cfg = {"data": cfg["data"], "splits": sp, "model": m, "seed": cfg.get("seed", 0)}
        all_rows.extend(run_model(model, series, splitter,
                                  daytime_modes=daytime_modes, config=run_cfg))

    df = append_results(all_rows, args.results)
    cols = [c for c in ("model", "fold", "daytime_only", "nrmse", "nmae", "skill_rmse",
                        "fit_s", "predict_s", "peak_rss_mb") if c in df.columns]
    print(df[cols].to_string(index=False))
    print(f"\nWrote {len(all_rows)} rows to {args.results}")


if __name__ == "__main__":
    main()
