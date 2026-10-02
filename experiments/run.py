"""Single experiment runner (plan §Software-architecture).

Loads a composed YAML config, builds the dataset + splitter + models, runs each
model through the shared harness, and appends tidy rows to the parquet results
store keyed by config hash. Rows are written after EACH model, so a crash in the
middle of an overnight run loses at most the model that was running.

Usage:
    python experiments/run.py --config configs/experiment.yaml
    python experiments/run.py --site "LSK North" --weather-mode none --models lightgbm,lstm
    python experiments/run.py --max-folds 1 --results results/smoke.parquet    # quick check
    python experiments/run.py --skip-done      # skip models whose config is already stored
    python experiments/run.py --config configs/experiment.yaml --synthetic     # no download
"""
from __future__ import annotations

import os

# must be set before torch is imported (darts imports it): unsupported MPS ops
# then fall back to CPU instead of crashing a run
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from pvfc import data as datamod  # noqa: E402
from pvfc.models import build  # noqa: E402
from pvfc.runner import append_results, run_model  # noqa: E402
from pvfc.splits import ExpandingWindowSplitter  # noqa: E402
from pvfc.utils import config_hash, set_seed, setup_logging  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("pvfc.run")


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def stored_hashes(results_path: str) -> set[str]:
    """Config hashes already in the results store (rows are written only after a
    model finishes all its folds, so a stored hash means a complete run)."""
    try:
        return set(pd.read_parquet(results_path, columns=["config_hash"])["config_hash"].dropna())
    except (FileNotFoundError, KeyError, ValueError):
        return set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"))
    ap.add_argument("--synthetic", action="store_true",
                    help="use a synthetic station instead of real HK data")
    ap.add_argument("--all-models", action="store_true",
                    help="with --synthetic, also run non-baseline models (needs darts)")
    ap.add_argument("--weather-mode", choices=datamod.WEATHER_MODES, default=None,
                    help="override data.weather_mode (nwp=archived forecast, none=no weather "
                         "covariates, oracle=measured tower, diagnostics only)")
    ap.add_argument("--site", default=None, help="override data.site_id (a discovered HKUST station)")
    ap.add_argument("--models", default=None,
                    help="comma-separated config names to run (default: every model in the config)")
    ap.add_argument("--max-folds", type=int, default=None, help="override splits.max_folds")
    ap.add_argument("--results", default=str(ROOT / "results" / "results.parquet"))
    ap.add_argument("--predictions-dir", default=str(ROOT / "results" / "predictions"),
                    help="where per-timestamp predictions are written (one parquet per run)")
    ap.add_argument("--no-predictions", action="store_true", help="do not save predictions")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip models whose exact config is already in the results store "
                         "(makes long batches resumable)")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    # CLI overrides flow into cfg, so they are captured in the config hash
    # (weather settings, per-site runs and fold caps become distinct, traceable rows).
    if args.weather_mode is not None:
        cfg["data"]["weather_mode"] = args.weather_mode
    if args.site is not None:
        cfg["data"]["site_id"] = args.site
    if args.max_folds is not None:
        cfg["splits"]["max_folds"] = args.max_folds
    etl = cfg["data"].get("etl_version")
    if etl != datamod.ETL_VERSION:
        raise SystemExit(f"{args.config}: data.etl_version is {etl!r} but pvfc.data builds ETL "
                         f"v{datamod.ETL_VERSION} frames; update the config (the version is "
                         "part of every run's config hash).")
    set_seed(cfg.get("seed", 0))

    # --- data ---
    sp = dict(cfg["splits"])
    if args.synthetic:
        series = datamod.synthetic_station(days=cfg["data"].get("synthetic_days", 540))
        sp["first_test_day"] = None  # calendar anchoring is for the real stations
    else:
        d = cfg["data"]
        series = datamod.load_station(
            d["site_id"],
            weather_mode=d.get("weather_mode", "nwp"),
            capacity=d.get("capacity_kw"),
        )

    # --- splitter ---
    splitter = ExpandingWindowSplitter(
        initial_train_days=sp["initial_train_days"],
        test_days=sp["test_days"],
        embargo_days=sp["embargo_days"],
        step_days=sp.get("step_days"),
        max_folds=sp.get("max_folds"),
        day_tz=sp.get("day_tz"),
        first_test_day=sp.get("first_test_day"),
    )

    # --- models -> harness ---
    # Each model is fit ONCE per fold; the runner scores the same predictions under
    # both the daytime-only (headline) and all-hours masks, distinguished by the
    # `daytime_only` column. A fresh model is built per model entry so stateful
    # (darts) models never carry fit state across entries.
    # The synthetic path is the fast, dependency-light harness check (`make smoke`):
    # run baselines only unless --all-models is given, so it needs no darts/torch.
    models = cfg["models"]
    if args.models:
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = sorted(set(wanted) - {m["name"] for m in models})
        if unknown:
            raise SystemExit(f"--models: not in {args.config}: {unknown}")
        models = [m for m in models if m["name"] in wanted]
    if args.synthetic and not args.all_models:
        _BASELINES = {"persistence", "smart_persistence", "seasonal_naive", "climatology",
                      "cliper"}
        models = [m for m in models if m["name"] in _BASELINES]

    daytime_modes = (True, False) if cfg.get("evaluation", {}).get("daytime_only", True) \
        else (False,)
    pred_dir = None if args.no_predictions else args.predictions_dir
    done = stored_hashes(args.results) if args.skip_done else set()
    n_rows = 0
    for m in models:
        run_cfg = {"data": cfg["data"], "splits": sp, "model": m, "seed": cfg.get("seed", 0)}
        if config_hash(run_cfg) in done:
            print(f"SKIP {m['name']}: config {config_hash(run_cfg)} already in {args.results}")
            continue
        model = build(m["name"], **m.get("params", {}))
        rows = run_model(model, series, splitter, daytime_modes=daytime_modes,
                         config=run_cfg, predictions_dir=pred_dir)
        append_results(rows, args.results)  # after EACH model: a crash loses one model only
        n_rows += len(rows)
        logger.info("saved %d rows for %s -> %s", len(rows), m["name"], args.results)

    df = append_results([], args.results)
    cols = [c for c in ("model", "fold", "daytime_only", "nrmse", "nmae", "skill_rmse",
                        "fit_s", "predict_s", "peak_rss_mb") if c in df.columns]
    names = {m["name"] for m in models}
    shown = df[df["model"].str.startswith(tuple(names))] if "model" in df.columns else df
    if "site" in shown.columns:
        shown = shown[shown["site"] == series.site_id]
    print(shown[cols].to_string(index=False))
    print(f"\nWrote {n_rows} rows to {args.results}")


if __name__ == "__main__":
    main()
