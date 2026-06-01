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
    ap.add_argument("--results", default=str(ROOT / "results" / "results.parquet"))
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 0))

    # --- data ---
    if args.synthetic:
        series = datamod.synthetic_station(days=cfg["data"].get("synthetic_days", 540))
    else:
        series = datamod.load_station(cfg["data"]["site_id"])

    # --- splitter ---
    sp = cfg["splits"]
    splitter = ExpandingWindowSplitter(
        initial_train_days=sp["initial_train_days"],
        test_days=sp["test_days"],
        embargo_days=sp["embargo_days"],
        step_days=sp.get("step_days"),
        max_folds=sp.get("max_folds"),
    )

    # --- models -> harness ---
    all_rows: list[dict] = []
    for m in cfg["models"]:
        model = build(m["name"], **m.get("params", {}))
        run_cfg = {"data": cfg["data"], "splits": sp, "model": m, "seed": cfg.get("seed", 0)}
        all_rows.extend(run_model(model, series, splitter, config=run_cfg))

    df = append_results(all_rows, args.results)
    print(df[["model", "fold", "nrmse", "nmae", "skill_rmse", "fit_predict_s"]].to_string(index=False))
    print(f"\nWrote {len(all_rows)} rows to {args.results}")


if __name__ == "__main__":
    main()
