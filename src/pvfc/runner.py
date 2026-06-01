"""The Runner — one loop that scores every model identically.

For each (model x fold[ x site]) it does, day by day across the test window:
issue a day-ahead forecast using only history before that day, collect 24-h
predictions, then evaluate the concatenated test predictions against truth on
the daytime mask, with smart persistence as the skill-score reference. Output
is one tidy results row per run, keyed by a config hash.

This guarantees identical splits / masking / metrics across all models by
construction (plan §Software-architecture).
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict

import pandas as pd

from .data import StationSeries
from .evaluation import evaluate
from .models.base import Forecaster
from .models.baselines import SmartPersistence
from .splits import ExpandingWindowSplitter
from .utils import config_hash, env_fingerprint

logger = logging.getLogger("pvfc.runner")

ISSUE_HOUR = 0  # day-ahead forecasts issued at local/UTC midnight; predict next 24 h


def _day_ahead_predictions(model: Forecaster, full: pd.DataFrame,
                           test_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Roll over whole-day blocks issuing 24-h-ahead forecasts."""
    preds = []
    for day in sorted(set(test_idx.normalize())):
        day_idx = test_idx[test_idx.normalize() == day]
        history = full.loc[full.index < day]  # only the past is visible at issue time
        if len(history) == 0:
            continue
        preds.append(model.predict(history, full.loc[day_idx]))
    return pd.concat(preds) if preds else pd.DataFrame()


def run_model(
    model: Forecaster,
    series: StationSeries,
    splitter: ExpandingWindowSplitter,
    reference: Forecaster | None = None,
    daytime_only: bool = True,
    config: dict | None = None,
) -> list[dict]:
    """Run one model over all folds; return one results row per fold."""
    full = series.frame
    reference = reference if reference is not None else SmartPersistence()
    rows: list[dict] = []
    for fold in splitter.split(full.index):
        train = full.loc[fold.train_idx]
        t0 = time.perf_counter()
        model.fit(train)
        reference.fit(train)
        pred = _day_ahead_predictions(model, full, fold.test_idx)
        ref_pred = _day_ahead_predictions(reference, full, fold.test_idx)
        fit_predict_s = time.perf_counter() - t0
        if pred.empty:
            continue

        common = pred.index.intersection(ref_pred.index)
        truth = full.loc[common, "power"]
        mask = full.loc[common, "is_day"].to_numpy() if daytime_only else slice(None)

        q_pred = None
        if model.supports_quantiles:
            qcols = {float(c[1:]): pred.loc[common, c].to_numpy()[mask]
                     for c in pred.columns if c.startswith("q")}
            q_pred = qcols or None

        scores = evaluate(
            y_true=truth.to_numpy()[mask],
            y_pred=pred.loc[common, "y_hat"].to_numpy()[mask],
            capacity=series.capacity,
            reference_pred=ref_pred.loc[common, "y_hat"].to_numpy()[mask],
            q_pred=q_pred,
        )
        row = {
            "model": model.name,
            "site": series.site_id,
            "fold": fold.index,
            "n_test": int(mask.sum()) if daytime_only else len(common),
            "fit_predict_s": round(fit_predict_s, 3),
            **scores.as_row(),
        }
        if config:
            row["config_hash"] = config_hash(config)
        rows.append(row)
        skill = row.get("skill_rmse")
        logger.info("model=%s fold=%d nRMSE=%.4f skill=%.3f",
                    model.name, fold.index, row["nrmse"],
                    skill if skill is not None else float("nan"))
    return rows


def append_results(rows: list[dict], path: str) -> pd.DataFrame:
    """Append rows to the canonical parquet results store (creating it if new)."""
    import os

    df_new = pd.DataFrame(rows)
    df_new["env"] = [env_fingerprint()] * len(df_new)
    if os.path.exists(path):
        df = pd.concat([pd.read_parquet(path), df_new], ignore_index=True)
    else:
        df = df_new
    df.to_parquet(path, index=False)
    return df
