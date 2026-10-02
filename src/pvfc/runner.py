"""The Runner — one loop that scores every model identically.

For each (model x fold[ x site]) it does, day by day across the test window:
issue a day-ahead forecast using only history before that day, collect 24-h
predictions, then evaluate the concatenated test predictions against truth on
the daytime mask, with smart persistence as the skill-score reference. Output
is one tidy results row per run, keyed by a config hash.

Protocol guards (see CRITICAL_REVIEW.md):
- "a day" is a *local* calendar day when the splitter carries ``day_tz`` —
  forecasts are issued at local midnight, not mid-morning;
- the ``future`` frame handed to models has all target-derived columns
  (``power``, ``k``) stripped, so reading the realized target is impossible
  rather than merely forbidden;
- the model is fit ONCE per fold and its predictions are scored under both the
  daytime-only and all-hours masks (predictions don't depend on the mask);
- ``fit_s`` and ``predict_s`` are timed separately, for the model only (the
  reference's predictions are not billed to the model under test);
- the point metrics (RMSE, MAE, MBE, R², skill) score
  ``pvfc.evaluation.point_forecast``: the mean of the quantile grid for a
  probabilistic model (squared error is minimised by the predictive mean, not the
  median), the model's own ``y_hat`` for a deterministic one. Every row records
  the rules it was scored under (``scoring_version``);
- with ``predictions_dir`` set, every scored prediction (truth, forecast,
  quantiles, the reference forecast) is kept, one parquet per run, for the
  forecast-trace figures, error-by-hour profiles and significance tests. The
  files keep the model's RAW output (``y_hat`` = its native point forecast, plus
  the quantiles), so ``experiments/rescore.py`` can rebuild every row from them
  under any scoring version without re-running a model.

This guarantees identical splits / masking / metrics across all models by
construction (plan §Software-architecture).
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import StationSeries
from .evaluation import METRIC_COLUMNS, SCORING_VERSION, evaluate, point_forecast
from .models.base import Forecaster
from .models.baselines import SmartPersistence
from .splits import ExpandingWindowSplitter
from .utils import config_hash, env_fingerprint, peak_rss_mb

logger = logging.getLogger("pvfc.runner")

#: target-derived columns a model must never see for the horizon being predicted
TARGET_COLS = ("power", "k")


def _day_ahead_predictions(model: Forecaster, full: pd.DataFrame,
                           test_idx: pd.DatetimeIndex,
                           day_tz: str | None = None) -> pd.DataFrame:
    """Roll over whole-day blocks issuing 24-h-ahead forecasts.

    Days are calendar days in ``day_tz`` (the site's local timezone) when given;
    ``history`` is everything strictly before that local midnight. The ``future``
    frame is stripped of target-derived columns (leakage guard, base.py contract).
    """
    norm = (test_idx.tz_convert(day_tz) if day_tz else test_idx).normalize()
    preds = []
    for day in sorted(set(norm)):
        day_idx = test_idx[norm == day]
        history = full.loc[full.index < day]  # only the past is visible at issue time
        if len(history) == 0:
            continue
        future = full.loc[day_idx].drop(columns=list(TARGET_COLS), errors="ignore")
        preds.append(model.predict(history, future))
    return pd.concat(preds) if preds else pd.DataFrame()


def predictions_path(predictions_dir: str | Path, site: str, weather_mode: str | None,
                     model: str, chash: str | None) -> Path:
    """One file per run, named like the results-row key (site x weather x model x hash)."""
    stem = "__".join(re.sub(r"[^A-Za-z0-9.-]+", "_", str(part)).strip("_")
                     for part in (site, weather_mode or "na", model, chash or "nohash"))
    return Path(predictions_dir) / f"{stem}.parquet"


def run_model(
    model: Forecaster,
    series: StationSeries,
    splitter: ExpandingWindowSplitter,
    reference: Forecaster | None = None,
    daytime_modes: tuple[bool, ...] = (True, False),
    config: dict | None = None,
    predictions_dir: str | Path | None = None,
) -> list[dict]:
    """Run one model over all folds; return one results row per (fold, mask).

    The model is fit and predicted once per fold; each mask in ``daytime_modes``
    re-scores the same predictions (one results row each, flagged ``daytime_only``).
    The point metrics score ``point_forecast(y_hat, q_pred)`` on the masked hours
    (the quantile-grid mean for probabilistic models, ``y_hat`` otherwise); the
    probabilistic metrics score the quantiles; each row carries ``scoring_version``.
    With ``predictions_dir``, the per-timestamp predictions of all folds are also
    written to ``predictions_path(...)`` (overwritten when the same config reruns),
    as the model's raw output: ``y_hat`` and the quantiles, never the scored mean.
    """
    full = series.frame
    reference = reference if reference is not None else SmartPersistence()
    day_tz = splitter.day_tz
    rows: list[dict] = []
    kept: list[pd.DataFrame] = []
    for fold in splitter.split(full.index):
        train = full.loc[fold.train_idx]

        t0 = time.perf_counter()
        model.fit(train)
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred = _day_ahead_predictions(model, full, fold.test_idx, day_tz)
        predict_s = time.perf_counter() - t0
        if pred.empty:
            continue

        reference.fit(train)
        ref_pred = _day_ahead_predictions(reference, full, fold.test_idx, day_tz)

        common = pred.index.intersection(ref_pred.index)
        if len(common) != len(fold.test_idx):
            raise ValueError(
                f"{model.name} fold {fold.index}: predictions cover {len(common)} of "
                f"{len(fold.test_idx)} test timestamps — models must not be scored on "
                "a subset (comparability guard)."
            )
        truth = full.loc[common, "power"].to_numpy()
        day = full.loc[common, "is_day"].to_numpy().astype(bool)
        finite = np.isfinite(truth)  # drop data gaps so NaNs can't poison the metrics
        if predictions_dir is not None:
            keep = pred.loc[common, [c for c in pred.columns
                                     if c == "y_hat" or c.startswith("q")]].copy()
            keep.insert(0, "y_true", truth)
            keep["ref_hat"] = ref_pred.loc[common, "y_hat"].to_numpy()
            keep["p_clear"] = full.loc[common, "p_clear"].to_numpy()
            keep["is_day"] = day
            keep["fold"] = fold.index
            kept.append(keep)

        for daytime_only in daytime_modes:
            mask = (day if daytime_only else np.ones(len(common), dtype=bool)) & finite

            q_pred = None
            if model.supports_quantiles:
                qcols = {float(c[1:]): pred.loc[common, c].to_numpy()[mask]
                         for c in pred.columns if c.startswith("q")}
                q_pred = qcols or None

            scores = evaluate(
                y_true=truth[mask],
                y_pred=point_forecast(pred.loc[common, "y_hat"].to_numpy()[mask], q_pred),
                capacity=series.capacity,
                reference_pred=ref_pred.loc[common, "y_hat"].to_numpy()[mask],
                q_pred=q_pred,
            )
            row = {
                "model": model.name,
                "site": series.site_id,
                "fold": fold.index,
                "daytime_only": daytime_only,
                "n_test": int(mask.sum()),
                "fit_s": round(fit_s, 3),
                "predict_s": round(predict_s, 3),
                "fit_predict_s": round(fit_s + predict_s, 3),
                "peak_rss_mb": peak_rss_mb(),
                "scoring_version": SCORING_VERSION,
                **scores.as_row(),
            }
            if config:
                row["config_hash"] = config_hash(config)
                row["weather_mode"] = (config.get("data") or {}).get("weather_mode")
            rows.append(row)
            if daytime_only:
                skill = row.get("skill_rmse")
                logger.info("model=%s fold=%d [%s..%s] nRMSE=%.4f skill=%.3f "
                            "fit=%.1fs predict=%.1fs",
                            model.name, fold.index,
                            fold.test_idx.min().date(), fold.test_idx.max().date(),
                            row["nrmse"], skill if skill is not None else float("nan"),
                            fit_s, predict_s)
    if predictions_dir is not None and kept:
        chash = config_hash(config) if config else None
        weather_mode = (config.get("data") or {}).get("weather_mode") if config else None
        out = pd.concat(kept).rename_axis("time")
        out["site"], out["model"] = series.site_id, model.name
        out["weather_mode"], out["config_hash"] = weather_mode, chash
        path = predictions_path(predictions_dir, series.site_id, weather_mode, model.name, chash)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path)
    return rows


def append_results(rows: list[dict], path: str) -> pd.DataFrame:
    """Upsert rows into the canonical parquet results store (creating it if new).

    The store is keyed by ``config_hash``: any existing rows whose ``config_hash``
    matches one in ``rows`` are dropped before the new rows are written, so
    re-running a config *replaces* its rows instead of duplicating them (the
    README's "config-hash keyed" contract). Rows without a ``config_hash`` (e.g.
    ad-hoc runs) are always appended.
    """
    import os

    if not rows:
        return pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()

    df_new = pd.DataFrame(rows)
    # a deterministic model leaves pinball/crps/picp_80/width_80 as None, i.e. all-None
    # object columns; as float NaN the store's metric columns stay float64 (pandas 3 stops
    # ignoring all-NA columns when concat picks the result dtype)
    for col in df_new.columns.intersection(METRIC_COLUMNS):
        df_new[col] = df_new[col].astype(float)
    df_new["env"] = [env_fingerprint()] * len(df_new)
    if os.path.exists(path):
        df_old = pd.read_parquet(path)
        if "config_hash" in df_new.columns and "config_hash" in df_old.columns:
            new_hashes = set(df_new["config_hash"].dropna())
            df_old = df_old[~df_old["config_hash"].isin(new_hashes)]
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_parquet(path, index=False)
    return df
