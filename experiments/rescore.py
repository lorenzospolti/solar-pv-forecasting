"""Rebuild every metric row of the results store from the saved predictions.

Why: the panel batch of 2 Oct 2026 was scored under scoring v1, which scored the
point metrics of a probabilistic model on its median (``y_hat``). The rules are now
v2 (``pvfc.evaluation.point_forecast``): the mean of the quantile grid, because
squared error is minimised by the predictive mean, not the median (Gneiting 2011).
Re-running the batch would cost a night of compute to change nothing but the
scores, and the runner kept every prediction it scored (``results/predictions/``,
one parquet per run, raw model output). This script re-scores those files exactly
the way the runner does — the same masks (daytime or all hours, minus data gaps),
the same reference (smart persistence, ``ref_hat``), the same capacity (the ETL
sidecar) and the same ``pvfc.evaluation.evaluate`` — so every row in the store is
scored under the current rules, whenever its model ran.

  * Fit/predict timing, peak RSS and ``env`` cannot be recomputed: they are carried
    over from the store row with the same (config_hash, fold, daytime_only), like
    any other column rescoring does not produce. A predictions file with no store
    rows is still scored (timing NaN) and reported.
  * A config is scored under the masks its store rows hold (run.py scores all hours
    only when a config sets ``evaluation.daytime_only: false``); predictions with no
    store rows under both.
  * Store rows with no predictions file (e.g. the 18 synthetic rows of the 2 Oct
    store, written before the runner kept predictions) are kept unchanged; a row
    from before ``scoring_version`` existed gets version 1, the rules it was scored
    under.
  * The capacity is read from the ``.cap.json`` sidecar that ``load_station`` reads,
    never through ``load_station``: on a cache miss it would rebuild the ETL and write
    ``data/processed``. A synthetic run (``make smoke`` saves predictions too) uses
    ``synthetic_station``'s default. A run with no capacity is not rescored: the
    summary warns, and its store rows are kept.
  * ``scoring_version`` records the rules (1 = native ``y_hat``, 2 = quantile-grid
    mean). ``skill_cliper`` = 1 - RMSE / RMSE_CLIPER on the row's hours, CLIPER
    (climatology-persistence) being the same site's ``cliper`` predictions joined on
    time (``pick_cliper`` says which file when there are several). It is NaN where
    there is none, or where CLIPER misses one of the row's hours; ``report.py``
    instead scores the hours CLIPER covers, and the two rules coincide whenever CLIPER
    covers every scored hour, as a CLIPER run on the same folds does. ``skill_rmse``
    stays the skill over smart persistence.

``--point native`` re-scores under v1, which turns the rebuild into a regression
check: it must reproduce ``results/results_panel_v1_median.parquet`` to 1e-9.
Don't run it in place while a batch is appending to the same store.

Usage:
    python experiments/rescore.py --dry-run          # what would change; writes nothing
    python experiments/rescore.py                    # rewrite results/results.parquet in place
    python experiments/rescore.py --point native --out /tmp/rescored_v1.parquet
"""
from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
from collections.abc import Callable, Container, Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from pvfc import data as datamod
from pvfc.evaluation import (
    METRIC_COLUMNS,
    SCORING_VERSION,
    evaluate,
    point_forecast,
    rmse,
    skill_score,
)

ROOT = Path(__file__).resolve().parents[1]

#: --point -> scoring_version of the rows it produces ("mean" = the current rules)
SCORING_VERSIONS = {"native": 1, "mean": SCORING_VERSION}
#: identity of a results row (unique in the store)
KEY = ["config_hash", "fold", "daytime_only"]
#: output row order; within a fold the daytime (headline) row first, as the runner writes
ORDER = ["site", "weather_mode", "model", "config_hash", "fold", "daytime_only"]
#: per-run constants stored in every predictions file
META = ["site", "model", "weather_mode", "config_hash"]
#: model label of the climatology-persistence reference behind ``skill_cliper``
CLIPER_MODEL = "cliper"
#: measured by the runner, not reproducible from predictions; always in the output
MEASURED = ["fit_s", "predict_s", "fit_predict_s", "peak_rss_mb", "env"]
#: what run.py --synthetic (``make smoke``) normalises with: synthetic_station's default
SYNTHETIC_CAPACITY = float(
    inspect.signature(datamod.synthetic_station).parameters["capacity"].default)


def quantile_columns(columns: Iterable[str]) -> list[str]:
    """The quantile columns (q0.1 .. q0.9) in file order, the order the runner scored them."""
    return [c for c in columns if c.startswith("q") and c[1:].replace(".", "", 1).isdigit()]


def capacity_sidecar(site: str, weather_mode: str) -> Path:
    """The ``.cap.json`` that ``pvfc.data.load_station`` writes next to its ETL cache."""
    return datamod.PROCESSED_DIR / f"{site}__{weather_mode}__etl{datamod.ETL_VERSION}.cap.json"


def station_capacity(site: str, weather_mode: str | None) -> float | None:
    """Nameplate kW from the runner's source, the ETL's ``.cap.json`` sidecar (the
    configs leave ``data.capacity_kw`` null), read directly: ``load_station`` would
    rebuild a missing cache from raw data. The run's own weather mode first, else any
    mode's (the ETL derives the capacity from the station alone); the synthetic
    station's default; None when there is no sidecar."""
    modes = [weather_mode] if weather_mode else []
    for mode in modes + [m for m in datamod.WEATHER_MODES if m not in modes]:
        path = capacity_sidecar(site, mode)
        if path.exists():
            return float(json.loads(path.read_text())["capacity"])
    return SYNTHETIC_CAPACITY if site == "synthetic" else None


def run_meta(pred: pd.DataFrame, name: str = "") -> dict:
    """The per-run constants of a predictions file (None where absent)."""
    meta = {}
    for col in META:
        vals = pred[col].dropna().unique() if col in pred.columns else []
        if len(vals) > 1:
            raise ValueError(f"{name}: {col!r} is not constant within one run: {list(vals)}")
        meta[col] = vals[0] if len(vals) else None
    return meta


def pick_cliper(clipers: dict[Path, tuple[dict, pd.Series]], site: str,
                weather_mode: str | None, index: pd.DatetimeIndex,
                known_hashes: Container[str] = frozenset()) -> Path | None:
    """The CLIPER predictions to score a run of ``site`` against: the file covering most
    of the run's hours; on a tie the same weather_mode (CLIPER reads no weather, so any
    setting gives the same forecast), then a config in the results store
    (``known_hashes``), then the newest file, as ``report.py`` prefers (an old config's
    file stays behind when CLIPER's settings change), then the file name, so the
    choice is stable."""
    def rank(path: Path) -> tuple:
        meta, y_hat = clipers[path]
        covered = int(index.isin(y_hat.index[np.isfinite(y_hat.to_numpy(dtype=float))]).sum())
        return (-covered, meta["weather_mode"] != weather_mode,
                meta["config_hash"] not in known_hashes, -path.stat().st_mtime, path.name)

    candidates = [p for p, (meta, _) in clipers.items() if meta["site"] == site]
    return min(candidates, key=rank) if candidates else None


def score_run(pred: pd.DataFrame, capacity: float, point: str = "mean",
              cliper: pd.Series | None = None,
              daytime_modes: tuple[bool, ...] = (True, False)) -> list[dict]:
    """The runner's metric rows for one predictions file, one per (fold, mask in
    ``daytime_modes``, the runner's argument of the same name).

    Mirrors ``pvfc.runner.run_model``: mask = (``is_day``, or every hour) & finite
    truth; the quantile dict is masked first and the point forecast is built from
    the masked arrays; ``evaluate`` does the rest. ``cliper`` is CLIPER's ``y_hat``
    indexed by time (joined on the file's time index); ``skill_cliper`` is NaN unless
    it covers every scored hour.
    """
    if point not in SCORING_VERSIONS:
        raise ValueError(f"point must be one of {sorted(SCORING_VERSIONS)}, got {point!r}")
    meta = run_meta(pred)
    qcols = quantile_columns(pred.columns)
    truth_all = pred["y_true"].to_numpy(dtype=float)
    y_hat_all = pred["y_hat"].to_numpy(dtype=float)
    ref_all = pred["ref_hat"].to_numpy(dtype=float)
    day_all = pred["is_day"].to_numpy(dtype=bool)
    q_all = {c: pred[c].to_numpy(dtype=float) for c in qcols}
    cliper_all = None if cliper is None else cliper.reindex(pred.index).to_numpy(dtype=float)
    folds = pred["fold"].to_numpy()
    rows = []
    for fold in np.unique(folds):
        at = np.flatnonzero(folds == fold)  # file order = the order the runner scored in
        truth, day = truth_all[at], day_all[at]
        finite = np.isfinite(truth)
        for daytime_only in daytime_modes:
            mask = (day if daytime_only else np.ones(len(at), dtype=bool)) & finite
            q_pred = {float(c[1:]): q_all[c][at][mask] for c in qcols} or None
            y_hat = y_hat_all[at][mask]
            y_point = point_forecast(y_hat, q_pred) if point == "mean" else y_hat
            scores = evaluate(truth[mask], y_point, capacity,
                              reference_pred=ref_all[at][mask], q_pred=q_pred)
            skill_cliper = float("nan")
            if cliper_all is not None and np.isfinite(cliper_all[at][mask]).all():
                skill_cliper = skill_score(scores.rmse, rmse(truth[mask], cliper_all[at][mask]))
            rows.append({
                "model": meta["model"], "site": meta["site"], "fold": int(fold),
                "daytime_only": daytime_only, "n_test": int(mask.sum()),
                "scoring_version": SCORING_VERSIONS[point], **scores.as_row(),
                "config_hash": meta["config_hash"], "weather_mode": meta["weather_mode"],
                "skill_cliper": skill_cliper,
            })
    return rows


@dataclasses.dataclass
class Summary:
    """What a rescore did, for the printed report (and the tests)."""

    point: str
    n_files: int = 0
    n_rows: int = 0
    n_carried: int = 0  # rescored rows whose timing/env came from the store
    unmatched: list[str] = dataclasses.field(default_factory=list)  # files, no store rows
    kept: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)  # rows w/o predictions
    kept_stale: int = 0  # ... of which the config has predictions, but not that fold/mask
    no_capacity: list[str] = dataclasses.field(default_factory=list)  # files not rescored
    no_capacity_rows: int = 0  # their store rows, kept unchanged (not in ``kept``)
    cliper_used: dict[str, str] = dataclasses.field(default_factory=dict)  # file -> CLIPER file
    cliper_ignored: list[str] = dataclasses.field(default_factory=list)  # CLIPER files unused
    skill: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)  # before vs after


def _check_unique(df: pd.DataFrame, what: str) -> None:
    dup = df[df["config_hash"].notna() & df.duplicated(KEY, keep=False)]
    if len(dup):
        raise ValueError(f"{what}: {len(dup)} rows share a (config_hash, fold, daytime_only) "
                         f"key, e.g.\n{dup[KEY + ['model', 'site']].head().to_string()}")


def _skill_change(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Daytime skill over smart persistence per model x weather_mode, mean over sites
    and folds, before vs after rescoring (rows present in both)."""
    if before.empty or after.empty or "skill_rmse" not in before.columns:
        return pd.DataFrame()
    before = before.loc[before["config_hash"].notna(), KEY + ["skill_rmse"]]
    both = after.merge(before, on=KEY, suffixes=("", "_before"))
    both = both[both["daytime_only"].astype(bool)]
    table = (both.groupby(["model", "weather_mode"], dropna=False)
             [["skill_rmse_before", "skill_rmse"]].mean())
    table["change"] = table["skill_rmse"] - table["skill_rmse_before"]
    return table.rename(columns={"skill_rmse_before": "before", "skill_rmse": "after"})


def rescore(store: pd.DataFrame | None, predictions_dir: str | Path, point: str = "mean",
            capacity_fn: Callable[[str, str | None], float | None] = station_capacity,
            ) -> tuple[pd.DataFrame, Summary]:
    """The rebuilt results store, and what changed.

    ``store`` is the current store (None or empty: score the predictions alone);
    ``capacity_fn(site, weather_mode)`` gives a station's nameplate kW, or None when
    it is unknown, which leaves that run's store rows as they are (tests inject one).
    Deterministic: rows sorted by ``ORDER``, and rescoring the output again returns
    the same frame.
    """
    store = store.reset_index(drop=True) if store is not None else pd.DataFrame()
    summary = Summary(point=point)
    keyed = set(KEY) <= set(store.columns)
    hashed = store[store["config_hash"].notna()] if keyed else pd.DataFrame(columns=KEY)
    known = set(hashed["config_hash"])
    masks: dict[str, tuple[bool, ...]] = {}  # the masks the runner scored, per config
    for chash, flags in hashed.groupby("config_hash")["daytime_only"]:
        present = set(flags.astype(bool))
        masks[chash] = tuple(m for m in (True, False) if m in present)
    files = sorted(Path(predictions_dir).glob("*.parquet"))
    metas = {p: run_meta(pd.read_parquet(p, columns=META), p.name) for p in files}
    clipers = {p: (m, pd.read_parquet(p, columns=["y_hat"])["y_hat"])
               for p, m in metas.items() if m["model"] == CLIPER_MODEL}

    rows: list[dict] = []
    for p in files:
        meta = metas[p]
        capacity = capacity_fn(meta["site"], meta["weather_mode"])
        if capacity is None:
            summary.no_capacity.append(p.name)
            continue
        pred = pd.read_parquet(p)
        cl = pick_cliper(clipers, meta["site"], meta["weather_mode"], pred.index, known)
        if cl is not None:
            summary.cliper_used[p.name] = cl.name
        rows += score_run(pred, capacity, point, clipers[cl][1] if cl is not None else None,
                          masks.get(meta["config_hash"], (True, False)))
    summary.cliper_ignored = sorted({p.name for p in clipers} - set(summary.cliper_used.values()))
    scored = [p for p in files if p.name not in summary.no_capacity]
    new = pd.DataFrame(rows)
    summary.n_files, summary.n_rows = len(scored), len(new)
    if new.empty:
        new = pd.DataFrame(columns=KEY)
    for c in new.columns:
        if c in METRIC_COLUMNS or c.startswith("cov_q") or c == "skill_cliper":
            new[c] = new[c].astype(float)  # all-None columns (no quantile model) stay float
    _check_unique(new, "predictions")

    # carry what cannot be recomputed (timing, env, any column rescoring doesn't produce)
    recomputed = set(new.columns) | set(METRIC_COLUMNS) | {c for c in store.columns
                                                    if c.startswith("cov_q")}
    carried = [c for c in store.columns if c not in recomputed]
    if keyed and len(new):
        _check_unique(store, "results store")
        source = store.loc[store["config_hash"].notna(), KEY + carried]
        new = new.merge(source, on=KEY, how="left", indicator="_match")
        matched = new["_match"] == "both"
        summary.n_carried = int(matched.sum())
        hit = set(new.loc[matched, "config_hash"])
        summary.unmatched = sorted(p.name for p in scored if metas[p]["config_hash"] not in hit)
        new = new.drop(columns="_match")
    else:
        summary.unmatched = [p.name for p in scored]

    # store rows that were not rescored stay as they are (a row without config_hash never
    # matches: isin() would pair NaN with NaN)
    if keyed and len(new):
        has_hash = store["config_hash"].notna().to_numpy()
        rescored = store.set_index(KEY).index.isin(new.set_index(KEY).index) & has_hash
        kept = store[~rescored].copy()
        summary.kept_stale = int((kept["config_hash"].isin(new["config_hash"].dropna())
                                  & kept["config_hash"].notna()).sum())
    else:
        kept = store.copy()
    if len(kept):
        kept["scoring_version"] = (kept["scoring_version"].fillna(1)
                                   if "scoring_version" in kept.columns else 1)
    # a run without a capacity has predictions, so its kept rows are reported apart
    no_cap = {metas[p]["config_hash"] for p in files if p.name in summary.no_capacity} - {None}
    skipped = (kept["config_hash"].isin(no_cap).to_numpy() if "config_hash" in kept.columns
               else np.zeros(len(kept), dtype=bool))
    summary.kept, summary.no_capacity_rows = kept[~skipped], int(skipped.sum())
    summary.skill = _skill_change(store, new) if keyed else pd.DataFrame()

    parts = [df for df in (new, kept) if len(df)]
    out = pd.concat(parts, ignore_index=True) if parts else new
    for c in MEASURED:
        if c not in out.columns:
            out[c] = None if c == "env" else np.nan
    canonical = ["model", "site", "fold", "daytime_only", "n_test", *MEASURED[:-1],
                 "scoring_version", *METRIC_COLUMNS,
                 *sorted(c for c in out.columns if c.startswith("cov_q")),
                 "config_hash", "weather_mode", "env", "skill_cliper"]
    cols = [c for c in store.columns if c in out.columns]
    cols += [c for c in canonical if c in out.columns and c not in cols]
    cols += [c for c in out.columns if c not in cols]
    out = out[cols]
    if len(out):
        out["scoring_version"] = out["scoring_version"].astype("int64")
        out["_pos"] = np.arange(len(out))  # keeps ties (rows without config_hash) stable
        out = (out.sort_values(ORDER + ["_pos"], ascending=[True] * 5 + [False, True],
                               na_position="last")
               .drop(columns="_pos").reset_index(drop=True))
    return out, summary


def describe(summary: Summary, out: pd.DataFrame) -> str:
    """The printed report of a rescore."""
    def names(files: list[str]) -> str:
        return ", ".join(files[:5]) + (" ..." if len(files) > 5 else "")

    rule = ("point metrics on the quantile-grid mean" if summary.point == "mean"
            else "point metrics on the native y_hat")
    lines = [f"Rescored {summary.n_files} prediction files -> {summary.n_rows} rows "
             f"(scoring v{SCORING_VERSIONS[summary.point]}: {rule})",
             f"  timing/env carried over from the store: {summary.n_carried} rows"]
    if summary.no_capacity:
        lines.append(f"  WARNING: no capacity sidecar in {datamod.PROCESSED_DIR} for "
                     f"{len(summary.no_capacity)} files, not rescored (their "
                     f"{summary.no_capacity_rows} store rows kept unchanged): "
                     f"{names(summary.no_capacity)}")
    if summary.unmatched:
        lines.append(f"  predictions with no store rows (timing NaN): {len(summary.unmatched)} "
                     f"files: {names(summary.unmatched)}")
    kept = summary.kept
    if len(kept):
        by_site = ", ".join(f"{s}: {n}" for s, n in kept["site"].value_counts().items()) \
            if "site" in kept.columns else ""
        lines.append(f"  store rows without predictions, kept unchanged: {len(kept)} ({by_site})")
        if summary.kept_stale:
            lines.append(f"    of which {summary.kept_stale} belong to a config whose "
                         "predictions lack that fold/mask")
    else:
        lines.append("  store rows without predictions, kept unchanged: 0")
    if summary.cliper_used:
        used = sorted(set(summary.cliper_used.values()))
        lines.append(f"  skill_cliper: {len(summary.cliper_used)} files scored against "
                     f"{len(used)} CLIPER file(s): {', '.join(used)}")
    else:
        lines.append(f"  skill_cliper: NaN (no {CLIPER_MODEL!r} predictions for a scored site)")
    if summary.cliper_ignored:
        lines.append(f"    CLIPER files no run was scored against: "
                     f"{names(summary.cliper_ignored)}")
    if not summary.skill.empty:
        lines.append("Daytime skill over smart persistence, mean over sites x folds "
                     "(store before -> after):")
        lines.append(summary.skill.round(4).to_string())
    lines.append(f"Output: {len(out)} rows x {out.shape[1]} columns")
    return "\n".join(lines)


def write_store(df: pd.DataFrame, path: Path) -> None:
    """Write atomically: a crash mid-write must not leave a truncated results store."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(ROOT / "results" / "results.parquet"))
    ap.add_argument("--predictions-dir", default=str(ROOT / "results" / "predictions"))
    ap.add_argument("--out", default=None,
                    help="where to write the rebuilt store (default: --results, in place)")
    ap.add_argument("--point", choices=sorted(SCORING_VERSIONS), default="mean",
                    help="point forecast the point metrics score: mean = quantile-grid mean "
                         "(scoring v2, current), native = raw y_hat (v1, regression check)")
    ap.add_argument("--dry-run", action="store_true", help="print the summary, write nothing")
    args = ap.parse_args()

    results = Path(args.results)
    if results.exists():
        store = pd.read_parquet(results)
    else:
        print(f"No results store at {results}: scoring the predictions alone.")
        store = None
    out, summary = rescore(store, args.predictions_dir, point=args.point)
    print(describe(summary, out))
    if args.dry_run:
        print("Dry run: nothing written.")
        return
    dest = Path(args.out or args.results)
    write_store(out, dest)
    print(f"Wrote {len(out)} rows to {dest}")


if __name__ == "__main__":
    main()
