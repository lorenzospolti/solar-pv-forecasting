"""Real-time selection of the zero-shot Chronos-2 configuration (DECISIONS.md, 13 Oct 2026).

The rule, fixed before any validation forecast existed (VALIDATION.md has the design):

* Candidates (``CANDIDATES``): Chronos-2 {small, base} x context {336, 1,024, 2,048,
  8,192} h, zero-shot, listed in cost order: smaller checkpoint first, then shorter
  context.
* Windows. Fold k has a 29-day *inner window*, the days between test window k - 1 and
  fold k's embargo day (``configs/validation.yaml``), and its 30-day *test window*.
  Here the inner window of fold k is window 2k and the test window 2k + 1, so the
  windows are numbered in date order.
* Pool of fold k: every window that ends before fold k's embargo day, i.e. windows
  0..2k: the inner windows of folds 0..k and the test windows of folds 0..k - 1. At
  fold k's first issue time all of them are history. Fold 0's pool is its inner window
  (29 days), fold 8's 501 days.
* Criterion: skill over CLIPER on daytime hours, each station pooled over the pool's
  days, then the mean over the six stations (``report.Bootstrap.skill``): the headline
  skill of the thesis, restricted to the pool. Separately per weather setting.
* Tie-break: the configurations whose pooled skill is within one standard error of the
  best are tied, the standard error being that of the paired difference best - c, from
  the 7-day moving-block bootstrap over the pool's days (10,000 resamples, seed 0, as
  in the thesis tables). Of the tied configurations the first in cost order is picked.
* Output: the selected row (``SELECTED``), for every station the test window k
  forecasts of fold k's pick, in each setting. It is scored like any run by
  ``experiments/report.py``.

Declared sensitivity variants (reported in the summary, never in place of the rule):
plain argmax over the pool, and the classic one-standard-error rule (the standard
error of the best configuration's own pooled skill instead of the paired one).

Optimism of the hindsight pick (on the 270 test days only), three rough estimates that
bracket it rather than measure it:

* Tibshirani & Tibshirani (2009), adapted: mean over the nine test windows of the best
  window skill among the candidates minus the window skill of the configuration that
  is best over all test days. Conservative: about (sqrt(9) - 1) = 2 times the optimism
  when the candidates are near-equal, and it also counts real differences between
  windows (season-dependent winners).
* Analytic: if the two best configurations were equally skilful, picking the better
  one overstates its skill by E[|D|] / 2 = sd(D) / sqrt(2 pi), D their paired skill
  difference. The two-way equal-skill case: an upper bound for a choice between two,
  too low when three or more configurations are near-tied.
* Bootstrap (added 13 Oct before the selection ran; DECISIONS.md amendment): on the
  paired resamples of the test days, each candidate's resampled skill centred on its
  own skill, the mean excess of the resample's winner over its own skill. A lone or
  dominant candidate gives exactly 0; two equal ones about sd(D) / sqrt(2 pi).

Usage:
    python experiments/select_zero_shot.py            # after the runs of VALIDATION.md
    python experiments/select_zero_shot.py --dry-run  # checks and selection, no files written
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report as R  # noqa: E402  (experiments/report.py)

from pvfc.runner import predictions_path  # noqa: E402
from pvfc.utils import config_hash  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("pvfc.select")

#: the version of the rule (DECISIONS.md entry of this date); part of the selected row's hash
RULE_VERSION = "2026-10-13"
#: (config name, checkpoint, context in hours) in cost order: the tie-break order
CANDIDATES: tuple[tuple[str, str, int], ...] = (
    ("chronos2", "small", 336),
    ("chronos2_ctx1024", "small", 1024),
    ("chronos2_ctx2048", "small", 2048),
    ("chronos2_ctx8192", "small", 8192),
    ("chronos2_base", "base", 336),
    ("chronos2_base_ctx1024", "base", 1024),
    ("chronos2_base_ctx2048", "base", 2048),
    ("chronos2_base_ctx8192", "base", 8192),
)
CANDIDATE_NAMES: tuple[str, ...] = tuple(c for c, _, _ in CANDIDATES)
#: the six zero-shot configurations run before 13 Oct (the set of the 13 Oct review)
ORIGINAL_SIX: tuple[str, ...] = (
    "chronos2", "chronos2_ctx1024", "chronos2_ctx2048", "chronos2_base",
    "chronos2_base_ctx2048", "chronos2_base_ctx8192")
REFERENCE = "cliper"
SETTINGS = ("nwp", "none")
SELECTED = "chronos2_selected"
#: the declared sensitivity variants, kept out of the thesis tables
VARIANTS = {"argmax": "chronos2_selected_argmax", "classic_1se": "chronos2_selected_1se"}
INNER_DAYS, TEST_DAYS, EMBARGO_DAYS = 29, 30, 1

#: where the test forecasts of the candidates and of CLIPER are (results/<store>/predictions)
TEST_STORES = ("", "extra", "covariates", "grid")


def inner_window(fold: int) -> int:
    return 2 * fold


def test_window(fold: int) -> int:
    return 2 * fold + 1


def pool_windows(fold: int) -> list[int]:
    """The windows of fold ``fold``'s pool: all that end before its embargo day."""
    return list(range(2 * fold + 1))


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def _local_days(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(R._local_days(frame.index), index=frame.index)


def check_calendar(val: pd.DataFrame, test: pd.DataFrame, key: tuple) -> None:
    """Inner window k is the ``INNER_DAYS`` local days that end the day before fold k's
    embargo day; the test window the ``TEST_DAYS`` days from the day after it."""
    vd, td = _local_days(val), _local_days(test)
    vfolds, tfolds = sorted(val["fold"].unique()), sorted(test["fold"].unique())
    if vfolds != tfolds:
        raise SystemExit(f"{key}: inner folds {vfolds} vs test folds {tfolds}")
    for k in tfolds:
        v = vd[val["fold"].to_numpy() == k]
        t = td[test["fold"].to_numpy() == k]
        nv, nt = v.nunique(), t.nunique()
        if nv != INNER_DAYS or nt != TEST_DAYS:
            raise SystemExit(f"{key} fold {k}: {nv} inner and {nt} test days")
        gap = (t.min() - v.max()).days
        if gap != EMBARGO_DAYS + 1:
            raise SystemExit(f"{key} fold {k}: inner window ends {v.max().date()}, test "
                             f"window starts {t.min().date()} ({gap} days later, expected "
                             f"{EMBARGO_DAYS + 1})")


def combine(val_runs: dict, test_runs: dict, models: tuple[str, ...]) -> dict:
    """One frame per (model, setting, site): the inner windows (fold k -> window 2k)
    and the test windows (fold k -> window 2k + 1), in date order."""
    out = {}
    for model in models:
        for setting in SETTINGS:
            for site in sorted({k[2] for k in test_runs}):
                key = (model, setting, site)
                test = test_runs.get(key)
                if test is None and model == REFERENCE:  # CLIPER ignores weather
                    other = next(s for s in SETTINGS if s != setting)
                    test = test_runs.get((model, other, site))
                val = val_runs.get(key)
                if val is None and model == REFERENCE:
                    val = val_runs.get((model, next(s for s in SETTINGS if s != setting), site))
                if test is None or val is None:
                    raise SystemExit(f"missing {'test' if test is None else 'inner-window'} "
                                     f"forecasts for {'/'.join(key)}")
                check_calendar(val, test, key)
                if val.index.intersection(test.index).size:
                    raise SystemExit(f"{key}: an inner-window hour is also a test hour")
                v = val.assign(fold=val["fold"].to_numpy() * 2)
                t = test.assign(fold=test["fold"].to_numpy() * 2 + 1)
                out[key] = pd.concat([v, t]).sort_index()
    return out


def build_panel(runs: dict, processed_dir: Path) -> R.Panel:
    runs = R.check_runs(runs)
    return R.Panel(runs=runs, capacity=R.load_capacities(runs, processed_dir))


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
@dataclass
class Choice:
    """The selection of one fold in one setting."""

    setting: str
    fold: int
    n_days: int
    skill: dict[str, float]
    best: str
    se_paired: dict[str, float]
    se_best: float
    tied: list[str]
    pick: str
    variants: dict[str, str] = field(default_factory=dict)


def choose(estimates: dict[str, R.Estimate], order: tuple[str, ...] = CANDIDATE_NAMES,
           setting: str = "", fold: int = -1, n_days: int = 0) -> Choice:
    """The rule on one pool: ``estimates`` maps each candidate to its pooled skill
    (point and paired bootstrap samples, the same weights for all)."""
    missing = [c for c in order if c not in estimates or not estimates[c].available]
    if missing:
        raise SystemExit(f"{setting} fold {fold}: no pooled skill for {missing}")
    skill = {c: estimates[c].point for c in order}
    best = max(order, key=lambda c: skill[c])          # first in cost order on an exact tie
    se_paired = {c: (0.0 if c == best
                     else float(np.std((estimates[best] - estimates[c]).samples, ddof=1)))
                 for c in order}
    tied = [c for c in order if skill[best] - skill[c] <= se_paired[c]]
    se_best = float(np.std(estimates[best].samples, ddof=1))
    classic = next(c for c in order if skill[best] - skill[c] <= se_best)
    return Choice(setting, fold, n_days, skill, best, se_paired, se_best, tied, tied[0],
                  {"argmax": best, "classic_1se": classic})


def select(boot: R.Bootstrap, setting: str, n_folds: int,
           order: tuple[str, ...] = CANDIDATE_NAMES) -> list[Choice]:
    """Fold by fold, the rule on the pool of that fold (windows 0..2k only)."""
    choices = []
    for k in range(n_folds):
        days = np.isin(boot.fold, pool_windows(k))
        if not days.any():
            raise SystemExit(f"{setting} fold {k}: empty pool")
        est = {c: boot.skill(c, setting, reference=REFERENCE, days=days) for c in order}
        choices.append(choose(est, order, setting, k, int(days.sum())))
    return choices


def assemble(test_runs: dict, picks: dict[int, str], setting: str, label: str) -> dict:
    """The selected row of one setting: per station, test window k from pick k."""
    out = {}
    for site in sorted({k[2] for k in test_runs if k[1] == setting}):
        parts = []
        for k, model in sorted(picks.items()):
            frame = test_runs[(model, setting, site)]
            part = frame[frame["fold"].to_numpy() == k].drop(columns=["y_point"],
                                                             errors="ignore").copy()
            part["source_model"] = model
            parts.append(part)
        row = pd.concat(parts).sort_index()
        row["model"] = label
        row["config_hash"] = config_hash({"rule": RULE_VERSION, "row": label, "site": site,
                                          "setting": setting,
                                          "picks": {str(k): m for k, m in picks.items()}})
        row.index.name = "time"
        out[(label, setting, site)] = row
    return out


# --------------------------------------------------------------------------- #
# Selection bias of the hindsight pick (test days only)
# --------------------------------------------------------------------------- #
def window_skill(boot: R.Bootstrap, model: str, setting: str, window: int) -> float:
    return boot.skill(model, setting, reference=REFERENCE, days=boot.fold == window).point


def tt_optimism(boot: R.Bootstrap, setting: str, models: tuple[str, ...],
                windows: list[int]) -> dict:
    """Tibshirani & Tibshirani (2009) for a skill (higher is better): the mean over test
    windows of max_c skill_w(c) - skill_w(c*), c* the best over all test days."""
    pooled = {c: boot.skill(c, setting, reference=REFERENCE).point for c in models}
    star = max(models, key=lambda c: pooled[c])
    per_window = {w: {c: window_skill(boot, c, setting, w) for c in models} for w in windows}
    gaps = [max(s.values()) - s[star] for s in per_window.values()]
    winners = [max(models, key=lambda c: s[c]) for s in per_window.values()]
    return {"setting": setting, "candidates": list(models), "hindsight_best": star,
            "hindsight_best_skill": pooled[star], "optimism": float(np.mean(gaps)),
            "per_window_gap": [float(g) for g in gaps], "window_winners": winners}


def bootstrap_optimism(boot: R.Bootstrap, setting: str, models: tuple[str, ...]) -> dict:
    """The winner's optimism on the paired resamples of the test days: with S*_bc the
    resampled skill of c centred on its full-sample skill S_c (its own resampling bias
    removed), the mean over resamples of max_c S*_bc - S_{argmax}; and how often each
    candidate wins a resample."""
    est = {c: boot.skill(c, setting, reference=REFERENCE) for c in models}
    point = np.array([est[c].point for c in models])
    samples = np.column_stack([est[c].samples for c in models])
    centred = samples - samples.mean(axis=0) + point
    win = centred.argmax(axis=1)
    excess = centred[np.arange(len(win)), win] - point[win]
    share = np.bincount(win, minlength=len(models)) / len(win)
    return {"setting": setting, "candidates": list(models), "optimism": float(excess.mean()),
            "win_share": {c: float(x) for c, x in zip(models, share)}}


def analytic_optimism(boot: R.Bootstrap, setting: str, models: tuple[str, ...]) -> dict:
    """sd(D) / sqrt(2 pi) for the two best configurations over the test days."""
    pooled = {c: boot.skill(c, setting, reference=REFERENCE).point for c in models}
    first, second = sorted(models, key=lambda c: pooled[c], reverse=True)[:2]
    diff = boot.skill_difference((first, setting), (second, setting), reference=REFERENCE)
    sd = float(np.std(diff.samples, ddof=1))
    return {"setting": setting, "first": first, "second": second,
            "difference": diff.point, "difference_ci": list(diff.interval),
            "sd_difference": sd, "optimism": sd / math.sqrt(2 * math.pi)}


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def path_frame(choices: list[Choice], test_boot: R.Bootstrap) -> pd.DataFrame:
    rows = []
    for ch in choices:
        rows.append({
            "setting": ch.setting, "fold": ch.fold, "pool_days": ch.n_days,
            "pick": ch.pick, "best_on_pool": ch.best,
            "pick_pool_skill": ch.skill[ch.pick], "best_pool_skill": ch.skill[ch.best],
            "se_paired_pick": ch.se_paired[ch.pick], "se_best": ch.se_best,
            "tied": ",".join(ch.tied), "argmax": ch.variants["argmax"],
            "classic_1se": ch.variants["classic_1se"],
            "pick_test_skill": window_skill(test_boot, ch.pick, ch.setting, ch.fold),
            **{f"pool_skill__{c}": ch.skill[c] for c in CANDIDATE_NAMES},
            **{f"se_paired__{c}": ch.se_paired[c] for c in CANDIDATE_NAMES},
        })
    return pd.DataFrame(rows)


def table_selection_path(path: pd.DataFrame, labels: dict[int, str]) -> R.Table:
    """Appendix table: for each test window, the pool, and per setting the configuration
    picked, the best on the pool and the pick's pooled skill."""
    display, numeric = [], []
    for fold in sorted(path["fold"].unique()):
        rec = {"Test window": labels.get(int(fold), str(fold))}
        num = {"fold": int(fold)}
        for setting in SETTINGS:
            r = path[(path["fold"] == fold) & (path["setting"] == setting)]
            if r.empty:
                continue
            r = r.iloc[0]
            rec["Pool (days)"] = f"{int(r['pool_days'])}"
            name = R.setting_label(setting)
            rec[f"{name}: selected"] = R.display_name(r["pick"])
            rec[f"{name}: best on pool"] = (R.display_name(r["best_on_pool"])
                                            if r["best_on_pool"] != r["pick"] else "same")
            rec[f"{name}: pool skill"] = R._num(r["pick_pool_skill"])
            num.update({"pool_days": int(r["pool_days"]), f"pick_{setting}": r["pick"],
                        f"best_{setting}": r["best_on_pool"],
                        f"pick_pool_skill_{setting}": r["pick_pool_skill"],
                        f"best_pool_skill_{setting}": r["best_pool_skill"]})
        display.append(rec)
        numeric.append(num)
    caption = (
        "The zero-shot Chronos-2 configuration selected in real time for each test window "
        "(the selected row, Chronos-sel ‡). Pool: the earlier out-of-sample days scored for "
        f"the choice, the {INNER_DAYS}-day inner window before each earlier embargo day and "
        "every earlier test window. Selected: the first configuration in cost order (small "
        "before base, then shorter context) whose skill over CLIPER on the pool is within "
        "one bootstrap standard error of the best (paired difference, 7-day blocks); best "
        "on pool: the highest pooled skill. The eight candidates are Chronos-S, Chronos-B "
        "and their -mid, -long and -max contexts (1,024, 2,048 and 8,192 h). The rule was "
        "recorded before any inner-window forecast was made; the candidates were chosen "
        "after the main run had been seen, so the row stays post hoc.")
    return R.Table("selection_path", "selection-path", caption,
                   pd.DataFrame(display), pd.DataFrame(numeric), appendix=True)


def timing_rows(choices: dict[str, list[Choice]], selected: dict, store_paths: list[Path],
                test_runs: dict) -> pd.DataFrame:
    """Per station-fold, the timing of the configuration picked for that fold, under the
    selected row's config hash (the compute table's input). The selection itself needs
    every candidate's forecasts every day; that cost is in the summary."""
    frames = [pd.read_parquet(p) for p in store_paths if p.exists()]
    store = pd.concat(frames, ignore_index=True)
    store = store[store["daytime_only"].astype(bool)]
    rows = []
    for (label, setting, site), frame in selected.items():
        for ch in choices[setting]:
            run_hash = test_runs[(ch.pick, setting, site)]["config_hash"].iloc[0]
            hit = store[(store["config_hash"] == run_hash) & (store["fold"] == ch.fold)]
            if len(hit) != 1:
                raise SystemExit(f"{ch.pick}/{setting}/{site} fold {ch.fold}: "
                                 f"{len(hit)} timing rows")
            r = hit.iloc[0]
            rows.append({"model": label, "site": site, "fold": ch.fold, "daytime_only": True,
                         "weather_mode": setting, "config_hash": frame["config_hash"].iloc[0],
                         "source_model": ch.pick,
                         **{c: float(r[c]) for c in R.TIMING_COLUMNS if c in r}})
    return pd.DataFrame(rows)


def write_selected(selected: dict, timing: pd.DataFrame, out_dir: Path) -> None:
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for old in pred_dir.glob("*.parquet"):
        old.unlink()
    for (label, setting, site), frame in selected.items():
        frame.to_parquet(predictions_path(pred_dir, site, setting, label,
                                          frame["config_hash"].iloc[0]))
    timing.to_parquet(out_dir / "results.parquet", index=False)


def _fmt(est: R.Estimate) -> str:
    lo, hi = est.interval
    return f"{est.point:.3f} [{lo:.3f}, {hi:.3f}]"


def summary_text(choices: dict[str, list[Choice]], test_boot: R.Bootstrap,
                 tt: dict, analytic: dict, labels: dict[int, str],
                 selection_cost_s: dict[str, float], boot_opt: dict) -> tuple[str, dict]:
    lines, data = [], {"rule_version": RULE_VERSION, "settings": {}}
    for setting in SETTINGS:
        name = R.setting_label(setting)
        lines.append(f"== {name}")
        lines.append("selection path (window: pick | best on pool | pool days | tied):")
        for ch in choices[setting]:
            lines.append(f"  {labels.get(ch.fold, ch.fold):>9}: {ch.pick:24s} | "
                         f"{ch.best:24s} | {ch.n_days:3d} | {','.join(ch.tied)}")
        sel = test_boot.skill(SELECTED, setting, reference=REFERENCE)
        rows = {"selected": _fmt(sel)}
        for variant, label in VARIANTS.items():
            rows[variant] = _fmt(test_boot.skill(label, setting, reference=REFERENCE))
        for c in CANDIDATE_NAMES:
            rows[c] = _fmt(test_boot.skill(c, setting, reference=REFERENCE))
        star = tt[setting]["8"]["hindsight_best"]
        diffs = {
            f"selected - {star}": _fmt(test_boot.skill_difference(
                (SELECTED, setting), (star, setting), reference=REFERENCE)),
            "selected - chronos2": _fmt(test_boot.skill_difference(
                (SELECTED, setting), ("chronos2", setting), reference=REFERENCE)),
        }
        lines.append("test skill over CLIPER [95 %]:")
        lines += [f"  {k:28s} {v}" for k, v in rows.items()]
        lines.append("paired differences on the test days [95 %] (not the selection bias):")
        lines += [f"  {k:28s} {v}" for k, v in diffs.items()]
        lines.append("optimism of the hindsight pick, three rough estimates (a bracket, not "
                     "a measurement):")
        lines.append("  Tibshirani & Tibshirani, adapted (conservative: about 2x the optimism "
                     "when candidates are near-equal; also counts real window-to-window "
                     "differences):")
        for n in ("8", "6"):
            t = tt[setting][n]
            lines.append(f"    {n} candidates: c* = {t['hindsight_best']}, "
                         f"{t['optimism']:.4f}; per window "
                         + ", ".join(f"{g:.3f}" for g in t["per_window_gap"]))
        a = analytic[setting]
        lines.append(f"  analytic, two-way equal-skill case (an upper bound for a choice between "
                     f"two; too low if 3+ are near-tied): {a['first']} vs {a['second']}: D = "
                     f"{a['difference']:.4f} [{a['difference_ci'][0]:.4f}, "
                     f"{a['difference_ci'][1]:.4f}], sd(D) = {a['sd_difference']:.4f}, "
                     f"sd/sqrt(2 pi) = {a['optimism']:.4f}")
        for n in ("8", "6"):
            b = boot_opt[setting][n]
            top = sorted(b["win_share"].items(), key=lambda kv: -kv[1])[:3]
            lines.append(f"  bootstrap, {n} candidates (centred paired resamples): "
                         f"{b['optimism']:.4f}; share of resamples won: "
                         + ", ".join(f"{c} {x:.2f}" for c, x in top))
        lines.append(f"selection cost: the eight candidates together take "
                     f"{selection_cost_s.get(setting, math.nan):.1f} s per test window "
                     "(sum of their median fit + predict times)")
        lines.append("")
        data["settings"][setting] = {
            "path": [{"fold": ch.fold, "pick": ch.pick, "best": ch.best, "pool_days": ch.n_days,
                      "tied": ch.tied, **ch.variants} for ch in choices[setting]],
            "test_skill": rows, "differences": diffs, "tt": tt[setting],
            "analytic": a, "bootstrap": boot_opt[setting],
            "selection_cost_s": selection_cost_s.get(setting)}
    return "\n".join(lines), data


def selection_cost(store_paths: list[Path], test_runs: dict) -> dict[str, float]:
    """Sum over the candidates of their median fit + predict seconds per station-fold."""
    frames = [pd.read_parquet(p) for p in store_paths if p.exists()]
    store = pd.concat(frames, ignore_index=True)
    store = store[store["daytime_only"].astype(bool)]
    out = {}
    for setting in SETTINGS:
        total = 0.0
        for c in CANDIDATE_NAMES:
            hashes = {f["config_hash"].iloc[0] for (m, s, _), f in test_runs.items()
                      if m == c and s == setting}
            rows = store[store["config_hash"].isin(hashes)]
            total += float((rows["fit_s"] + rows["predict_s"]).median())
        out[setting] = total
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--dry-run", action="store_true", help="select and summarise, write nothing")
    args = ap.parse_args(argv)
    R.setup_logging()
    root = Path(args.root)
    results = root / "results"
    processed = root / "data" / "processed"
    test_dirs = [results / s / "predictions" if s else results / "predictions"
                 for s in TEST_STORES]
    stores = [results / s / "results.parquet" if s else results / "results.parquet"
              for s in TEST_STORES]
    keep = set(CANDIDATE_NAMES) | {REFERENCE}

    test_all = R.load_predictions(test_dirs, known_hashes=R.stored_hashes(stores))
    test_runs = {k: f for k, f in test_all.items() if k[0] in keep}
    val_runs = {k: f for k, f in R.load_predictions(results / "validation" / "predictions").items()
                if k[0] in keep}
    logger.info("test runs: %d, inner-window runs: %d", len(test_runs), len(val_runs))

    combined = build_panel(combine(val_runs, test_runs, (REFERENCE, *CANDIDATE_NAMES)),
                           processed)
    if len(combined.runs) != len(SETTINGS) * len(combined.sites) * (len(CANDIDATES) + 1):
        raise SystemExit("a candidate run failed the coverage check (see the warnings)")
    boot = R.Bootstrap(combined)
    n_folds = int(boot.fold.max() // 2 + 1)
    choices = {s: select(boot, s, n_folds) for s in SETTINGS}

    test_panel_runs = dict(test_runs)
    selected = {}
    for setting in SETTINGS:
        picks = {ch.fold: ch.pick for ch in choices[setting]}
        selected.update(assemble(test_runs, picks, setting, SELECTED))
        for variant, label in VARIANTS.items():
            vp = {ch.fold: ch.variants[variant] for ch in choices[setting]}
            test_panel_runs.update(assemble(test_runs, vp, setting, label))
    for key, frame in selected.items():
        test_panel_runs[key] = frame.assign(y_point=R.point_forecast_frame(frame))
    for key, frame in list(test_panel_runs.items()):
        if "y_point" not in frame.columns:
            test_panel_runs[key] = frame.assign(y_point=R.point_forecast_frame(frame))
    test_panel = build_panel(test_panel_runs, processed)
    test_boot = R.Bootstrap(test_panel)
    labels = R.fold_labels(R.Panel(runs=test_runs, capacity=test_panel.capacity))
    windows = sorted(int(w) for w in np.unique(test_boot.fold))

    tt = {s: {"8": tt_optimism(test_boot, s, CANDIDATE_NAMES, windows),
              "6": tt_optimism(test_boot, s, ORIGINAL_SIX, windows)} for s in SETTINGS}
    analytic = {s: analytic_optimism(test_boot, s, CANDIDATE_NAMES) for s in SETTINGS}
    boot_opt = {s: {"8": bootstrap_optimism(test_boot, s, CANDIDATE_NAMES),
                    "6": bootstrap_optimism(test_boot, s, ORIGINAL_SIX)} for s in SETTINGS}
    cost = selection_cost(stores, test_runs)
    text, data = summary_text(choices, test_boot, tt, analytic, labels, cost, boot_opt)
    print(text)
    if args.dry_run:
        return

    out = results / "validation"
    path = pd.concat([path_frame(choices[s], test_boot) for s in SETTINGS], ignore_index=True)
    path.to_csv(out / "selection_path.csv", index=False)
    (out / "selection_summary.txt").write_text(text + "\n", encoding="utf-8")
    (out / "selection_summary.json").write_text(json.dumps(data, indent=2, default=float),
                                                encoding="utf-8")
    timing = timing_rows(choices, selected, stores, test_runs)
    write_selected(selected, timing, out / "selected")
    R.write_table(table_selection_path(path, labels), root / "thesis" / "tables")
    logger.info("wrote %s, the selected row (%d files) and the selection-path table",
                out / "selection_path.csv", len(selected))


if __name__ == "__main__":
    main()
