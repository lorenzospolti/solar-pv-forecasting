"""The covariate runs (configs/covariates.yaml) next to the reported models.

Every number is computed by experiments/report.py's own functions, so it is defined
exactly as in the thesis tables:

* skill over CLIPER on daytime hours: each station pooled over its test days, then the
  mean over stations; probabilistic models are scored on the mean of their nine
  quantiles;
* 95 % intervals from the 7-day moving-block bootstrap over the 270 test days (10,000
  resamples, seed 0), one weight matrix for every station and model, so differences
  are paired (``report.Bootstrap``);
* PICP-80 with its interval, QS / C, the median seconds per test window and the
  maximum peak RSS (``report.compute_medians``).

The paired skill differences are the ones declared in DECISIONS.md (12 Oct 2026) before
the runs: A - B, positive when A is more skilful, post hoc, no tests. A comparison is
"not available" unless both sides ran at every station.

Usage (the stores and the ETL cache live in the main checkout):
    python experiments/covariate_summary.py --root ~/Thesis/solar-pv-forecasting
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report as R  # noqa: E402  (experiments/report.py)

ROOT = Path(__file__).resolve().parents[1]
STORES = ("", "extra", "covariates")   # results/, results/extra/, results/covariates/

#: the configurations of configs/covariates.yaml, in config order (the order of their rows)
NEW_MODELS = ("tirex_xreg", "timesfm_xreg", "timesfm_ctx16k", "chronos2_base_ctx8192",
              "timesfm3", "tirex2")
#: the reported rows shown next to the new ones
REFERENCE_ROWS = (("lightgbm", "nwp"), ("lightgbm", "none"),
                  ("chronos2_base_ctx2048", "nwp"), ("chronos2_base_ctx2048", "none"),
                  ("tirex", "none"), ("tirex_ctx2048", "none"), ("timesfm", "none"))
#: univariate configurations run once, under No weather (DECISIONS.md, 12 Oct 2026):
#: with one setting only, report.detect_no_covariates cannot see that they take none
RUN_ONCE = frozenset({"timesfm_ctx16k"})
#: the declared comparisons (DECISIONS.md, 12 Oct 2026), numbered after E1-E5
DECLARED: tuple[tuple[str, tuple[str, str], tuple[str, str]], ...] = (
    ("E6", ("tirex_xreg", "nwp"), ("chronos2_base_ctx2048", "nwp")),
    ("E7", ("tirex_xreg", "nwp"), ("lightgbm", "nwp")),
    ("E8", ("tirex_xreg", "nwp"), ("tirex_ctx2048", "none")),
    ("E9", ("timesfm_xreg", "nwp"), ("chronos2_base_ctx2048", "nwp")),
    ("E10", ("timesfm_xreg", "nwp"), ("lightgbm", "nwp")),
    ("E11", ("timesfm_xreg", "nwp"), ("timesfm_ctx16k", "none")),
    ("E12", ("timesfm_ctx16k", "none"), ("chronos2_base_ctx2048", "nwp")),
    ("E13", ("timesfm_ctx16k", "none"), ("lightgbm", "nwp")),
    ("E14", ("chronos2_base_ctx8192", "nwp"), ("chronos2_base_ctx2048", "nwp")),
    ("E15", ("chronos2_base_ctx8192", "nwp"), ("lightgbm", "nwp")),
    ("E16", ("timesfm3", "nwp"), ("chronos2_base_ctx2048", "nwp")),
    ("E17", ("timesfm3", "nwp"), ("lightgbm", "nwp")),
    ("E18", ("tirex2", "nwp"), ("chronos2_base_ctx2048", "nwp")),
    ("E19", ("tirex2", "nwp"), ("lightgbm", "nwp")),
)


#: the post hoc mark and configurations of the earlier runs (report.py dropped them for the
#: definitive run, DEFINITIVE_RUN.md; kept here for these runs only)
POST_HOC_MARK = "‡"
POST_HOC_MODELS = frozenset({
    "chronos2_ctx1024", "chronos2_ctx2048", "chronos2_base", "chronos2_base_ctx2048",
    "chronos2_ft_s1", "chronos2_ft_s2", "tirex_ctx2048", "chronos2_base_ctx8192",
    "timesfm_ctx16k", "chronos2_ctx8192", "chronos2_base_ctx1024", "chronos2_selected"})
#: display names of the earlier runs' configurations shown here, as in their tables
#: (report.py labels the definitive roster only)
LABELS = {"chronos2_base_ctx2048": "Chronos-B-long", "chronos2_base_ctx8192": "Chronos-B-max",
          "timesfm": "TimesFM", "timesfm_ctx16k": "TimesFM-max", "tirex": "TiRex",
          "tirex_ctx2048": "TiRex-long"}


def _label(panel: R.Panel, model: str) -> str:
    """Display name, with dagger (no covariates) and double dagger (post hoc)."""
    mark = R.NO_COV_MARK if model in panel.no_covariates | RUN_ONCE else ""
    mark += POST_HOC_MARK if model in POST_HOC_MODELS | set(NEW_MODELS) else ""
    return f"{LABELS.get(model, R.display_name(model))} {mark}".strip()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(ROOT),
                    help="checkout holding results/ and data/processed (default: this one)")
    ap.add_argument("--out-dir", default=None,
                    help="where the two CSVs go (default: <root>/results/covariates)")
    args = ap.parse_args(argv)
    R.setup_logging()
    root = Path(args.root).expanduser().resolve()
    pred_dirs = [root / "results" / d / "predictions" for d in STORES]
    stores = [root / "results" / d / "results.parquet" for d in STORES]

    runs = R.check_runs(R.load_predictions(pred_dirs, False, R.stored_hashes(stores)))
    panel = R.Panel(runs=runs, capacity=R.load_capacities(runs, root / "data" / "processed"))
    panel.no_covariates = R.detect_no_covariates(runs)
    panel.timing = R.load_timing(stores, runs)
    summary = R.Summary(R.compute_station_metrics(panel))
    medians = R.compute_medians(panel)
    boot = R.Bootstrap(panel)
    logging.getLogger("pvfc.report").info("%d test days in the bootstrap", len(boot.days))

    order = list(REFERENCE_ROWS) + sorted({(m, s) for (m, s, _) in runs if m in NEW_MODELS},
                                          key=lambda k: (NEW_MODELS.index(k[0]), k[1] != "nwp"))
    rows = []
    for model, setting in order:
        if not panel.has(model, setting):
            continue
        skill, picp = boot.skill(model, setting), boot.picp(model, setting)
        timing = (medians.loc[(model, setting)] if (model, setting) in medians.index
                  else pd.Series(math.nan, index=medians.columns))
        rows.append({
            "model": model, "label": _label(panel, model),
            "setting": R.setting_label(setting),
            "stations": summary.n_stations(model, setting, R.skill_metric()),
            "skill_cliper": skill.point, "skill_lo": skill.interval[0],
            "skill_hi": skill.interval[1],
            "nrmse": summary.stat(model, setting, "nrmse"),
            "qs_c": summary.stat(model, setting, "ncrps") / 2.0,
            "picp_80": picp.point, "picp_80_lo": picp.interval[0],
            "picp_80_hi": picp.interval[1],
            "window_s": timing["window_s"],
            "max_peak_rss_gb": timing["max_peak_rss_mb"] / R.MB_PER_GB,
            "_skill": skill, "_picp": picp})

    comparisons = []
    for cid, a, b in DECLARED:
        a, b = R.tested_side(panel, *a), R.tested_side(panel, *b)
        est = (boot.skill_difference(a, b)
               if R.complete(panel, *a) and R.complete(panel, *b) else R.NOT_AVAILABLE)
        comparisons.append({"id": cid, "model_a": a[0], "setting_a": a[1], "model_b": b[0],
                            "setting_b": b[1], "available": est.available,
                            "skill_diff": est.point, "skill_diff_lo": est.interval[0],
                            "skill_diff_hi": est.interval[1], "_est": est})

    table = pd.DataFrame([{
        "Model": r["label"], "Setting": r["setting"], "Stations": r["stations"],
        "Skill vs CLIPER [95 %]": R._interval_cell(r["_skill"]), "nRMSE": R._num(r["nrmse"]),
        "QS / C": R._num(r["qs_c"], 4), "PICP-80 [95 %]": R._interval_cell(r["_picp"]),
        "s / window": R._num(r["window_s"], 1), "Max GB": R._num(r["max_peak_rss_gb"], 1)}
        for r in rows])
    diffs = pd.DataFrame([{
        "Id": c["id"],
        "A": f"{_label(panel, c['model_a'])} ({R.setting_label(c['setting_a'])})",
        "B": f"{_label(panel, c['model_b'])} ({R.setting_label(c['setting_b'])})",
        "Skill A - B [95 %]": R._interval_cell(c["_est"]) if c["available"] else "not available"}
        for c in comparisons])
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(table.to_string(index=False))
        print()
        print(diffs.to_string(index=False))

    out = Path(args.out_dir) if args.out_dir else root / "results" / "covariates"
    if out.is_dir():
        pd.DataFrame(rows).drop(columns=["_skill", "_picp"]).to_csv(
            out / "covariate_summary_rows.csv", index=False)
        pd.DataFrame(comparisons).drop(columns=["_est"]).to_csv(
            out / "covariate_summary_comparisons.csv", index=False)
        print(f"\nwrote {out / 'covariate_summary_rows.csv'} and "
              f"{out / 'covariate_summary_comparisons.csv'}")


if __name__ == "__main__":
    main()
