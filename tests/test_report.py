"""Tests for experiments/report.py: the significance machinery (Diebold-Mariano with
Newey-West variance, Holm, block bootstrap), the day-block bootstrap intervals, the
data-driven no-covariate detection, the input checks (coverage, duplicates, several
input folders), the table and figure layout rules, an end-to-end run on a tiny
synthetic panel built in a temporary directory, whose daily losses, CLIPER skill,
quantile score and P9 test are recomputed here by hand, and the definitive run's roster,
tested family P1-P11 and intervals-only block W1-W4 on a synthetic panel of all 13
models (DEFINITIVE_RUN.md §2 and §7)."""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pvfc.evaluation import DEFAULT_QUANTILES, point_forecast_frame

ROOT = Path(__file__).resolve().parents[1]
TABLES = ("master_skill", "chronos2_configs", "probabilistic", "compute", "significance")
APPENDIX_TABLES = ("master_skill_full", "significance_detail", "skill_by_fold",
                   "skill_by_station", "point_rule", "p2_robustness", "all_hours",
                   "panel_diagnostics")
FIGURES = ("skill_by_model", "skill_vs_compute", "reliability", "skill_by_fold",
           "skill_by_station", "weather_gain")
CAPACITY = {"Alpha": 10.0, "Beta": 20.0}
Z = {0.1: -1.28, 0.2: -0.84, 0.3: -0.52, 0.4: -0.25, 0.5: 0.0, 0.6: 0.25, 0.7: 0.52,
     0.8: 0.84, 0.9: 1.28}


@pytest.fixture(scope="module")
def report():
    name = "pvfc_report_script"
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / "report.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up while being created
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Significance machinery
# --------------------------------------------------------------------------- #
def test_dm_identical_losses_give_statistic_zero_and_p_one(report):
    loss = np.random.default_rng(0).gamma(2.0, 0.01, size=120)
    result = report.diebold_mariano(loss, loss.copy())
    assert result.statistic == 0.0
    assert result.p_value == 1.0


def test_dm_strong_constant_shift_is_significant(report):
    rng = np.random.default_rng(1)
    loss_b = rng.gamma(2.0, 0.01, size=270)
    loss_a = loss_b + 0.01 + rng.normal(0.0, 0.002, size=270)
    result = report.diebold_mariano(loss_a, loss_b)
    assert result.statistic > 0          # A has the higher loss
    assert result.p_value < 1e-6


def test_newey_west_variance_matches_hand_computation(report):
    d = np.array([1, 3, 2, 5, 4, 6, 8, 7], dtype=float)
    # n = 8 -> L = floor(8^(1/3)) = 2, Bartlett weights 2/3 and 1/3; mean 4.5, so
    # e = (-3.5, -1.5, -2.5, 0.5, -0.5, 1.5, 3.5, 2.5) and, with divisor n:
    # gamma_0 = 42 / 8, gamma_1 = 20.75 / 8, gamma_2 = 12 / 8
    expected = 42 / 8 + 2 * (2 / 3) * (20.75 / 8) + 2 * (1 / 3) * (12 / 8)
    assert expected == pytest.approx(9.708333333333, rel=1e-12)
    assert report.newey_west_variance(d) == pytest.approx(expected, rel=1e-12)
    result = report.diebold_mariano(d, np.zeros_like(d))
    assert result.variance == pytest.approx(expected, rel=1e-12)
    hln = np.sqrt((8 + 1 - 2) / 8)       # Harvey-Leybourne-Newbold factor with h = 1
    assert result.statistic == pytest.approx(4.5 / np.sqrt(expected / 8) * hln, rel=1e-12)


def test_lag_truncation_is_an_exact_integer_cube_root(report):
    # 64 ** (1/3) is 3.9999999999999996 in floating point
    assert [report._floor_cbrt(n) for n in (7, 8, 63, 64, 125, 270)] == [1, 2, 3, 4, 5, 6]


def test_holm_on_a_known_vector(report):
    p = [0.01, 0.04, 0.03, 0.005]
    # ascending 0.005, 0.01, 0.03, 0.04 times 4, 3, 2, 1 -> 0.02, 0.03, 0.06, 0.04,
    # then the running maximum makes the last 0.06
    np.testing.assert_allclose(report.holm(p), [0.03, 0.06, 0.06, 0.02])
    from statsmodels.stats.multitest import multipletests

    q = np.random.default_rng(3).uniform(0.0, 0.2, size=11)
    np.testing.assert_allclose(report.holm(q), multipletests(q, method="holm")[1])


def test_block_bootstrap_p_is_not_tiny_under_the_null(report):
    d = np.random.default_rng(5).normal(0.0, 1.0, size=270)
    assert report.block_bootstrap_p(d) > 0.05
    # a shift far beyond the sampling spread: no resample comes close
    assert report.block_bootstrap_p(d + 1.0) == pytest.approx(1 / (10_000 + 1))


def test_block_bootstrap_resamples_week_long_blocks(report):
    # Persistent errors (AR(1), phi = 0.8): one-day blocks break the dependence and
    # understate the spread of the mean, so their p-value is far smaller than with
    # the specified 7-day blocks, which the default must use.
    rng = np.random.default_rng(7)
    d = np.zeros(270)
    for t in range(1, d.size):
        d[t] = 0.8 * d[t - 1] + rng.normal()
    d += 0.25
    assert report.BLOCK_DAYS == 7
    weekly = report.block_bootstrap_p(d)
    assert weekly == report.block_bootstrap_p(d, block=7)
    assert weekly > 2 * report.block_bootstrap_p(d, block=1)


# --------------------------------------------------------------------------- #
# No-covariate detection, comparison sides
# --------------------------------------------------------------------------- #
def _forecast(seed: int, index: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.DataFrame({"y_hat": np.random.default_rng(seed).random(len(index))},
                         index=index)
    for tau in DEFAULT_QUANTILES:
        frame[f"q{tau}"] = frame["y_hat"] + (tau - 0.5)
    return frame


def test_no_covariate_models_are_detected_from_the_data(report):
    index = pd.date_range("2022-07-04 16:00", periods=48, freq="h", tz="UTC")
    runs = {}
    for site in ("A", "B"):
        same = _forecast(1, index)
        same.iloc[3, 0] = np.nan            # the same gap in both settings is still identical
        runs[("uni", "nwp", site)], runs[("uni", "none", site)] = same, same.copy()
        runs[("cov", "nwp", site)], runs[("cov", "none", site)] = (_forecast(2, index),
                                                                     _forecast(3, index))
        runs[("one_setting", "nwp", site)] = _forecast(4, index)
    runs[("partial", "nwp", "A")], runs[("partial", "none", "A")] = (_forecast(5, index),
                                                                       _forecast(5, index))
    nudged = _forecast(6, index)
    nudged.loc[index[5], "q0.9"] += 0.1     # one quantile, one hour, one station
    runs[("partial", "nwp", "B")], runs[("partial", "none", "B")] = _forecast(6, index), nudged
    assert report.detect_no_covariates(runs) == frozenset({"uni"})


def test_settingless_models_are_tested_where_t1_reports_them(report):
    frame = pd.DataFrame(index=pd.date_range("2022-07-04 16:00", periods=4, freq="h",
                                             tz="UTC"))
    keys = [("cliper", "none"), ("smart_persistence", "nwp"), ("smart_persistence", "none"),
            ("lightgbm", "none"), ("tirex", "nwp"), ("tirex", "none")]
    panel = report.Panel(runs={(m, s, "A"): frame for m, s in keys}, capacity={"A": 1.0},
                         no_covariates=frozenset({"smart_persistence", "tirex"}))
    # identical in both settings: tested on the No-weather runs, as T1 reports it
    assert report.tested_side(panel, "smart_persistence", "nwp") == ("smart_persistence", "none")
    assert report.tested_side(panel, "tirex", "none") == ("tirex", "none")
    # CLIPER ignores weather by construction: its only setting stands in (P11)
    assert report.tested_side(panel, "cliper", "nwp") == ("cliper", "none")
    # for a covariate model the setting is the hypothesis: never swapped
    assert report.tested_side(panel, "lightgbm", "nwp") == ("lightgbm", "nwp")


# --------------------------------------------------------------------------- #
# Input checks
# --------------------------------------------------------------------------- #
def _run_frame(n_folds: int = 2, **columns) -> pd.DataFrame:
    """48 h per fold of a toy run; truth missing at hour 3 (a data gap)."""
    index = pd.date_range("2022-07-04 16:00", periods=48 * n_folds, freq="h", tz="UTC")
    y = np.linspace(0.0, 1.0, len(index))
    y[3] = np.nan
    frame = pd.DataFrame({"y_true": y, "y_hat": y + 0.1, "ref_hat": y + 0.2, "is_day": True,
                          "fold": np.repeat(np.arange(n_folds), 48)}, index=index)
    for name, value in columns.items():
        frame[name] = value
    frame["y_point"] = point_forecast_frame(frame)
    return frame


def test_runs_that_miss_scored_hours_or_forecasts_are_excluded(report, caplog):
    def with_nan(column: str, row: int) -> pd.DataFrame:
        frame = _run_frame()
        frame.iloc[row, frame.columns.get_loc(column)] = np.nan
        frame["y_point"] = point_forecast_frame(frame)
        return frame

    quantile = _run_frame(**{f"q{tau}": 0.5 for tau in DEFAULT_QUANTILES})
    quantile.iloc[10, quantile.columns.get_loc("q0.9")] = np.nan
    runs = {
        ("a", "nwp", "S"): _run_frame(),
        ("b", "nwp", "S"): _run_frame(),
        ("gap_only", "nwp", "S"): with_nan("y_hat", 3),     # NaN where truth is missing: fine
        ("truncated", "nwp", "S"): _run_frame().loc[lambda f: f["fold"] == 0],
        ("nan_point", "nwp", "S"): with_nan("y_hat", 20),
        ("nan_sp", "nwp", "S"): with_nan("ref_hat", 20),
        ("nan_quantile", "nwp", "S"): quantile,
        ("a", "nwp", "T"): _run_frame(n_folds=1),           # each station has its own hours
    }
    with caplog.at_level(logging.WARNING, logger="pvfc.report"):
        kept = report.check_runs(runs)
    assert set(kept) == {("a", "nwp", "S"), ("b", "nwp", "S"), ("gap_only", "nwp", "S"),
                         ("a", "nwp", "T")}
    excluded = [r.getMessage() for r in caplog.records if "excluded" in r.getMessage()]
    assert len(excluded) == 4
    assert any("truncated/nwp/S" in m and "1 of 2 test windows" in m for m in excluded)


def test_a_partial_duplicate_never_replaces_the_full_run(report, tmp_path):
    full = _run_frame(site="S", model="m", weather_mode="nwp", config_hash="full")
    smoke = full[full["fold"] == 0].assign(config_hash="smoke")
    full.drop(columns="y_point").to_parquet(tmp_path / "S__nwp__m__full.parquet")
    smoke.drop(columns="y_point").to_parquet(tmp_path / "S__nwp__m__smoke.parquet")
    os.utime(tmp_path / "S__nwp__m__smoke.parquet", (2e9, 2e9))   # the newer file
    runs = report.load_predictions(tmp_path, known_hashes={"full", "smoke"})
    assert runs[("m", "nwp", "S")]["config_hash"].iloc[0] == "full"


def test_runs_left_out_of_the_thesis_are_skipped_on_loading(report, tmp_path):
    # TiRex and TimesFM with the weather regression ran but are not in the thesis
    kept = _run_frame(site="S", model="m", weather_mode="nwp")
    kept.drop(columns="y_point").to_parquet(tmp_path / "S__nwp__m__a.parquet")
    for model in sorted(report.EXCLUDED_MODELS):
        frame = kept.assign(model=model)
        frame.drop(columns="y_point").to_parquet(tmp_path / f"S__nwp__{model}__a.parquet")
    assert set(report.load_predictions(tmp_path)) == {("m", "nwp", "S")}
    assert report.EXCLUDED_MODELS == {"timesfm_xreg", "tirex_xreg"}
    assert not report.EXCLUDED_MODELS & {m for m, _, _ in report.MODELS}


def test_windows_are_labelled_by_the_month_holding_their_middle(report):
    def panel(starts: list[str], days: int) -> object:
        frame = pd.concat([pd.DataFrame({"fold": fold}, index=pd.date_range(
            start, periods=days * 24, freq="h", tz="Asia/Hong_Kong").tz_convert("UTC"))
            for fold, start in enumerate(starts)])
        return report.Panel(runs={("m", "none", "A"): frame}, capacity={"A": 1.0})

    # like the panel's last windows: 30 Jun - 29 Jul 2023 is a July window
    assert report.fold_labels(panel(["2023-04-30", "2023-06-30", "2023-10-28"], 30)) == {
        0: "May 2023", 1: "Jul 2023", 2: "Nov 2023"}
    # two windows in one month: their start dates
    assert report.fold_labels(panel(["2023-07-01", "2023-07-15"], 10)) == {
        0: "1 Jul 2023", 1: "15 Jul 2023"}


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def test_column_widths_follow_printed_width_not_length(report):
    def dashes(frame: pd.DataFrame) -> list[int]:
        return [len(c) for c in report.pipe_table(frame).splitlines()[1].strip("|").split("|")]

    short = pd.DataFrame({"Comparison": ["LightGBM vs SARIMA"], "p": ["<0.001"],
                          "q": ["0.0010"]})
    long = short.assign(Comparison="NWP: LightGBM vs SARIMA, and many more words to wrap")
    # a text column claims its longest word, however long its cells: the rest wraps
    assert dashes(short) == dashes(long)
    # "<" is wider than a digit, so "<0.001" needs more room than "0.0010"
    assert dashes(short)[1] > dashes(short)[2]
    # a header may break after a hyphen ("PICP-" / "80"), a model name may not
    assert dashes(pd.DataFrame({"PICP-80": ["0.751"]})) < dashes(
        pd.DataFrame({"PICPx80": ["0.751"]}))
    assert dashes(pd.DataFrame({"M": ["Chronos-2"]})) == [
        round(report.DASHES_PER_DIGIT * report._print_width("Chronos-2"))]


def test_direct_labels_avoid_joining_lines(report):
    with plt.rc_context(report.STYLE):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        line = [(5.0, 5.0), (9.5, 5.0)]     # runs right of the point, where a label goes first
        ax.plot(*zip(*line))
        fig.canvas.draw()
        report._place_labels(ax, [("m", 5.0, 5.0, "Label")], [(5.0, 5.0)], [], [line])
        box = ax.texts[0].get_window_extent(fig.canvas.get_renderer())
        p, q = ax.transData.transform(line)
        assert box.count_contains(p + np.linspace(0, 1, 400)[:, None] * (q - p)) == 0
        plt.close(fig)


def test_station_dots_are_not_the_reference_grey(report):
    assert report.STATION_DOT.lower() != report.FAMILIES[report.REFERENCE][1].lower()


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
def _write_panel(report, root: Path) -> dict[str, Path]:
    """2 stations (capacities 10 and 20 kW) x 2 settings x 4 models, 2 folds: smart
    persistence; a LightGBM-like quantile model that uses weather; a quantile model
    named TiRex-2 that does not (to test the detection); a deterministic CLIPER that
    does not. The quantiles
    are skewed (mean != median) and Beta misses four daytime hours on one day, so
    the point forecast, the capacity and the station weighting all move the losses."""
    pred_dir, processed = root / "predictions", root / "processed"
    pred_dir.mkdir()
    processed.mkdir()
    rng = np.random.default_rng(42)
    days = [*range(10), *range(40, 50)]                    # two 10-day test windows
    start = pd.Timestamp("2022-07-05", tz="Asia/Hong_Kong")
    local = pd.DatetimeIndex([start + pd.Timedelta(days=d, hours=h)
                              for d in days for h in range(24)])
    hour = local.hour.to_numpy()
    fold = np.repeat([0, 1], 10 * 24)
    shape = np.clip(np.sin(np.pi * (hour - 6) / 12), 0.0, None)
    results = []
    timing = {"smart_persistence": 0.01, "lightgbm": 30.0, "tirex2": 3.0, "cliper": 0.1}
    for site, cap in CAPACITY.items():
        p_clear = cap * shape
        cloud = np.repeat(rng.uniform(0.3, 1.0, size=len(days)), 24)
        y_true = p_clear * cloud
        y_true[30] = np.nan                                 # a data gap (day 1, 06:00)
        if site == "Beta":
            y_true[3 * 24 + 9:3 * 24 + 13] = np.nan         # day 3, 09:00-12:00
        ref = np.concatenate([p_clear[:24] * 0.7, y_true[:-24]])
        ref = np.where(np.isfinite(ref), ref, p_clear * 0.7)
        base = pd.DataFrame({"y_true": y_true, "ref_hat": ref, "p_clear": p_clear,
                             "is_day": shape > 0, "fold": fold, "site": site},
                            index=local.tz_convert("UTC").rename("time"))
        truth = np.nan_to_num(y_true)
        tirex = truth + rng.normal(0, 0.12 * cap, size=len(base)) * (shape > 0)
        cliper = 0.5 * ref + 0.5 * 0.65 * p_clear
        for setting in ("nwp", "none"):
            sigma = (0.06 if setting == "nwp" else 0.15) * cap
            lgbm = truth + rng.normal(0, sigma, size=len(base)) * (shape > 0)
            for model, centre, spread in (("smart_persistence", ref, None),
                                          ("lightgbm", lgbm, sigma),
                                          ("tirex2", tirex, 0.12 * cap),
                                          ("cliper", cliper, None)):
                frame = base.copy()
                frame["y_hat"] = np.clip(centre, 0.0, None)
                if spread is not None:
                    for tau, z in Z.items():
                        skew = 1.6 if z > 0 else 1.0        # a longer upper tail
                        frame[f"q{tau}"] = np.clip(centre + z * skew * spread * (shape > 0),
                                                   0.0, None)
                chash = f"{model[:4]}{setting}{site[:2]}"
                frame["model"], frame["weather_mode"], frame["config_hash"] = (
                    model, setting, chash)
                frame.to_parquet(pred_dir / f"{site}__{setting}__{model}__{chash}.parquet")
                for f in (0, 1):
                    for daytime_only in (True, False):
                        results.append({"model": model, "site": site, "fold": f,
                                        "daytime_only": daytime_only, "n_test": 100,
                                        "fit_s": timing[model] + f, "predict_s": 0.5,
                                        "fit_predict_s": timing[model] + f + 0.5,
                                        "peak_rss_mb": 300.0, "config_hash": chash,
                                        "weather_mode": setting})
            report.capacity_sidecar(site, setting, processed).write_text(
                json.dumps({"capacity": cap}))
    pd.DataFrame(results).to_parquet(root / "results.parquet", index=False)
    return {"root": root, "predictions": pred_dir, "processed": processed,
            "results": root / "results.parquet"}


@pytest.fixture(scope="module")
def panel_inputs(report, tmp_path_factory):
    return _write_panel(report, tmp_path_factory.mktemp("panel"))


def _main(report, inputs: dict[str, Path], out: Path) -> list[Path]:
    return report.main([
        "--results", str(inputs["results"]), "--predictions-dir", str(inputs["predictions"]),
        "--processed-dir", str(inputs["processed"]), "--tables-dir", str(out / "tables"),
        "--figures-dir", str(out / "figures")])


def _frames(inputs: dict[str, Path], model: str, setting: str) -> dict[str, pd.DataFrame]:
    frames = (pd.read_parquet(p) for p in
              sorted(inputs["predictions"].glob(f"*__{setting}__{model}__*.parquet")))
    return {f["site"].iloc[0]: f for f in frames}


def _daily_loss(inputs: dict[str, Path], model: str, setting: str) -> pd.Series:
    """By hand: per station and Hong Kong calendar day, the mean of ((point - y) / C)^2
    over the scored daytime hours; then the mean over the stations of each day."""
    per_site = {}
    for site, frame in _frames(inputs, model, setting).items():
        y = frame["y_true"].to_numpy()
        mask = frame["is_day"].to_numpy() & np.isfinite(y)
        loss = ((point_forecast_frame(frame) - y) / CAPACITY[site])[mask] ** 2
        per_site[site] = pd.Series(loss).groupby(frame.index[mask].tz_convert(
            "Asia/Hong_Kong").date).mean()
    return pd.DataFrame(per_site).mean(axis=1).sort_index()


def test_end_to_end_writes_every_table_and_figure(report, panel_inputs, tmp_path):
    written = _main(report, panel_inputs, tmp_path)
    tables, figures = tmp_path / "tables", tmp_path / "figures"

    appendix = tables / "appendix"
    # no Chronos-2 run here: T2 is skipped, not written empty
    assert not (tables / "chronos2_configs.md").exists()
    main_tables = [s for s in TABLES if s != "chronos2_configs"]
    for folder, stems in ((tables, main_tables), (appendix, APPENDIX_TABLES)):
        for stem in stems:
            assert (folder / f"{stem}.csv").exists()
            lines = (folder / f"{stem}.md").read_text(encoding="utf-8").splitlines()
            assert lines[0].startswith(": ") and lines[0].endswith("}")
            assert f"{{#tbl:{stem.replace('_', '-')}}}" in lines[0]
            assert lines[1] == "" and lines[2].startswith("| ")
            # captions are typeset: true minus, times, superscript, never ASCII stand-ins
            assert "^" not in lines[0] and " - " not in lines[0] and " x " not in lines[0]
            assert "pre-specified" not in lines[0].lower()
    # the appendix tables are not also in the main folder
    assert not any((tables / f"{stem}.md").exists() for stem in APPENDIX_TABLES)
    assert (tables / "metrics_long.csv").exists()
    for stem in FIGURES:
        for ext in ("pdf", "png"):
            path = figures / f"{stem}.{ext}"
            assert path.exists() and path.stat().st_size > 0
            assert path in written
    probabilistic = (tables / "probabilistic.md").read_text(encoding="utf-8")
    assert "| QS / C" in probabilistic and "twice the QS" in probabilistic
    assert "CRPS /" not in probabilistic and "CRPS skill" not in probabilistic

    master = pd.read_csv(appendix / "master_skill_full.csv").set_index("model")
    # this TiRex-2, CLIPER and smart persistence never change with the weather: shown once
    assert set(master.index[master["no_covariates"]]) == {"smart_persistence", "tirex2",
                                                          "cliper"}
    assert np.isnan(master.loc["tirex2", "skill_sp_nwp_mean"])
    assert np.isfinite(master.loc["tirex2", "skill_sp_none_mean"])
    assert (master["n_stations_none"] == 2).all()
    master_md = (appendix / "master_skill_full.md").read_text(encoding="utf-8")
    assert "TiRex-2 †" in master_md and "Skill vs CLIPER (NWP)" in master_md
    assert "fewer than" not in master_md
    # T1: skill over the headline reference (CLIPER) with a 95 % interval
    headline = pd.read_csv(tables / "master_skill.csv").set_index("model")
    assert headline.loc["lightgbm", "skill_nwp"] == pytest.approx(
        master.loc["lightgbm", "skill_cliper_nwp_mean"], rel=1e-12)
    assert headline.loc["lightgbm", "skill_nwp_lo"] < headline.loc["lightgbm", "skill_nwp"] \
        < headline.loc["lightgbm", "skill_nwp_hi"]
    t1 = (tables / "master_skill.md").read_text(encoding="utf-8")
    assert "Skill (NWP)" in t1 and "nRMSE (No weather)" in t1 and "Skill vs" not in t1
    assert "skill over CLIPER" in t1 and "@tbl:master-skill-full" in t1
    assert "10,000 resamples" in t1 and "7-day moving-block bootstrap" in t1
    # the reference's own row: 0, no interval; † models: an em dash under NWP
    cliper_row = next(line for line in t1.splitlines() if line.startswith("| CLIPER"))
    assert [c.strip() for c in cliper_row.strip("|").split("|")][1:3] == ["—", "0"]
    tirex_row = next(line for line in t1.splitlines() if line.startswith("| TiRex-2"))
    assert tirex_row.split("|")[2].strip() == "—" and "[" in tirex_row.split("|")[3]
    assert list(headline.index) == [m for m in report.MAIN_MODELS if m in headline.index]

    # headline = mean over stations of 1 - RMSE / RMSE_SP pooled over each station's
    # scored daytime hours, scored on the mean of the nine quantiles; likewise vs CLIPER
    lgbm, cliper = _frames(panel_inputs, "lightgbm", "nwp"), _frames(panel_inputs, "cliper", "nwp")
    skills, skills_cliper = [], []
    for site, frame in lgbm.items():
        mask = frame["is_day"].to_numpy() & np.isfinite(frame["y_true"].to_numpy())
        y = frame["y_true"].to_numpy()[mask]
        rmse = np.sqrt(np.mean((point_forecast_frame(frame)[mask] - y) ** 2))
        rmse_sp = np.sqrt(np.mean((frame["ref_hat"].to_numpy()[mask] - y) ** 2))
        rmse_cliper = np.sqrt(np.mean((cliper[site]["y_hat"].to_numpy()[mask] - y) ** 2))
        skills.append(1 - rmse / rmse_sp)
        skills_cliper.append(1 - rmse / rmse_cliper)
    assert master.loc["lightgbm", "skill_sp_nwp_mean"] == pytest.approx(np.mean(skills))
    assert master.loc["lightgbm", "skill_sp_nwp_min"] == pytest.approx(min(skills))
    assert master.loc["lightgbm", "skill_cliper_nwp_mean"] == pytest.approx(
        np.mean(skills_cliper))

    # P9 by hand: Hong Kong days, losses / C, equal weight per station, quantile mean
    sig = pd.read_csv(appendix / "significance_detail.csv").set_index("id")
    loss_a, loss_b = _daily_loss(panel_inputs, "lightgbm", "nwp"), _daily_loss(
        panel_inputs, "lightgbm", "none")
    d = (loss_a - loss_b).to_numpy()
    n, e = d.size, d - d.mean()
    lags = 2                                                 # floor(20 ** (1/3))
    variance = e @ e / n + sum(2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
                               for k in range(1, lags + 1))
    dm = d.mean() / np.sqrt(variance / n) * np.sqrt((n - 1) / n)
    assert sig.loc["P9", "n_days"] == n == 20
    assert sig.loc["P9", "loss_a"] == pytest.approx(loss_a.mean(), rel=1e-10)
    assert sig.loc["P9", "loss_b"] == pytest.approx(loss_b.mean(), rel=1e-10)
    assert sig.loc["P9", "dm_stat"] == pytest.approx(dm, rel=1e-10)
    assert sig.loc["P9", "verdict"] == "A better"           # LightGBM with NWP vs without

    # P11 (CLIPER vs SP) runs on the No-weather files both models are reported under, as
    # does the no-covariate side of P8; the Holm family is exactly the comparisons that
    # could be run
    assert list(sig.index[sig["available"]]) == ["P8", "P9", "P11"]
    assert sig.loc["P11", ["setting_a", "setting_b"]].tolist() == ["none", "none"]
    assert sig.loc["P8", ["setting_a", "setting_b"]].tolist() == ["nwp", "none"]
    from statsmodels.stats.multitest import multipletests
    tested = ["P8", "P9", "P11"]
    np.testing.assert_allclose(sig.loc[tested, "p_holm"],
                               multipletests(sig.loc[tested, "p_dm"], method="holm")[1])
    significance = (appendix / "significance_detail.md").read_text(encoding="utf-8")
    assert "adjusts over the 3 comparisons" in significance
    assert "LightGBM, NWP vs TiRex-2 †" in significance
    assert "over the same 20 days" in significance and "| Days" not in significance
    assert "((forecast − truth) / C)²" in significance and "loss A − loss B" in significance
    assert "| P9  | LightGBM: NWP vs No weather " in significance
    assert "No weather: CLIPER † vs Smart persistence †" in significance
    assert "not available" in significance and "| n/a" in significance
    # T5: the skill difference A - B over CLIPER, paired, with Holm p and verdict
    t5 = pd.read_csv(tables / "significance.csv").set_index("id")
    p9 = t5.loc["P9"]
    assert p9["skill_diff"] == pytest.approx(
        master.loc["lightgbm", "skill_cliper_nwp_mean"]
        - master.loc["lightgbm", "skill_cliper_none_mean"], rel=1e-10)
    assert p9["skill_diff_lo"] <= p9["skill_diff"] <= p9["skill_diff_hi"]
    t5_md = (tables / "significance.md").read_text(encoding="utf-8")
    assert "| ID" in t5_md and "Skill difference (A − B)" in t5_md
    assert "form the tested family" in t5_md and "@tbl:significance-detail" in t5_md
    assert "earlier runs" not in t5_md and "main run" not in t5_md


def test_an_incomplete_run_is_excluded_and_the_captions_say_so(report, panel_inputs, tmp_path,
                                                               caplog):
    inputs = dict(panel_inputs, predictions=tmp_path / "predictions")
    shutil.copytree(panel_inputs["predictions"], inputs["predictions"])
    path = next(inputs["predictions"].glob("Beta__nwp__lightgbm__*.parquet"))
    frame = pd.read_parquet(path)
    frame[frame["fold"] == 0].to_parquet(path)              # what a --max-folds 1 run leaves
    with caplog.at_level(logging.WARNING, logger="pvfc.report"):
        _main(report, inputs, tmp_path)
    assert any("excluded lightgbm/nwp/Beta" in r.getMessage() for r in caplog.records)

    master = pd.read_csv(tmp_path / "tables" / "master_skill.csv").set_index("model")
    by_station = pd.read_csv(tmp_path / "tables" / "appendix" / "skill_by_station.csv")
    alpha = by_station.query("model == 'lightgbm' and setting == 'nwp'")["Alpha"].item()
    assert master.loc["lightgbm", "n_stations_nwp"] == 1
    assert master.loc["lightgbm", "skill_nwp"] == pytest.approx(alpha)   # over CLIPER
    for stem in ("master_skill", "probabilistic", "appendix/master_skill_full",
                 "appendix/skill_by_station", "appendix/skill_by_fold", "appendix/all_hours"):
        caption = (tmp_path / "tables" / f"{stem}.md").read_text(encoding="utf-8")
        assert "fewer than the 2 stations" in caption and "LightGBM, NWP (1)" in caption


def test_per_model_figures_share_one_order(report, panel_inputs, tmp_path, monkeypatch):
    runs = report.check_runs(report.load_predictions(panel_inputs["predictions"]))
    panel = report.Panel(runs=runs, capacity=report.load_capacities(
        runs, panel_inputs["processed"]))
    panel.no_covariates = report.detect_no_covariates(runs)
    summary = report.Summary(report.compute_station_metrics(panel))
    order = report.figure_order(panel, summary)
    # the skill order is not the thesis (family) order here, so the test can tell them apart
    assert order != report.model_order(order)

    captured: dict[str, plt.Figure] = {}
    monkeypatch.setattr(report, "_save", lambda fig, _, stem: captured.update({stem: fig}) or [])
    with plt.rc_context(report.STYLE):
        report.fig_skill_by_model(panel, summary, report.Bootstrap(panel, n_boot=200), tmp_path)
        report.fig_reliability(panel, summary, tmp_path)
        report.fig_skill_by_fold(panel, summary, report.compute_fold_skill(panel),
                                 report.fold_labels(panel), tmp_path)
        report.fig_skill_by_station(panel, summary, tmp_path)
        report.fig_weather_gain(panel, summary, tmp_path)
    by_label = {report._figure_label(panel, m): m for m in panel.models}
    by_label.update({report.display_name(m): m for m in panel.models})
    sequences = [[by_label[t.get_text()] for t in ax.get_yticklabels()]
                 for stem in ("skill_by_model", "skill_by_fold", "skill_by_station",
                              "weather_gain") for ax in captured[stem].axes
                 if ax.get_label() != "<colorbar>"]
    sequences.append([by_label[ax.get_title(loc="left")]
                      for ax in captured["reliability"].axes if ax.get_visible()])
    plt.close("all")
    assert any(len(s) >= 3 for s in sequences)
    for sequence in sequences:
        assert sequence == sorted(sequence, key=order.index)


# --------------------------------------------------------------------------- #
# Several input folders
# --------------------------------------------------------------------------- #
def test_several_prediction_folders_and_stores_load_as_one(report, panel_inputs, tmp_path,
                                                          caplog):
    first, second, missing = tmp_path / "panel", tmp_path / "extra", tmp_path / "missing"
    first.mkdir()
    second.mkdir()
    for path in panel_inputs["predictions"].glob("*.parquet"):
        shutil.copy(path, first if path.name.startswith("Alpha") else second)
    single = report.load_predictions(panel_inputs["predictions"])
    with caplog.at_level(logging.WARNING, logger="pvfc.report"):
        split = report.load_predictions([first, second, missing])
    assert any("no predictions folder" in r.getMessage() and "missing" in r.getMessage()
               for r in caplog.records)
    assert set(split) == set(single)
    for key, frame in single.items():
        pd.testing.assert_frame_equal(split[key], frame)
    assert set(report.load_predictions(str(panel_inputs["predictions"]))) == set(single)

    results = panel_inputs["results"]
    assert report.stored_hashes([results, tmp_path / "none.parquet"]) == \
        report.stored_hashes(results)
    pd.testing.assert_frame_equal(
        report.load_timing([results, tmp_path / "none.parquet"], single),
        report.load_timing(results, single))

    # the command line takes several folders and stores, and gives the same tables
    out_one, out_two = tmp_path / "one", tmp_path / "two"
    _main(report, panel_inputs, out_one)
    report.main(["--results", str(results), str(tmp_path / "none.parquet"),
                 "--predictions-dir", str(first), str(second), str(missing),
                 "--processed-dir", str(panel_inputs["processed"]),
                 "--tables-dir", str(out_two / "tables"),
                 "--figures-dir", str(out_two / "figures")])
    for stem in ("master_skill", "compute", "significance"):
        pd.testing.assert_frame_equal(pd.read_csv(out_one / "tables" / f"{stem}.csv"),
                                      pd.read_csv(out_two / "tables" / f"{stem}.csv"))


# --------------------------------------------------------------------------- #
# Day-block bootstrap
# --------------------------------------------------------------------------- #
def _panel(report, inputs: dict[str, Path]):
    runs = report.check_runs(report.load_predictions(inputs["predictions"]))
    panel = report.Panel(runs=runs, capacity=report.load_capacities(runs, inputs["processed"]))
    panel.no_covariates = report.detect_no_covariates(runs)
    return panel, report.Summary(report.compute_station_metrics(panel))


def test_day_weights_are_the_block_bootstrap_resamples(report):
    d = np.random.default_rng(11).normal(size=50)
    weights = report.day_weights(50, block=7, n_boot=500, seed=3)
    assert weights.shape == (500, 50) and weights.dtype.kind == "i"
    assert (weights.sum(axis=1) == 50).all()
    index = report.block_indices(50, block=7, n_boot=500, seed=3)
    np.testing.assert_allclose(weights @ d / 50, d[index].mean(axis=1), rtol=1e-12)


def test_bootstrap_points_are_the_pooled_estimates(report, panel_inputs):
    panel, summary = _panel(report, panel_inputs)
    boot = report.Bootstrap(panel, n_boot=300)
    checked = 0
    for model in panel.models:
        for s in panel.reported_settings(model):
            for reference, metric in (("cliper", "skill_cliper"),
                                      ("smart_persistence", "skill_sp")):
                est = boot.skill(model, s, reference=reference)
                assert est.point == pytest.approx(summary.stat(model, s, metric),
                                                  rel=1e-12, abs=1e-12)
                lo, hi = est.interval
                assert lo <= est.point <= hi
                checked += 1
            if panel.is_quantile(model, s):
                assert boot.picp(model, s).point == pytest.approx(
                    summary.stat(model, s, "picp_80"), rel=1e-12)
    assert checked >= 8
    # all-ones weights: every resample is the point estimate
    n_days = len(boot.days)
    boot._weights[n_days] = np.ones((5, n_days))
    est = boot.skill("lightgbm", "nwp")
    np.testing.assert_allclose(est.samples, est.point, rtol=1e-12)


def test_bootstrap_is_paired_reproducible_and_zero_for_the_reference(report, panel_inputs):
    panel, _ = _panel(report, panel_inputs)
    boot, again = report.Bootstrap(panel, n_boot=300), report.Bootstrap(panel, n_boot=300)
    np.testing.assert_array_equal(boot.skill("lightgbm", "nwp").samples,
                                  again.skill("lightgbm", "nwp").samples)
    other = report.Bootstrap(panel, n_boot=300, seed=1)
    assert not np.array_equal(boot.skill("lightgbm", "nwp").samples,
                              other.skill("lightgbm", "nwp").samples)
    # a model scored against itself: 0 in every resample, so [0, 0]
    own = boot.skill("cliper", "none", reference="cliper")
    assert own.point == 0.0 and own.interval == (0.0, 0.0)
    same = boot.skill_difference(("lightgbm", "nwp"), ("lightgbm", "nwp"))
    assert same.point == 0.0 and same.interval == (0.0, 0.0)
    # paired: the difference is taken resample by resample on the same days
    a, b = boot.skill("lightgbm", "nwp"), boot.skill("lightgbm", "none")
    np.testing.assert_array_equal(boot.skill_difference(("lightgbm", "nwp"),
                                                        ("lightgbm", "none")).samples,
                                  a.samples - b.samples)


def test_bootstrap_samples_match_an_explicit_shared_resample(report, panel_inputs):
    """Rebuilds the resampled skills by hand with one W shared by every station and
    both sides of a difference: catches unpaired resamples, weights that differ by
    station, and an interval at the wrong level (none of which the checks above see)."""
    panel, _ = _panel(report, panel_inputs)
    n_boot = 200
    boot = report.Bootstrap(panel, n_boot=n_boot)
    n_days = len(boot.days)
    w = report.day_weights(n_days, report.BLOCK_DAYS, n_boot, report.BOOTSTRAP_SEED).astype(float)

    def by_hand(model: str, setting: str) -> np.ndarray:
        per_station = []
        for site in panel.sites:
            frame = panel.runs[(model, setting, site)]
            mask = report.scored_mask(frame)
            y = frame["y_true"].to_numpy(float)[mask]
            point = frame["y_point"].to_numpy(float)[mask]
            ref = (report.reference_forecast(panel, "cliper", site, setting)
                   .reindex(frame.index).to_numpy(float)[mask])
            ok = np.isfinite(ref)
            days = frame.index[mask].tz_convert(report.HKUST_TZ).tz_localize(None).normalize()
            pos = boot.days.get_indexer(days)[ok]
            sse = np.bincount(pos, weights=((point - y) ** 2)[ok], minlength=n_days)
            sse_ref = np.bincount(pos, weights=((ref - y) ** 2)[ok], minlength=n_days)
            per_station.append(1.0 - np.sqrt((w @ sse) / (w @ sse_ref)))
        return np.mean(per_station, axis=0)

    nwp, none = by_hand("lightgbm", "nwp"), by_hand("lightgbm", "none")
    np.testing.assert_allclose(boot.skill("lightgbm", "nwp", reference="cliper").samples,
                               nwp, rtol=1e-10)
    diff = boot.skill_difference(("lightgbm", "nwp"), ("lightgbm", "none"), reference="cliper")
    np.testing.assert_allclose(diff.samples, nwp - none, rtol=1e-10, atol=1e-12)
    assert report.INTERVAL == (2.5, 97.5)
    assert diff.interval == pytest.approx(tuple(np.percentile(nwp - none, [2.5, 97.5])))


def test_compute_reports_the_maximum_memory(report, panel_inputs):
    panel, _ = _panel(report, panel_inputs)
    panel.timing = pd.DataFrame({
        "model": "lightgbm", "weather_mode": "nwp", "site": ["Alpha", "Alpha", "Beta", "Beta"],
        "fold": [0, 1, 0, 1], "fit_s": [10.0, 20.0, 30.0, 40.0], "predict_s": 1.0,
        "fit_predict_s": [11.0, 21.0, 31.0, 41.0], "peak_rss_mb": [300.0, 2048.0, 500.0, 400.0]})
    medians = report.compute_medians(panel)
    row = medians.loc[("lightgbm", "nwp")]
    assert row["max_peak_rss_mb"] == 2048.0 and row["peak_rss_mb"] == 450.0
    assert row["window_s"] == 26.0 and row["n"] == 4
    table = report.table_compute(panel, medians)
    shown = table.display.iloc[table.numeric.query(
        "model == 'lightgbm' and setting == 'nwp'").index.item()]
    assert shown["Max memory (GB)"] == "2.0" and shown["Fit (s)"] == "25.0"
    assert shown["Time per window (s)"] == "26.0"           # the median of fit + predict
    assert "GPU (Metal) memory" in table.caption and "maximum" in table.caption
    assert "time per window" in table.caption
    # every model is a compute row (DEFINITIVE_RUN.md §7), the references too
    assert {"Smart persistence †", "CLIPER †"} <= set(table.display["Model"])


def test_quantile_score_is_half_the_store_crps(report, panel_inputs, tmp_path):
    from pvfc.evaluation import pinball_loss

    _main(report, panel_inputs, tmp_path)
    t3 = pd.read_csv(tmp_path / "tables" / "probabilistic.csv")
    qs = t3.query("model == 'lightgbm' and setting == 'nwp'")["qs_c"].item()
    by_hand = []
    for site, frame in _frames(panel_inputs, "lightgbm", "nwp").items():
        mask = frame["is_day"].to_numpy() & np.isfinite(frame["y_true"].to_numpy())
        q = {tau: frame[f"q{tau}"].to_numpy()[mask] for tau in DEFAULT_QUANTILES}
        by_hand.append(pinball_loss(frame["y_true"].to_numpy()[mask], q) / CAPACITY[site])
    assert qs == pytest.approx(np.mean(by_hand), rel=1e-12)
    long = pd.read_csv(tmp_path / "tables" / "metrics_long.csv")
    crps = long.query("model == 'lightgbm' and weather_mode == 'nwp' and hours == 'daytime' "
                      "and metric == 'ncrps'")["value"].mean()
    assert qs == pytest.approx(crps / 2, rel=1e-12)
    # deterministic models have no quantile score row
    assert not {"cliper", "smart_persistence"} & set(t3["model"])


def test_headline_reference_switches_every_main_table(report, panel_inputs, tmp_path,
                                                      monkeypatch):
    monkeypatch.setattr(report, "HEADLINE_REF", "smart_persistence")
    _main(report, panel_inputs, tmp_path)
    tables = tmp_path / "tables"
    t1 = pd.read_csv(tables / "master_skill.csv").set_index("model")
    full = pd.read_csv(tables / "appendix" / "master_skill_full.csv").set_index("model")
    assert t1.loc["smart_persistence", "skill_none"] == 0.0
    assert t1.loc["cliper", "skill_none"] == pytest.approx(
        full.loc["cliper", "skill_sp_none_mean"], rel=1e-12)
    md = (tables / "master_skill.md").read_text(encoding="utf-8")
    sp_row = next(line for line in md.splitlines() if line.startswith("| Smart persistence"))
    assert sp_row.split("|")[3].strip() == "0"
    assert "skill over Smart persistence" in md and "with skill over CLIPER" in md
    t5 = pd.read_csv(tables / "significance.csv").set_index("id")
    assert t5.loc["P11", "skill_diff"] == pytest.approx(
        full.loc["cliper", "skill_sp_none_mean"], rel=1e-12)
    by_station = (tables / "appendix" / "skill_by_station.md").read_text(encoding="utf-8")
    assert "Skill over Smart persistence" in by_station and "| Smart persistence" not in \
        by_station and "| CLIPER" in by_station


# --------------------------------------------------------------------------- #
# The definitive run: roster, tested family, intervals-only block
# --------------------------------------------------------------------------- #
FOUNDATION = {"chronos2": "Chronos-B", "chronos2_small": "Chronos-S", "timesfm3": "TimesFM-3",
              "tirex2": "TiRex-2", "chronos2_ft": "Chronos-S-FT"}
REFERENCES = ("persistence", "seasonal_naive", "smart_persistence", "climatology", "cliper")
QUANTILES = [f"q{tau}" for tau in DEFAULT_QUANTILES]
TESTED = [f"P{i}" for i in range(1, 12)]


def test_roster_labels_families_and_main_models(report):
    assert report.MAIN_MODELS == (
        "persistence", "seasonal_naive", "smart_persistence", "climatology", "cliper",
        "sarima", "lightgbm", "lstm", "chronos2", "chronos2_small", "timesfm3", "tirex2",
        "chronos2_ft")
    assert list(report.MAIN_MODELS) == report.model_order(report.MAIN_MODELS)
    assert {m: report.display_name(m) for m in FOUNDATION} == FOUNDATION
    assert {report.family(m) for m in REFERENCES} == {report.REFERENCE}
    assert {report.family(m) for m in ("sarima", "lightgbm", "lstm")} == {report.TRAINED}
    assert {m for m in FOUNDATION if report.family(m) == report.FOUNDATION_ZS} == {
        "chronos2", "chronos2_small", "timesfm3", "tirex2"}
    assert report.family("chronos2_ft") == report.FOUNDATION_FT
    names = [display for _, display, _ in report.MODELS]
    assert len(names) == len(set(names))
    # the configurations of the earlier runs are not main rows, and nothing is post hoc
    assert not {"timesfm", "tirex", "timesfm_ctx16k", "tirex_ctx2048", "chronos2_selected",
                "chronos2_base", "chronos2_base_ctx2048", "chronos2_base_ctx8192"} \
        & set(report.MAIN_MODELS)
    assert not report.RUN_ONCE & set(report.MAIN_MODELS)
    for name in ("POST_HOC", "POST_HOC_MODELS", "POST_HOC_MARK"):
        assert not hasattr(report, name)
    assert report.CHRONOS2_SETUP["chronos2"][:2] == ("base", 8192)
    assert report.CHRONOS2_SETUP["chronos2_small"][:2] == ("small", 336)
    assert report.CHRONOS2_SETUP["chronos2_ft"][:2] == ("small", 336)


def test_comparisons_are_the_declared_table(report):
    # DEFINITIVE_RUN.md §7: the tested family, A - B as listed
    assert report.COMPARISONS == (
        ("P1", ("lightgbm", "nwp"), ("chronos2", "nwp")),
        ("P2", ("lightgbm", "nwp"), ("chronos2_ft", "nwp")),
        ("P3", ("chronos2_ft", "nwp"), ("sarima", "nwp")),
        ("P4", ("chronos2_ft", "nwp"), ("chronos2_small", "nwp")),
        ("P5", ("chronos2_ft", "none"), ("chronos2_small", "none")),
        ("P6", ("sarima", "none"), ("chronos2", "none")),
        ("P7", ("lightgbm", "nwp"), ("timesfm3", "nwp")),
        ("P8", ("lightgbm", "nwp"), ("tirex2", "nwp")),
        ("P9", ("lightgbm", "nwp"), ("lightgbm", "none")),
        ("P10", ("chronos2", "nwp"), ("chronos2", "none")),
        ("P11", ("cliper", "nwp"), ("smart_persistence", "nwp")),
    )
    # and the paired differences with intervals only
    assert report.INTERVALS_ONLY == (
        ("W1", ("timesfm3", "nwp"), ("timesfm3", "none")),
        ("W2", ("tirex2", "nwp"), ("tirex2", "none")),
        ("W3", ("timesfm3", "nwp"), ("chronos2", "nwp")),
        ("W4", ("tirex2", "nwp"), ("chronos2", "nwp")),
    )
    ids = [c[0] for c in (*report.COMPARISONS, *report.INTERVALS_ONLY)]
    assert len(ids) == len(set(ids))
    assert not any(i[0] in "EV" for i in ids)     # never confused with the old E/V rows


def test_the_default_stores_are_the_main_run_only(report, monkeypatch, tmp_path):
    seen = {}

    def hashes(paths):
        seen["results"] = list(paths)
        return set()

    def stop(dirs, *args, **kwargs):
        seen["predictions"] = list(dirs)
        raise SystemExit("stopped before reading")

    monkeypatch.setattr(report, "stored_hashes", hashes)
    monkeypatch.setattr(report, "load_predictions", stop)
    with pytest.raises(SystemExit, match="stopped before reading"):
        report.main(["--tables-dir", str(tmp_path / "t"), "--figures-dir", str(tmp_path / "f")])
    assert seen == {"results": [report.ROOT / "results" / "results.parquet"],
                    "predictions": [report.ROOT / "results" / "predictions"]}


#: model -> (source model in the synthetic panel, factor on its forecasts, quantiles kept)
ROSTER_SOURCES = {
    "persistence": ("smart_persistence", 1.05, False),
    "seasonal_naive": ("smart_persistence", 1.1, False),
    "climatology": ("cliper", 0.95, False),
    "sarima": ("lightgbm", 1.12, True),
    "lstm": ("lightgbm", 1.06, False),
    "chronos2": ("lightgbm", 1.03, True),
    "chronos2_small": ("lightgbm", 1.08, True),
    "timesfm3": ("lightgbm", 1.02, True),
    "tirex2": ("lightgbm", 1.05, True),
    "chronos2_ft": ("lightgbm", 0.99, True),
}


@pytest.fixture(scope="module")
def roster_inputs(report, panel_inputs, tmp_path_factory):
    """The synthetic panel completed to the 13 models of the definitive run by scaling
    its runs: the trained and foundation-model rows from LightGBM's (so they read the
    weather; the no-covariate TiRex-2 run is replaced), the other references from smart
    persistence and CLIPER (so they do not), each with its timing rows. Chronos-S's
    No-weather median differs from y_hat on a few hours."""
    root = tmp_path_factory.mktemp("roster")
    pred = root / "predictions"
    pred.mkdir()
    for path in panel_inputs["predictions"].glob("*.parquet"):
        if "__tirex2__" not in path.name:
            shutil.copy(path, pred)
    results = pd.read_parquet(panel_inputs["results"]).query("model != 'tirex2'")
    rows = []
    for model, (source, factor, quantiles) in ROSTER_SOURCES.items():
        for path in sorted(panel_inputs["predictions"].glob(f"*__{source}__*.parquet")):
            frame = pd.read_parquet(path)
            site, setting = frame["site"].iloc[0], frame["weather_mode"].iloc[0]
            columns = ["y_hat", *(c for c in QUANTILES if c in frame.columns)]
            frame[columns] = frame[columns] * factor
            if not quantiles:
                frame = frame.drop(columns=[c for c in QUANTILES if c in frame.columns])
            if model == "chronos2_small" and setting == "none":
                frame.iloc[12:15, frame.columns.get_loc("q0.5")] += 0.5
            chash = f"{model}{setting}{site}"
            frame["model"], frame["config_hash"] = model, chash
            frame.to_parquet(pred / f"{site}__{setting}__{model}__{chash}.parquet")
            src = results[(results["model"] == source) & (results["weather_mode"] == setting)
                          & (results["site"] == site)]
            rows.append(src.assign(model=model, config_hash=chash, fit_s=src["fit_s"] * factor))
    pd.concat([results, *rows], ignore_index=True).to_parquet(root / "results.parquet",
                                                              index=False)
    return dict(panel_inputs, root=root, predictions=pred, results=root / "results.parquet")


@pytest.fixture(scope="module")
def roster_out(report, roster_inputs, tmp_path_factory):
    out = tmp_path_factory.mktemp("roster_out")
    written = _main(report, roster_inputs, out)
    return {"tables": out / "tables", "figures": out / "figures", "written": written}


def test_the_definitive_roster_fills_every_table_and_figure(report, roster_out):
    tables, figures = roster_out["tables"], roster_out["figures"]
    for folder, stems in ((tables, TABLES), (tables / "appendix", APPENDIX_TABLES)):
        for stem in stems:
            assert (folder / f"{stem}.csv").exists() and (folder / f"{stem}.md").exists()
    for stem in FIGURES:
        assert figures / f"{stem}.pdf" in roster_out["written"]
    t1 = pd.read_csv(tables / "master_skill.csv").set_index("model")
    assert list(t1.index) == list(report.MAIN_MODELS)
    # the references take no weather; every trained and foundation-model row reads it
    assert set(t1.index[t1["no_covariates"]]) == set(REFERENCES)
    readers = [m for m in report.MAIN_MODELS if m not in REFERENCES]
    assert t1.loc[readers, ["skill_nwp", "skill_none"]].notna().all().all()
    md = (tables / "master_skill.md").read_text(encoding="utf-8")
    for label in FOUNDATION.values():
        row = next(line for line in md.splitlines() if line.startswith(f"| {label} "))
        assert "†" not in row and row.count("[") == 2
    full = pd.read_csv(tables / "appendix" / "master_skill_full.csv")
    assert set(full["model"]) == set(report.MAIN_MODELS)
    t3 = pd.read_csv(tables / "probabilistic.csv")
    for model in FOUNDATION:
        assert sorted(t3.loc[t3["model"] == model, "setting"]) == ["none", "nwp"]
    t4 = pd.read_csv(tables / "compute.csv")
    assert set(t4["model"]) == set(report.MAIN_MODELS)
    assert len(t4) == 2 * len(readers) + len(REFERENCES)    # a reference once, No weather
    assert t4[["median_fit_s", "median_window_s", "max_peak_rss_mb"]].notna().all().all()
    assert "| Time per window (s)" in (tables / "compute.md").read_text(encoding="utf-8")
    t2 = pd.read_csv(tables / "chronos2_configs.csv")
    assert list(t2["model"]) == ["chronos2", "chronos2_small", "chronos2_ft"]
    t2_md = (tables / "chronos2_configs.md").read_text(encoding="utf-8")
    row = next(line for line in t2_md.splitlines() if line.startswith("| Chronos-B "))
    assert "base" in row and "8,192" in row
    # no mark, note or row of the earlier post hoc design anywhere
    for path in [*tables.glob("*.md"), *(tables / "appendix").glob("*.md")]:
        text = path.read_text(encoding="utf-8")
        assert "‡" not in text and "post hoc" not in text.lower(), path.name
        assert "Chronos-sel" not in text and "selection-path" not in text, path.name
    for path in [*tables.glob("*.csv"), *(tables / "appendix").glob("*.csv")]:
        assert "‡" not in path.read_text(encoding="utf-8"), path.name


def test_the_intervals_only_block_has_no_tests_and_stays_out_of_holm(report, roster_out):
    from statsmodels.stats.multitest import multipletests

    tables = roster_out["tables"]
    t5 = pd.read_csv(tables / "significance.csv").set_index("id")
    tested, untested = t5[t5["block"] == "tested"], t5[t5["block"] == "intervals only"]
    assert list(tested.index) == TESTED
    assert list(untested.index) == ["W1", "W2", "W3", "W4"]
    assert tested["available"].all() and untested["available"].all()
    for col in ("p_dm", "p_holm", "p_bootstrap", "dm_stat", "n_days"):
        assert untested[col].isna().all()
    assert tested["p_dm"].notna().all()
    np.testing.assert_allclose(tested["p_holm"], multipletests(tested["p_dm"], method="holm")[1])
    assert (untested["skill_diff_lo"] <= untested["skill_diff"]).all()
    assert (untested["skill_diff"] <= untested["skill_diff_hi"]).all()
    t1 = pd.read_csv(tables / "master_skill.csv").set_index("model")
    assert untested.loc["W1", "skill_diff"] == pytest.approx(
        t1.loc["timesfm3", "skill_nwp"] - t1.loc["timesfm3", "skill_none"], rel=1e-10)
    assert untested.loc["W4", "skill_diff"] == pytest.approx(
        t1.loc["tirex2", "skill_nwp"] - t1.loc["chronos2", "skill_nwp"], rel=1e-10)
    assert untested.loc["W2", ["setting_a", "setting_b"]].tolist() == ["nwp", "none"]
    md = (tables / "significance.md").read_text(encoding="utf-8")
    rows = {line.split("|")[1].strip(): [c.strip() for c in line.strip("|").split("|")]
            for line in md.splitlines()[4:]}
    assert rows["W1"][1] == "TimesFM-3: NWP vs No weather"
    assert rows["W3"][1] == "NWP: TimesFM-3 vs Chronos-B"
    assert rows["W3"][3:] == ["—", "no test"] and "[" in rows["W3"][2]
    assert rows["P7"][1] == "NWP: LightGBM vs TimesFM-3" and rows["P7"][3] != "—"
    assert rows["P7"][4] in ("n.s.", "A better", "B better")
    caption = md.splitlines()[0]
    assert "Holm-adjusted over the 11 comparisons" in caption
    assert "W1–W4 are outside the tested family" in caption and "intervals only" in caption


def test_p_values_are_computed_for_the_tested_family_only(report, roster_out):
    tables = roster_out["tables"]
    t5 = pd.read_csv(tables / "significance.csv").set_index("id")
    for col in ("p_dm", "p_holm", "p_bootstrap"):
        assert list(t5.index[t5[col].notna()]) == TESTED
    detail = pd.read_csv(tables / "appendix" / "significance_detail.csv").set_index("id")
    assert list(detail.index) == TESTED
    sides = {i: (r["model_a"], r["setting_a"], r["model_b"], r["setting_b"])
             for i, r in detail.iterrows()}
    assert sides["P1"] == ("lightgbm", "nwp", "chronos2", "nwp")
    assert sides["P4"] == ("chronos2_ft", "nwp", "chronos2_small", "nwp")
    assert sides["P5"] == ("chronos2_ft", "none", "chronos2_small", "none")
    assert sides["P6"] == ("sarima", "none", "chronos2", "none")
    assert sides["P7"] == ("lightgbm", "nwp", "timesfm3", "nwp")
    assert sides["P8"] == ("lightgbm", "nwp", "tirex2", "nwp")
    assert sides["P10"] == ("chronos2", "nwp", "chronos2", "none")
    # CLIPER and smart persistence take no weather: tested where T1 reports them
    assert sides["P11"] == ("cliper", "none", "smart_persistence", "none")


def test_figures_hold_the_full_roster(report, roster_inputs, tmp_path, monkeypatch):
    panel, summary = _panel(report, roster_inputs)
    captured: dict[str, plt.Figure] = {}
    monkeypatch.setattr(report, "_save", lambda fig, _, stem: captured.update({stem: fig}) or [])
    with plt.rc_context(report.STYLE):
        report.fig_weather_gain(panel, summary, tmp_path)
        gain = [t.get_text() for t in captured["weather_gain"].axes[0].get_yticklabels()]
        # every model that reads the weather: the five foundation-model rows and the trained
        assert set(gain) == {*FOUNDATION.values(), "SARIMA", "LightGBM", "LSTM"}
        report.fig_reliability(panel, summary, tmp_path)
        fig = captured["reliability"]
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert {ax.get_title(loc="left") for ax in visible} == {
            *FOUNDATION.values(), "SARIMA", "LightGBM"}      # the seven quantile models
        assert visible[0].get_subplotspec().get_gridspec().get_geometry() == (3, 3)
        # more than nine panels: five columns
        runs = {k: f if report.has_quantiles(f) else f.assign(**{c: f["y_hat"] for c in QUANTILES})
                for k, f in panel.runs.items()}
        wide = report.Panel(runs=runs, capacity=panel.capacity, no_covariates=panel.no_covariates)
        report.fig_reliability(wide, report.Summary(report.compute_station_metrics(wide)),
                               tmp_path)
        fig = captured["reliability"]
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) == len(report.MAIN_MODELS)
        assert visible[0].get_subplotspec().get_gridspec().get_geometry() == (3, 5)
    plt.close("all")


def test_chronos2_table_and_p2_robustness_follow_the_runs_given(report, roster_inputs,
                                                                 roster_out, tmp_path, caplog):
    # the definitive run has no extra fine-tuning seeds: one all-windows row
    p2 = pd.read_csv(roster_out["tables"] / "appendix" / "p2_robustness.csv")
    assert p2["variant"].iloc[0] == "All windows"
    assert p2["variant"].str.startswith("Without").sum() == 2
    assert list(p2["n_days"]) == [20, 10, 10]
    caption = (roster_out["tables"] / "appendix" / "p2_robustness.md").read_text(
        encoding="utf-8").splitlines()[0]
    assert "on all test windows, and leaving out one test window" in caption
    assert "seed" not in caption
    point_rule = (roster_out["tables"] / "appendix" / "point_rule.md").read_text(
        encoding="utf-8")
    assert "y_hat is not the 0.5 quantile: Chronos-S, No weather" in point_rule

    # Chronos-S missing at one station, and a store with a second fine-tuning seed
    inputs = dict(roster_inputs, predictions=tmp_path / "predictions")
    shutil.copytree(roster_inputs["predictions"], inputs["predictions"])
    next(inputs["predictions"].glob("Beta__nwp__chronos2_small__*.parquet")).unlink()
    for path in sorted(roster_inputs["predictions"].glob("*__nwp__chronos2_ft__*.parquet")):
        frame = pd.read_parquet(path)
        site, chash = frame["site"].iloc[0], f"s1{frame['site'].iloc[0]}"
        frame[["y_hat", *QUANTILES]] = frame[["y_hat", *QUANTILES]] * 1.01
        frame.assign(model="chronos2_ft_s1", config_hash=chash).to_parquet(
            inputs["predictions"] / f"{site}__nwp__chronos2_ft_s1__{chash}.parquet")
    with caplog.at_level(logging.WARNING, logger="pvfc.report"):
        _main(report, inputs, tmp_path)
    assert any("chronos2_small left out" in r.getMessage() for r in caplog.records)
    tables = tmp_path / "tables"
    t2 = pd.read_csv(tables / "chronos2_configs.csv")
    assert list(t2["model"]) == ["chronos2", "chronos2_ft"]
    md = (tables / "chronos2_configs.md").read_text(encoding="utf-8")
    assert "Not shown (a run is missing at some station): Chronos-S." in md.splitlines()[0]
    assert "| Configuration" in md and "Time per window (s)" in md
    p2 = pd.read_csv(tables / "appendix" / "p2_robustness.csv")
    assert list(p2["variant"][:2]) == ["All windows (seed 0)", "All windows, seed 1"]
    assert list(p2["n_days"]) == [20, 20, 10, 10]
    p2_md = (tables / "appendix" / "p2_robustness.md").read_text(encoding="utf-8")
    assert "with each fine-tuning seed, and with seed 0 leaving out" in p2_md
    assert "‡" not in p2_md


def test_identical_settings_void_the_weather_comparison(report, panel_inputs, monkeypatch,
                                                        caplog):
    """This TiRex-2's NWP and No-weather files are identical (what a wrapper that drops
    the covariates would leave): its weather comparisons are void, with a warning, never
    a self-comparison of 0 [0, 0] or a degenerate test counted in the Holm family."""
    panel, _ = _panel(report, panel_inputs)
    assert "tirex2" in panel.no_covariates
    monkeypatch.setattr(report, "COMPARISONS", (
        ("P8", ("lightgbm", "nwp"), ("tirex2", "nwp")),
        ("P9", ("tirex2", "nwp"), ("tirex2", "none")),
        ("P11", ("cliper", "nwp"), ("smart_persistence", "nwp"))))
    boot = report.Bootstrap(panel)
    with caplog.at_level(logging.WARNING, logger="pvfc.report"):
        sig = report.add_skill_differences(report.compute_significance(panel), boot)
        untested = report.compute_intervals_only(panel, boot)
    for pid in ("P9", "W2"):
        assert any(r.levelno == logging.WARNING and r.getMessage().startswith(
            f"{pid} not available: NWP and No-weather forecasts of tirex2 identical")
            for r in caplog.records), pid
    sig, untested = sig.set_index("id"), untested.set_index("id")
    for frame, pid in ((sig, "P9"), (untested, "W2")):
        row = frame.loc[pid]
        assert not row["available"] and np.isnan(row["skill_diff"])
        assert row[["model_a", "setting_a", "model_b", "setting_b"]].tolist() == [
            "tirex2", "nwp", "tirex2", "none"]                  # the declared sides
    assert sig.loc["P9", "verdict"] == "not available" and np.isnan(sig.loc["P9", "p_holm"])
    assert list(sig.index[sig["p_holm"].notna()]) == ["P8", "P11"]
    # a model with weather compared across settings stays a real comparison
    assert sig.loc["P8", "available"] and sig.loc["P8", "setting_b"] == "none"

    table = report.table_significance(panel, sig.reset_index(), untested.reset_index())
    shown = table.display.set_index("ID")
    for pid in ("P9", "W2"):
        assert shown.loc[pid, "Comparison (A vs B)"] == "TiRex-2: NWP vs No weather"
        assert shown.loc[pid, "Verdict"] == "not available"
        assert shown.loc[pid, "Skill difference (A − B)"] == "—"
    assert ("P9, W2: the forecasts of TiRex-2 are identical in both weather settings, so "
            "the weather comparison is void (not available).") in table.caption
    assert "Holm-adjusted over the 2 comparisons" in table.caption
    detail = report.table_significance_detail(panel, sig.reset_index())
    assert "P9: the forecasts of TiRex-2" in detail.caption and "void (n/a)" in detail.caption
    assert "no run at some station" not in detail.caption


def test_a_side_missing_at_one_station_is_never_tested(report, roster_inputs, caplog):
    """TimesFM-3 with NWP missing at Beta: P7 is not available (no test on the other
    stations alone, as for W1 and W3), with a warning; the other pairs still run."""
    panel, _ = _panel(report, roster_inputs)
    del panel.runs[("timesfm3", "nwp", "Beta")]
    boot = report.Bootstrap(panel)
    with caplog.at_level(logging.WARNING, logger="pvfc.report"):
        sig = report.compute_significance(panel).set_index("id")
        untested = report.compute_intervals_only(panel, boot).set_index("id")
    assert not sig.loc["P7", "available"] and sig.loc["P7", "verdict"] == "not available"
    assert np.isnan(sig.loc["P7", "p_holm"])
    assert list(sig.index[sig["available"]]) == [p for p in TESTED if p != "P7"]
    assert sig["p_holm"].notna().sum() == 10
    assert list(untested.index[~untested["available"]]) == ["W1", "W3"]
    assert any(r.levelno == logging.WARNING and r.getMessage() == (
        "P7 not available: no run at every station for timesfm3/nwp") for r in caplog.records)
    detail = report.table_significance_detail(panel, sig.reset_index())
    assert "n/a: not available (a model has no run at some station)." in detail.caption


def test_one_test_window_builds_every_table_and_removes_old_tables(report, roster_inputs,
                                                                   tmp_path):
    """A store with one test window (a --max-folds 1 or pre-flight run) builds the whole
    report: P2's robustness table has no leave-one-window-out rows (nothing would be
    left to resample). The selection table of the earlier runs is removed."""
    pred = tmp_path / "predictions"
    pred.mkdir()
    for path in roster_inputs["predictions"].glob("*.parquet"):
        frame = pd.read_parquet(path)
        frame[frame["fold"] == 0].to_parquet(pred / path.name)
    results = tmp_path / "results.parquet"
    pd.read_parquet(roster_inputs["results"]).query("fold == 0").to_parquet(results)
    out = tmp_path / "out"
    stale = [out / "tables" / "appendix" / f"selection_path.{ext}" for ext in ("csv", "md")]
    for path in stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(": Chronos-sel ‡ {#tbl:selection-path}\n", encoding="utf-8")
    _main(report, dict(roster_inputs, predictions=pred, results=results), out)
    tables = out / "tables"
    for folder, stems in ((tables, TABLES), (tables / "appendix", APPENDIX_TABLES)):
        for stem in stems:
            assert (folder / f"{stem}.md").exists(), stem
    assert not any(path.exists() for path in stale)
    p2 = pd.read_csv(tables / "appendix" / "p2_robustness.csv")
    assert list(p2["variant"]) == ["All windows"] and list(p2["n_days"]) == [10]
    caption = (tables / "appendix" / "p2_robustness.md").read_text(
        encoding="utf-8").splitlines()[0]
    assert "on all test windows." in caption and "leaving out" not in caption
