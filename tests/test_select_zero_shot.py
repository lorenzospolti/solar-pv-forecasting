"""Tests for experiments/select_zero_shot.py, the real-time selection of the zero-shot
Chronos-2 configuration (DECISIONS.md, 13 Oct 2026): the candidate order, the configs of
the inner-window and grid runs, the pool of each fold, the tie-break, that a fold's
choice never reads its own test window or anything later, the assembled row, and the
optimism estimates. Everything runs on a small synthetic panel."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
TZ = "Asia/Hong_Kong"
CAPACITY = {"Alpha": 10.0, "Beta": 20.0}
#: the commit that imported the earlier checkout: its configs are those of the test runs
#: the selection read (configs/experiment.yaml now holds the definitive roster, whose
#: "chronos2" is Chronos-2-base at 8,192 h; DEFINITIVE_RUN.md)
IMPORT_COMMIT = "25c5c3b"


@pytest.fixture(scope="module")
def sel():
    name = "pvfc_select_zero_shot_script"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "experiments" / "select_zero_shot.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _models(path: Path) -> dict[str, dict]:
    return {m["name"]: m.get("params", {}) for m in yaml.safe_load(path.read_text())["models"]}


def _at_import(name: str) -> dict:
    """configs/<name> as the earlier checkout had it (the import commit)."""
    try:
        text = subprocess.run(["git", "show", f"{IMPORT_COMMIT}:configs/{name}"], cwd=ROOT,
                              check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git show {IMPORT_COMMIT}:configs/{name} is not available")
    return yaml.safe_load(text)


# --------------------------------------------------------------------------- #
# Candidates and configs
# --------------------------------------------------------------------------- #
def test_candidates_are_the_full_grid_in_cost_order(sel):
    rank = {"small": 0, "base": 1}
    keys = [(rank[ckpt], ctx) for _, ckpt, ctx in sel.CANDIDATES]
    assert keys == sorted(keys) and len(set(keys)) == 8
    assert {(c, x) for _, c, x in sel.CANDIDATES} == {
        (c, x) for c in ("small", "base") for x in (336, 1024, 2048, 8192)}
    assert set(sel.ORIGINAL_SIX) < set(sel.CANDIDATE_NAMES)
    # report.py reports the definitive run (DEFINITIVE_RUN.md): no selected row, and its
    # "chronos2" is Chronos-2-base at 8,192 h, not this candidate
    assert sel.SELECTED not in sel.R.MAIN_MODELS


def test_inner_window_runs_copy_the_test_runs_exactly(sel):
    """Every model of configs/validation.yaml has the params of its test run, and the
    data and evaluation blocks are the test ones; only the windows move. The test runs
    are those of the earlier checkout (panel v2 and the post hoc stores), so their configs
    are read at the import commit."""
    test = {}
    for name in ("experiment.yaml", "extras.yaml", "covariates.yaml", "grid.yaml"):
        test.update({m["name"]: m.get("params", {}) for m in _at_import(name)["models"]})
    val_cfg = yaml.safe_load((CONFIGS / "validation.yaml").read_text())
    val = _models(CONFIGS / "validation.yaml")
    assert set(val) == {sel.REFERENCE, *sel.CANDIDATE_NAMES}
    for name, params in val.items():
        assert params == test[name], name
    exp = _at_import("experiment.yaml")
    assert val_cfg["data"] == exp["data"] and val_cfg["evaluation"] == exp["evaluation"]
    assert val_cfg["seed"] == exp["seed"]
    v, t = val_cfg["splits"], exp["splits"]
    assert (v["test_days"], v["embargo_days"]) == (sel.INNER_DAYS, sel.EMBARGO_DAYS)
    assert t["test_days"] == sel.TEST_DAYS
    for k in ("step_days", "max_folds", "day_tz", "embargo_days"):
        assert v[k] == t[k], k
    # inner window k ends the day before test window k's embargo day
    end = pd.Timestamp(v["first_test_day"]) + pd.Timedelta(days=v["test_days"] - 1)
    assert end + pd.Timedelta(days=v["embargo_days"] + 1) == pd.Timestamp(t["first_test_day"])


def test_a_root_whose_test_configs_differ_is_refused(sel, tmp_path):
    """This checkout's configs/experiment.yaml is the definitive roster ("chronos2" is
    Chronos-2-base at 8,192 h), so its stores are not the candidates' test runs; the
    earlier checkout's configs pass."""
    with pytest.raises(SystemExit, match=r"test runs of chronos2 do not have the params"):
        sel.check_test_configs(ROOT)
    (tmp_path / "configs").mkdir()
    for name in (*sel.TEST_CONFIGS, "validation.yaml"):
        (tmp_path / "configs" / name).write_text(yaml.safe_dump(_at_import(name)))
    sel.check_test_configs(tmp_path)


def test_labels_are_those_of_the_earlier_runs(sel):
    """report.py now labels the definitive roster ("chronos2" is Chronos-B there); the
    selection keeps the candidates' own names."""
    assert sel.display_name("chronos2") == "Chronos-S"
    assert sel.display_name("chronos2_base") == "Chronos-B"
    assert sel.display_name("chronos2_base_ctx8192") == "Chronos-B-max"
    assert sel.display_name(sel.SELECTED) == "Chronos-sel"
    assert set(sel.CANDIDATE_NAMES) < set(sel.LABELS)
    assert len(set(sel.LABELS.values())) == len(sel.LABELS)
    path = pd.DataFrame([{"setting": "nwp", "fold": 0, "pool_days": 29, "pick": "chronos2",
                          "best_on_pool": "chronos2_base_ctx2048", "pick_pool_skill": 0.2,
                          "best_pool_skill": 0.21}])
    shown = sel.table_selection_path(path, {0: "Jul 2022"}).display.iloc[0]
    assert (shown["Selected"], shown["Best on pool"]) == ("Chronos-S", "Chronos-B-long")


def test_grid_runs_complete_the_grid_with_their_neighbours_params():
    grid = _models(CONFIGS / "grid.yaml")
    assert set(grid) == {"chronos2_ctx8192", "chronos2_base_ctx1024"}
    base_max = _models(CONFIGS / "covariates.yaml")["chronos2_base_ctx8192"]
    small_mid = _models(CONFIGS / "extras.yaml")["chronos2_ctx1024"]
    assert grid["chronos2_ctx8192"] == {**base_max, "hub_model_name": "autogluon/chronos-2-small"}
    assert grid["chronos2_base_ctx1024"] == {**small_mid, "hub_model_name": "amazon/chronos-2"}
    exp = yaml.safe_load((CONFIGS / "experiment.yaml").read_text())
    cfg = yaml.safe_load((CONFIGS / "grid.yaml").read_text())
    assert cfg["data"] == exp["data"] and cfg["splits"] == exp["splits"]


def test_validation_splitter_puts_inner_windows_between_test_windows():
    from pvfc.splits import ExpandingWindowSplitter

    def splitter(name):
        sp = yaml.safe_load((CONFIGS / name).read_text())["splits"]
        return ExpandingWindowSplitter(**sp)

    # SQ567, the latest-starting station, from local midnight on 1 Jul 2021
    index = pd.date_range("2021-06-30 16:00", "2023-12-31 15:00", freq="h", tz="UTC")
    inner = list(splitter("validation.yaml").split(index))
    tests = list(splitter("experiment.yaml").split(index))
    assert len(inner) == len(tests) == 9
    for v, t in zip(inner, tests):
        vd = v.test_idx.tz_convert(TZ).normalize().unique()
        td = t.test_idx.tz_convert(TZ).normalize().unique()
        assert len(vd) == 29 and len(td) == 30
        assert (td.min() - vd.max()).days == 2                    # one embargo day between
        prev = [x for x in tests if x.index == t.index - 1]
        if prev:                                                  # after the last test day
            assert vd.min() > prev[0].test_idx.tz_convert(TZ).normalize().max()


# --------------------------------------------------------------------------- #
# Pools and the rule
# --------------------------------------------------------------------------- #
def test_pool_is_every_window_before_the_embargo_day(sel):
    for k in range(9):
        pool = sel.pool_windows(k)
        assert max(pool) < sel.test_window(k)
        assert {sel.inner_window(j) for j in range(k + 1)} <= set(pool)
        assert {sel.test_window(j) for j in range(k)} <= set(pool)
        assert len(pool) == 2 * k + 1


def _est(R, point, samples):
    return R.Estimate(float(point), np.asarray(samples, dtype=float))


def test_tie_break_takes_the_cheapest_within_one_paired_standard_error(sel):
    R = sel.R
    rng = np.random.default_rng(0)
    common = rng.normal(0.0, 0.10, 4000)            # weather shared by every candidate
    est = {
        "a": _est(R, 0.20, 0.20 + common + rng.normal(0, 0.005, 4000)),
        "b": _est(R, 0.25, 0.25 + common + rng.normal(0, 0.020, 4000)),
        "c": _est(R, 0.26, 0.26 + common + rng.normal(0, 0.005, 4000)),
    }
    ch = sel.choose(est, ("a", "b", "c"))
    assert ch.best == "c" and ch.variants["argmax"] == "c"
    assert ch.tied == ["b", "c"] and ch.pick == "b"        # 0.01 < sd(c - b) ~ 0.02
    assert "a" not in ch.tied                              # 0.06 > sd(c - a) ~ 0.007
    # the classic rule uses sd(skill of c) ~ 0.10, wide enough to take a
    assert ch.variants["classic_1se"] == "a"
    assert ch.se_paired["c"] == 0.0


def test_exact_tie_goes_to_the_cheaper(sel):
    R = sel.R
    s = np.linspace(0.1, 0.3, 50)
    ch = sel.choose({"a": _est(R, 0.2, s), "b": _est(R, 0.2, s)}, ("a", "b"))
    assert ch.best == "a" and ch.pick == "a"


def _day_profile(n_days: int, rng) -> np.ndarray:
    hours = np.arange(24)
    shape = np.clip(np.sin((hours - 6) / 12 * np.pi), 0, None)
    return (rng.uniform(0.2, 1.0, n_days)[:, None] * shape).ravel()


def _windows(n_folds: int):
    """(fold, inner days, test days) as local dates."""
    start = pd.Timestamp("2022-06-05")
    for k in range(n_folds):
        inner = pd.date_range(start + pd.Timedelta(days=60 * k), periods=29, freq="D")
        test = pd.date_range(inner[-1] + pd.Timedelta(days=2), periods=30, freq="D")
        yield k, inner, test


def _frame(days: pd.DatetimeIndex, fold: int, site: str, setting: str, model: str,
           truth: pd.Series, error_sd: float, rng) -> pd.DataFrame:
    idx = pd.date_range(days[0], periods=24 * len(days), freq="h", tz=TZ).tz_convert("UTC")
    y = truth.reindex(idx).to_numpy()
    local_hour = idx.tz_convert(TZ).hour
    is_day = (local_hour >= 6) & (local_hour <= 18)
    noise = rng.normal(0, error_sd, len(idx)) * CAPACITY[site] * is_day
    y_hat = np.clip(y + noise, 0, None)
    return pd.DataFrame({
        "y_true": y, "y_hat": y_hat, "y_point": y_hat,
        "ref_hat": np.clip(y + rng.normal(0, 0.3, len(idx)) * CAPACITY[site] * is_day, 0, None),
        "is_day": is_day, "fold": fold, "site": site, "model": model,
        "weather_mode": setting, "config_hash": f"{model}{setting}{site}",
    }, index=pd.DatetimeIndex(idx, name="time"))


@pytest.fixture(scope="module")
def synthetic(sel):
    """Inner-window and test runs of CLIPER and the eight candidates at two stations,
    three folds. Each candidate's error level drifts with the fold, so picks change."""
    rng = np.random.default_rng(42)
    n_folds = 3
    val, test = {}, {}
    sd = {c: 0.08 + 0.01 * i for i, c in enumerate(sel.CANDIDATE_NAMES)}
    for site in CAPACITY:
        all_days = pd.date_range("2022-06-01", "2022-12-31", freq="D")
        truth_idx = pd.date_range(all_days[0], periods=24 * len(all_days), freq="h",
                                  tz=TZ).tz_convert("UTC")
        truth = pd.Series(_day_profile(len(all_days), rng) * CAPACITY[site], index=truth_idx)
        for k, inner, tdays in _windows(n_folds):
            for setting in sel.SETTINGS:
                for model in (sel.REFERENCE, *sel.CANDIDATE_NAMES):
                    level = 0.2 if model == sel.REFERENCE else sd[model] * (1 + 0.4 * k * (
                        -1 if sel.CANDIDATE_NAMES.index(model) % 2 else 1))
                    for store, days in ((val, inner), (test, tdays)):
                        f = _frame(days, k, site, setting, model, truth, level, rng)
                        key = (model, setting, site)
                        store[key] = pd.concat([store[key], f]) if key in store else f
    return val, test


def _panel(sel, runs):
    return sel.R.Panel(runs=sel.R.check_runs(runs), capacity=dict(CAPACITY))


def test_combine_numbers_windows_in_date_order(sel, synthetic):
    val, test = synthetic
    runs = sel.combine(val, test, (sel.REFERENCE, *sel.CANDIDATE_NAMES))
    frame = runs[("chronos2", "nwp", "Alpha")]
    first = frame.index.to_series().groupby(frame["fold"].to_numpy()).min()
    assert list(first.index) == list(range(6)) and first.is_monotonic_increasing


def test_combine_rejects_a_misplaced_inner_window(sel, synthetic):
    val, test = synthetic
    bad = dict(val)
    key = ("chronos2", "nwp", "Alpha")
    bad[key] = val[key].iloc[24:]                        # fold 0 loses its first day
    with pytest.raises(SystemExit, match="28 inner"):
        sel.combine(bad, test, (sel.REFERENCE, *sel.CANDIDATE_NAMES))


def test_a_folds_choice_never_reads_its_test_window_or_later(sel, synthetic):
    """Fold k's choice is the same whether or not the panel holds anything after its pool:
    its test window and every later window are invisible to it."""
    val, test = synthetic
    full = _panel(sel, sel.combine(val, test, (sel.REFERENCE, *sel.CANDIDATE_NAMES)))
    boot = sel.R.Bootstrap(full)
    choices = {s: sel.select(boot, s, 3) for s in sel.SETTINGS}
    picks = {ch.pick for s in sel.SETTINGS for ch in choices[s]}
    assert len(picks) > 1                                 # the synthetic picks do change
    for k in range(3):
        cut = {key: f[f["fold"].to_numpy() <= 2 * k] for key, f in full.runs.items()}
        # scramble the future too: it must not matter
        later = {key: f[f["fold"].to_numpy() > 2 * k].assign(
            y_point=lambda d: d["y_point"][::-1].to_numpy()) for key, f in full.runs.items()}
        for runs in (cut, {key: pd.concat([cut[key], later[key]]) for key in cut}):
            b = sel.R.Bootstrap(_panel(sel, runs))
            for s in sel.SETTINGS:
                ch = sel.select(b, s, k + 1)[k]
                ref = choices[s][k]
                assert (ch.pick, ch.best, ch.tied) == (ref.pick, ref.best, ref.tied)
                assert ch.skill == pytest.approx(ref.skill, rel=1e-12)
                assert ch.n_days == ref.n_days == 29 * (k + 1) + 30 * k


def test_assembled_row_takes_each_test_window_from_its_pick(sel, synthetic):
    _, test = synthetic
    picks = {0: "chronos2", 1: "chronos2_base_ctx8192", 2: "chronos2_ctx1024"}
    rows = sel.assemble(test, picks, "nwp", sel.SELECTED)
    assert set(rows) == {(sel.SELECTED, "nwp", s) for s in CAPACITY}
    for (_, _, site), row in rows.items():
        assert len(row) == len(test[("chronos2", "nwp", site)])
        assert row.index.is_monotonic_increasing and row.index.name == "time"
        for k, model in picks.items():
            part = row[row["fold"].to_numpy() == k]
            src = test[(model, "nwp", site)]
            src = src[src["fold"].to_numpy() == k]
            assert (part["source_model"] == model).all()
            np.testing.assert_array_equal(part["y_hat"].to_numpy(), src["y_hat"].to_numpy())
            np.testing.assert_array_equal(part.index, src.index)
        assert (row["model"] == sel.SELECTED).all() and "y_point" not in row.columns
    hashes = {r["config_hash"].iloc[0] for r in rows.values()}
    assert len(hashes) == len(CAPACITY)


def test_optimism_is_zero_when_one_candidate_wins_every_window(sel, synthetic):
    _, test = synthetic
    runs = {k: f for k, f in test.items() if k[0] in (sel.REFERENCE, "chronos2", "chronos2_base")}
    # make chronos2_base strictly better than chronos2 everywhere
    for key in [k for k in runs if k[0] == "chronos2_base"]:
        f = runs[key]
        better = f["y_true"] + 0.1 * (f["y_point"] - f["y_true"])
        runs[key] = f.assign(y_point=better, y_hat=better)
    boot = sel.R.Bootstrap(_panel(sel, runs))
    tt = sel.tt_optimism(boot, "nwp", ("chronos2", "chronos2_base"), [0, 1, 2])
    assert tt["hindsight_best"] == "chronos2_base"
    assert tt["optimism"] == pytest.approx(0.0, abs=1e-12)
    assert tt["window_winners"] == ["chronos2_base"] * 3
    a = sel.analytic_optimism(boot, "nwp", ("chronos2", "chronos2_base"))
    assert a["first"] == "chronos2_base" and a["optimism"] == pytest.approx(
        a["sd_difference"] / np.sqrt(2 * np.pi))


def test_optimism_matches_a_hand_computation(sel, synthetic):
    _, test = synthetic
    models = sel.CANDIDATE_NAMES[:4]
    runs = {k: f for k, f in test.items() if k[0] in (sel.REFERENCE, *models)}
    boot = sel.R.Bootstrap(_panel(sel, runs))
    tt = sel.tt_optimism(boot, "nwp", models, [0, 1, 2])

    def skill(model, fold):
        vals = []
        for site in CAPACITY:
            f = runs[(model, "nwp", site)]
            ref = runs[(sel.REFERENCE, "nwp", site)]
            m = f["is_day"].to_numpy() & (f["fold"].to_numpy() == fold)
            e = f["y_point"].to_numpy()[m] - f["y_true"].to_numpy()[m]
            r = ref["y_point"].to_numpy()[m] - f["y_true"].to_numpy()[m]
            vals.append(1 - np.sqrt(np.mean(e ** 2) / np.mean(r ** 2)))
        return np.mean(vals)

    star = tt["hindsight_best"]
    expected = np.mean([max(skill(c, w) for c in models) - skill(star, w) for w in range(3)])
    assert tt["optimism"] == pytest.approx(expected, rel=1e-9)
    assert tt["optimism"] >= 0


def test_report_refuses_the_inner_window_forecasts(sel, tmp_path):
    folder = tmp_path / "results" / "validation" / "predictions"
    folder.mkdir(parents=True)
    with pytest.raises(SystemExit, match="not test forecasts"):
        sel.R.main(["--predictions-dir", str(folder), "--results", str(tmp_path / "x.parquet"),
                    "--tables-dir", str(tmp_path / "t"), "--figures-dir", str(tmp_path / "f")])


def test_bootstrap_optimism_is_zero_for_a_dominant_candidate(sel, synthetic):
    _, test = synthetic
    runs = {k: f for k, f in test.items() if k[0] in (sel.REFERENCE, "chronos2", "chronos2_base")}
    for key in [k for k in runs if k[0] == "chronos2_base"]:
        f = runs[key]
        better = f["y_true"] + 0.05 * (f["y_point"] - f["y_true"])
        runs[key] = f.assign(y_point=better, y_hat=better)
    boot = sel.R.Bootstrap(_panel(sel, runs))
    b = sel.bootstrap_optimism(boot, "nwp", ("chronos2", "chronos2_base"))
    assert b["win_share"] == {"chronos2": 0.0, "chronos2_base": 1.0}
    assert b["optimism"] == pytest.approx(0.0, abs=1e-12)


def test_bootstrap_optimism_of_two_equal_candidates_is_about_the_analytic_value(sel, synthetic):
    """Two candidates with identical forecasts but for independent noise of equal size:
    the centred-bootstrap optimism is close to sd(D) / sqrt(2 pi) and both win often."""
    _, test = synthetic
    rng = np.random.default_rng(7)
    runs = {k: f for k, f in test.items() if k[0] == sel.REFERENCE}
    for site in CAPACITY:
        base = test[("chronos2", "nwp", site)]
        for model in ("chronos2", "chronos2_base"):
            noise = rng.normal(0, 0.08, len(base)) * CAPACITY[site] * base["is_day"].to_numpy()
            y = np.clip(base["y_true"].to_numpy() + noise, 0, None)
            runs[(model, "nwp", site)] = base.assign(y_point=y, y_hat=y, model=model)
    boot = sel.R.Bootstrap(_panel(sel, runs))
    b = sel.bootstrap_optimism(boot, "nwp", ("chronos2", "chronos2_base"))
    a = sel.analytic_optimism(boot, "nwp", ("chronos2", "chronos2_base"))
    assert min(b["win_share"].values()) > 0.05
    assert 0 < b["optimism"] <= 2.5 * a["optimism"]
