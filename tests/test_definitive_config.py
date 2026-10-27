"""The definitive run's configs (DEFINITIVE_RUN.md sections 2-5): the 13-model roster, the
params copied from panel v2 and chronos2_base_ctx8192, the pinned revisions and maximum
contexts of TimesFM-3 and TiRex-2, the development-day check, the batch plan, the
checkpoint pins, and the guards of the check scripts."""
from __future__ import annotations

import importlib.util
import math
import re
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
#: the commit that imported the earlier checkout (panel v2 configs) byte for byte
IMPORT_COMMIT = "25c5c3b"
ROSTER = ["persistence", "smart_persistence", "seasonal_naive", "climatology", "cliper",
          "sarima", "lightgbm", "lstm", "chronos2", "chronos2_small", "timesfm3", "tirex2",
          "chronos2_ft"]
UNCHANGED = ["persistence", "smart_persistence", "seasonal_naive", "climatology", "cliper",
             "sarima", "lightgbm", "lstm", "chronos2_ft"]
NEW = {"timesfm3": "google/timesfm-3.0-pytorch", "tirex2": "NX-AI/TiRex-2"}


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _models(cfg: dict) -> dict[str, dict]:
    return {m["name"]: m.get("params", {}) for m in cfg["models"]}


def _script(name: str):
    module_name = f"pvfc_{name}_script"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _at_import(path: str) -> dict:
    try:
        text = subprocess.run(["git", "show", f"{IMPORT_COMMIT}:{path}"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git show {IMPORT_COMMIT}:{path} is not available")
    return yaml.safe_load(text)


@pytest.fixture(scope="module")
def exp() -> dict:
    return _load(CONFIGS / "experiment.yaml")


# --------------------------------------------------------------------------- #
# configs/experiment.yaml
# --------------------------------------------------------------------------- #
def test_roster_is_the_13_models_of_the_definitive_run(exp):
    names = [m["name"] for m in exp["models"]]
    assert sorted(names) == sorted(ROSTER) and len(names) == 13


def test_unchanged_models_keep_the_params_of_their_earlier_runs(exp):
    old = _at_import("configs/experiment.yaml")
    cov = _models(_at_import("configs/covariates.yaml"))
    new, panel_v2 = _models(exp), _models(old)
    for name in UNCHANGED:
        assert new[name] == panel_v2[name], name
    assert new["chronos2_small"] == {**panel_v2["chronos2"], "arch": "chronos2"}
    assert new["chronos2"] == cov["chronos2_base_ctx8192"]
    for block in ("seed", "data", "splits", "evaluation"):
        assert exp[block] == old[block], block


def test_new_models_pin_revisions_at_their_maximum_context(exp):
    from pvfc.models.covariate_foundation import TimesFM3Forecaster, TiRex2Forecaster

    params = _models(exp)
    for name, cls in (("timesfm3", TimesFM3Forecaster), ("tirex2", TiRex2Forecaster)):
        p = params[name]
        assert p["arch"] == name and p["device"] == "mps", name
        assert re.fullmatch(r"[0-9a-f]{40}", str(p["revision"])), name
        assert p["quantile"] is True and p["output_chunk_length"] == 24
        assert p["input_chunk_length"] == cls.max_context
        # the history handed to the model fills the context (+1 day for the cut)
        assert p["context_days"] == math.ceil(cls.max_context / 24) + 1
    assert (params["timesfm3"]["input_chunk_length"], params["tirex2"]["input_chunk_length"]) \
        == (15_360, 2048)
    assert params["timesfm3"]["accept_license"] is True
    chronos = params["chronos2"]
    assert chronos["hub_model_name"] == "amazon/chronos-2"
    assert max(chronos["input_chunk_length"]) == 8192
    assert chronos["context_days"] * 24 >= 8192 + 24
    for name, p in params.items():
        assert "covariate_features" not in p and "xreg" not in p, name
        assert p.get("use_future_covariates", True), name


# --------------------------------------------------------------------------- #
# configs/dev_check.yaml
# --------------------------------------------------------------------------- #
def test_dev_check_runs_the_definitive_params(exp):
    dev = _load(CONFIGS / "dev_check.yaml")
    assert [m["name"] for m in dev["models"]] == ["cliper", "chronos2", "timesfm3", "tirex2"]
    for name, params in _models(dev).items():
        assert params == _models(exp)[name], name
    for block in ("seed", "data", "evaluation"):
        assert dev[block] == exp[block], block
    assert dev["splits"] == {"initial_train_days": 150, "first_test_day": "2022-01-04",
                             "test_days": 30, "embargo_days": 1, "step_days": 60,
                             "max_folds": 3, "day_tz": TZ}


def _dev_windows(index: pd.DatetimeIndex) -> list:
    from pvfc.splits import ExpandingWindowSplitter

    return list(ExpandingWindowSplitter(**_load(CONFIGS / "dev_check.yaml")["splits"])
                .split(index))


def test_dev_check_windows_are_development_days():
    # SQ567, the latest-starting panel station, from local midnight on 1 Jul 2021
    index = pd.date_range("2021-06-30 16:00", "2023-12-31 15:00", freq="h", tz="UTC")
    folds = _dev_windows(index)
    spans = []
    for fold in folds:
        days = fold.test_idx.tz_convert(TZ).normalize().unique()
        spans.append((str(days.min().date()), str(days.max().date()), len(days)))
        assert days.max() < pd.Timestamp("2022-06-04", tz=TZ)  # first inner window's embargo
        assert fold.train_idx.max() < fold.test_idx.min()
    assert spans == [("2022-01-04", "2022-02-02", 30), ("2022-03-05", "2022-04-03", 30),
                     ("2022-05-04", "2022-06-02", 30)]
    train_days = folds[0].train_idx.tz_convert(TZ).normalize().nunique()
    assert train_days >= 150


def test_every_panel_station_has_150_training_days_before_the_first_window(exp):
    from pvfc.data import ETL_VERSION, PROCESSED_DIR

    for site in exp["data"]["panel"]:
        cache = PROCESSED_DIR / f"{site}__nwp__etl{ETL_VERSION}.parquet"
        if not cache.exists():
            pytest.skip(f"no ETL cache {cache.name} (data/ is not in git)")
        index = pd.read_parquet(cache, columns=["power"]).index
        folds = _dev_windows(index)  # the splitter raises below initial_train_days
        assert len(folds) == 3, site
        assert folds[0].train_idx.tz_convert(TZ).normalize().nunique() >= 150, site


# --------------------------------------------------------------------------- #
# The batch plan and the checkpoints
# --------------------------------------------------------------------------- #
def test_panel_plan_is_156_runs_with_the_new_models_early(exp):
    run_panel = _script("run_panel")
    names = [m["name"] for m in exp["models"]]
    assert set(names) <= set(run_panel.RUN_ORDER) <= set(run_panel.SECONDS_PER_FOLD)
    plan = run_panel.build_plan(names, ["nwp", "none"], exp["data"]["panel"])
    assert len(plan) == 156 and len(set(plan)) == 156
    order = list(dict.fromkeys(m for m, _, _ in plan))
    assert order == [m for m in run_panel.RUN_ORDER if m in names]
    assert order[5:9] == ["chronos2", "chronos2_small", "tirex2", "timesfm3"]
    assert max(order.index("tirex2"), order.index("timesfm3")) < order.index("lightgbm")


def test_fetch_checkpoints_pins_the_config_revisions(exp):
    fetch = _script("fetch_checkpoints")
    params = _models(exp)
    for name, repo in NEW.items():
        got_repo, kwargs = fetch.CHECKPOINTS[name]
        assert got_repo == repo and kwargs["revision"] == params[name]["revision"], name
    assert fetch.CHECKPOINTS["tirex2"][1]["allow_patterns"] == ["model-config.yaml", "model.ckpt"]
    repos = {repo for repo, _ in fetch.CHECKPOINTS.values()}
    assert {p["hub_model_name"] for p in params.values() if "hub_model_name" in p} <= repos
    assert len(repos) == 4
    # every checkpoint by full commit hash; Chronos-2 at the revisions of DEFINITIVE_RUN.md
    # section 3, which darts finds at refs/main (fetch_checkpoints.pin_main)
    manifest = (ROOT / "experiments" / "batch_manifest.sh").read_text()
    for repo, kwargs in fetch.CHECKPOINTS.values():
        assert re.fullmatch(r"[0-9a-f]{40}", kwargs["revision"]) and kwargs["revision"] in manifest
    assert fetch.CHECKPOINTS["chronos2"][1]["revision"].startswith("29ec376")
    assert fetch.CHECKPOINTS["chronos2_small"][1]["revision"].startswith("ddec013")
    assert set(fetch.LOADED_AT_MAIN) == {"chronos2", "chronos2_small"}


def test_fetch_checkpoints_writes_a_missing_refs_main_and_refuses_another(tmp_path, monkeypatch):
    fetch = _script("fetch_checkpoints")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    pin = "29ec3766d36d6f73f0696f85560a422f50e8498c"
    assert fetch.pin_main("amazon/chronos-2", pin) == "refs/main written"
    ref = tmp_path / "models--amazon--chronos-2" / "refs" / "main"
    assert ref.read_text() == pin and fetch.pin_main("amazon/chronos-2", pin) == "refs/main ok"
    ref.write_text("f" * 40)
    with pytest.raises(SystemExit, match="not the pinned"):
        fetch.pin_main("amazon/chronos-2", pin)


# --------------------------------------------------------------------------- #
# The check scripts
# --------------------------------------------------------------------------- #
def _frame(start: str = "2022-01-03 16:00", hours: int = 48) -> pd.DataFrame:
    index = pd.date_range(start, periods=hours, freq="h", tz="UTC", name="time")
    hour = index.tz_convert(TZ).hour
    p_clear = np.where((hour >= 7) & (hour <= 17), 10.0, 0.0)
    frame = pd.DataFrame({"y_true": 0.5 * p_clear, "y_hat": 0.5 * p_clear}, index=index)
    for tau in (0.1, 0.5, 0.9):
        frame[f"q{tau}"] = tau * p_clear
    frame["ref_hat"] = frame["y_true"]
    frame["p_clear"], frame["is_day"], frame["fold"] = p_clear, p_clear > 0, 0
    frame["site"], frame["model"], frame["weather_mode"] = "X", "timesfm3", "nwp"
    return frame


def test_check_runs_flags_missing_unsorted_and_night_forecasts():
    check = _script("check_runs")
    assert check.check_file(_frame(), "timesfm3", rows=48, folds=1)["fail"] == []
    broken = _frame()
    broken.iloc[10, broken.columns.get_loc("q0.5")] = np.nan          # missing
    day = np.flatnonzero(broken["p_clear"].to_numpy() > 0)[0]
    broken.iloc[day, broken.columns.get_loc("q0.9")] = -1.0           # out of order
    broken.iloc[0, broken.columns.get_loc("y_hat")] = 0.2             # p_clear 0, not 0
    fails = " ".join(check.check_file(broken, "timesfm3", rows=48, folds=1)["fail"])
    assert "missing" in fails and "out of order" in fails and "p_clear is 0" in fails
    # persistence copies measured power, which may read above 0 at night
    night = _frame()
    night.iloc[0, night.columns.get_loc("y_hat")] = 0.01
    assert check.check_file(night, "persistence", rows=48, folds=1)["fail"] == []
    assert "48 rows" in check.check_file(night, "persistence", rows=72, folds=1)["fail"][0]
    assert check.REPRODUCE["chronos2"] == ("covariates/predictions", "chronos2_base_ctx8192")
    assert check.REPRODUCE["chronos2_small"] == ("predictions", "chronos2")


def test_dev_skill_is_refused_on_later_days():
    check = _script("check_runs")
    runs = {("timesfm3", "nwp", "X"): (Path("late.parquet"), _frame("2022-07-04 16:00"))}
    with pytest.raises(SystemExit, match="development days only"):
        check.report_dev_skill(runs)


def test_check_runs_plans_the_156_and_48_runs_and_their_memory_lines(exp):
    check = _script("check_runs")
    plan = check.load_plan(CONFIGS / "experiment.yaml")
    assert len(plan["runs"]) == len(set(plan["runs"])) == 156
    assert plan["mps"] == {"timesfm3", "tirex2"}
    dev = check.load_plan(CONFIGS / "dev_check.yaml")
    assert len(dev["runs"]) == 48 and dev["mps"] == {"timesfm3", "tirex2"}
    synthetic = check.load_plan(CONFIGS / "experiment.yaml", modes="nwp", sites="synthetic")
    assert len(synthetic["runs"]) == 13
    from pvfc.models.covariate_foundation import _NativeCovariateForecaster

    assert check.MEMORY_LOG_EVERY == _NativeCovariateForecaster.MEMORY_LOG_EVERY
    # 9 windows of 30 days: one memory line per window
    assert 6480 // 24 // check.MEMORY_LOG_EVERY == 9


MEM = ("12:00:00 INFO    pvfc.models.covariates | timesfm3: MPS driver memory, peak over the "
       "last 30 forecasts: {gb:.2f} GB\n")
LEDGER_HEAD = "started,model,weather_mode,site,status,exit_code,wall_s,log\n"


def _batch(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A complete mini batch: CLIPER and TimesFM-3 x 2 settings x stations A, B; one 30-day
    window; a ledger and a log per run (TimesFM-3: one memory line)."""
    cfg = _load(CONFIGS / "experiment.yaml")
    cfg["models"] = [m for m in cfg["models"] if m["name"] in ("cliper", "timesfm3")]
    cfg["data"]["panel"], cfg["splits"]["max_folds"] = ["A", "B"], 1
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = tmp_path / "mini.yaml"
    config.write_text(yaml.safe_dump(cfg))
    pred, logs = tmp_path / "pred", tmp_path / "logs"
    pred.mkdir()
    logs.mkdir()
    ledger = LEDGER_HEAD
    for model in ("cliper", "timesfm3"):
        for setting in ("nwp", "none"):
            for site in ("A", "B"):
                frame = _frame(hours=720)
                if model == "timesfm3" and setting == "none":
                    cols = ["y_hat", "q0.1", "q0.5", "q0.9"]
                    frame[cols] = frame[cols] * 0.9
                frame["site"], frame["model"], frame["weather_mode"] = site, model, setting
                frame.to_parquet(pred / f"{site}__{setting}__{model}__h.parquet")
                log = f"{site}__{setting}__{model}.log"
                (logs / log).write_text("INFO done\n" + (MEM.format(gb=1.2)
                                                          if model == "timesfm3" else ""))
                ledger += f"2026-10-15T01:00:00,{model},{setting},{site},ok,0,1,/gone/{log}\n"
    (logs / "panel_ledger.csv").write_text(ledger)
    return config, pred, logs


def _check(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "experiments" / "check_runs.py"),
                           *map(str, args)], cwd=ROOT, capture_output=True, text=True)


def test_check_runs_fails_an_incomplete_batch(tmp_path):
    config, pred, logs = _batch(tmp_path)
    base = ("--config", config, "--predictions-dir", pred, "--logs-dir", logs)
    out = _check(*base, "--expect-files", 8)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "8 of 8 planned runs" in out.stdout and "Structure: ok; ledgers and logs: ok" \
        in out.stdout
    assert "plan has 8 runs, expected 156" in _check(*base, "--expect-files", 156).stdout
    # a planned run without its file (and so without its NWP partner)
    (pred / "B__none__timesfm3__h.parquet").unlink()
    out = _check(*base)
    assert out.returncode == 1
    assert "no file: timesfm3 / none: B" in out.stdout and "B: no No-weather file" in out.stdout
    # a batch that stopped: the ledger lacks the last run
    _, pred, logs = _batch(tmp_path / "stopped")
    ledger = logs / "panel_ledger.csv"
    ledger.write_text("".join(ledger.read_text().splitlines(keepends=True)[:-1]))
    out = _check("--config", config, "--predictions-dir", pred, "--logs-dir", logs)
    assert out.returncode == 1 and "no ledger row: timesfm3 / none / B" in out.stdout
    # files outside the plan fail; synthetic-station files are left out with a note
    _, pred, logs = _batch(tmp_path / "extra")
    frame = _frame(hours=720)
    frame["site"], frame["model"], frame["weather_mode"] = "synthetic", "cliper", "nwp"
    frame.to_parquet(pred / "synthetic__nwp__cliper__h.parquet")
    out = _check("--config", config, "--predictions-dir", pred, "--logs-dir", logs)
    assert out.returncode == 0 and "1 synthetic-station files left out" in out.stdout
    frame["site"] = "C"
    frame.to_parquet(pred / "C__nwp__cliper__h.parquet")
    out = _check("--config", config, "--predictions-dir", pred, "--logs-dir", logs)
    assert out.returncode == 1 and "C__nwp__cliper__h.parquet: not in the plan" in out.stdout


def test_check_runs_needs_the_memory_lines_and_reads_the_last_attempt(tmp_path):
    config, pred, logs = _batch(tmp_path)
    base = ("--config", config, "--predictions-dir", pred)
    (logs / "A__nwp__timesfm3.log").write_text("INFO done\n")      # no memory line
    out = _check(*base, "--logs-dir", logs)
    assert out.returncode == 1
    assert "0 MPS driver memory lines for timesfm3, expected 1" in out.stdout
    (logs / "A__nwp__timesfm3.log").write_text(MEM.format(gb=4.5))
    out = _check(*base, "--logs-dir", logs)
    assert out.returncode == 1 and "ABOVE 4 GB" in out.stdout
    # a failed attempt, resumed into a new folder: the rerun's log counts, not the first one
    (logs / "A__nwp__timesfm3.log").write_text("Traceback (most recent call last):\n")
    ledger = logs / "panel_ledger.csv"
    ledger.write_text(ledger.read_text().replace("timesfm3,nwp,A,ok,0", "timesfm3,nwp,A,FAILED,1"))
    resume = tmp_path / "logs_resume1"
    resume.mkdir()
    (resume / "A__nwp__timesfm3.log").write_text(MEM.format(gb=1.5))
    (resume / "B__nwp__timesfm3.log").write_text("SKIP timesfm3: config h already in x\n")
    (resume / "panel_ledger.csv").write_text(
        LEDGER_HEAD + "2026-10-15T09:00:00,timesfm3,nwp,A,ok,0,1,A__nwp__timesfm3.log\n"
        "2026-10-15T09:01:00,timesfm3,nwp,B,skipped,0,1,B__nwp__timesfm3.log\n")
    assert _check(*base, "--logs-dir", logs).returncode == 1
    out = _check(*base, "--logs-dir", logs, resume)
    assert out.returncode == 0, out.stdout
    assert "1 earlier attempts" in out.stdout and "FAILED timesfm3 / nwp / A" in out.stdout
    # the logs of one run.py process, with no ledger: memory lines counted per model
    single = tmp_path / "single"
    single.mkdir()
    (single / "A__nwp__all.log").write_text(MEM.format(gb=1.0) * 3)
    out = _check(*base, "--logs-dir", single, "--no-ledger")
    assert out.returncode == 1 and "timesfm3: 3 MPS driver memory lines, expected 4" in out.stdout
    assert "no panel_ledger.csv" in _check(*base, "--logs-dir", single).stdout


def test_check_runs_counts_the_statsforecast_warning_without_failing(tmp_path):
    check = _script("check_runs")
    log = tmp_path / "SQ1__nwp__sarima.log"
    log.write_text(
        "/env/site-packages/statsforecast/arima.py:486: UserWarning: possible convergence "
        "problem: minimize gave code 2]\n  warnings.warn(\n"
        "12:00:00 INFO    pvfc.runner | model=sarima fold=0\n")
    res = check.scan_log(log)
    assert res["hits"] == [] and res["allowed"] == 1
    log.write_text(log.read_text() + "/x/y.py:1: UserWarning: something else\n  warnings.warn(\n")
    assert len(check.scan_log(log)["hits"]) == 2


def test_check_runs_reproduction_counts_a_one_sided_nan(tmp_path, capsys):
    check = _script("check_runs")
    old = _frame()
    old["site"], old["model"], old["weather_mode"] = "A", "cliper", "nwp"
    (tmp_path / "predictions").mkdir()
    old.to_parquet(tmp_path / "predictions" / "A__nwp__cliper__old.parquet")
    check.report_reproduction({("cliper", "nwp", "A"): (Path("new.parquet"), old.copy())},
                              tmp_path)
    assert "Reproduction: identical" in capsys.readouterr().out
    new = old.copy()
    new.iloc[12:15, new.columns.get_loc("y_hat")] = np.nan
    check.report_reproduction({("cliper", "nwp", "A"): (Path("new.parquet"), new)}, tmp_path)
    out = capsys.readouterr().out
    assert "1 files with a NaN on one side only" in out and "Reproduction: differs: cliper" in out


def test_covariate_response_refuses_test_days():
    out = subprocess.run([sys.executable, str(ROOT / "experiments" / "check_covariate_response.py"),
                          "--day", "2022-07-05"], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode != 0 and "development days only" in out.stderr
