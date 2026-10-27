"""Tests for the experiment scripts: the panel batch order and resumable reruns."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_run_panel():
    spec = importlib.util.spec_from_file_location("run_panel", ROOT / "experiments" / "run_panel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_panel_plan_is_model_major_in_run_order():
    plan = _load_run_panel().build_plan(
        ["chronos2_ft", "lightgbm", "persistence", "chronos2"], ["nwp", "none"], ["A", "B"])
    assert [m for m, _, _ in plan[::4]] == ["persistence", "chronos2", "lightgbm", "chronos2_ft"]
    assert plan[:4] == [("persistence", "nwp", "A"), ("persistence", "nwp", "B"),
                        ("persistence", "none", "A"), ("persistence", "none", "B")]


def test_skip_done_makes_a_rerun_free(tmp_path):
    """A batch restarted after an interruption must not recompute finished runs."""
    store = tmp_path / "results.parquet"
    cmd = [sys.executable, str(ROOT / "experiments" / "run.py"), "--synthetic",
           "--results", str(store), "--no-predictions", "--skip-done"]
    first = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=True)
    n_rows = len(pd.read_parquet(store))
    second = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=True)
    assert "SKIP" not in first.stdout and n_rows > 0
    # one SKIP per synthetic baseline, however many there are
    assert second.stdout.count("SKIP") == pd.read_parquet(store)["model"].nunique()
    assert len(pd.read_parquet(store)) == n_rows
