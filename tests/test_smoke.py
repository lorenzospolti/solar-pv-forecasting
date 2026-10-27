"""Harness smoke tests (plan §Verify step 2).

These run WITHOUT any downloaded data, using a synthetic station, and prove the
spine works end-to-end:
- persistence and smart persistence run through the Runner;
- a tidy results row with RMSE / nMAE / skill score is produced;
- smart-persistence skill against itself is ~0; naive persistence skill is < 0
  (i.e. naive persistence is worse than smart persistence, as expected).
"""
from __future__ import annotations

import numpy as np

from pvfc.data import synthetic_station
from pvfc.evaluation import enforce_monotone_quantiles, evaluate, skill_score
from pvfc.models.baselines import Persistence, SmartPersistence
from pvfc.runner import run_model
from pvfc.splits import ExpandingWindowSplitter


def _splitter():
    return ExpandingWindowSplitter(initial_train_days=300, test_days=20, embargo_days=1, max_folds=2)


def test_metrics_basic():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    assert evaluate(y, y, capacity=5.0).rmse == 0.0
    assert skill_score(0.5, 1.0) == 0.5  # half the reference error


def test_quantile_monotonicity():
    q = {0.1: np.array([2.0, 2.0]), 0.5: np.array([1.0, 1.0]), 0.9: np.array([3.0, 3.0])}
    fixed = enforce_monotone_quantiles(q)
    assert np.all(fixed[0.1] <= fixed[0.5]) and np.all(fixed[0.5] <= fixed[0.9])


def test_persistence_runs_through_harness():
    series = synthetic_station(days=360, seed=1)
    rows = run_model(Persistence(), series, _splitter())
    assert rows, "expected at least one fold"
    assert {"model", "fold", "rmse", "nrmse", "skill_rmse"} <= set(rows[0])
    assert np.isfinite(rows[0]["nrmse"])


def test_smart_persistence_beats_naive():
    series = synthetic_station(days=360, seed=2)
    sp = _splitter()
    persist = run_model(Persistence(), series, sp)
    smart = run_model(SmartPersistence(), series, sp)
    # smart-persistence skill measured against itself should be ~0
    assert abs(smart[0]["skill_rmse"]) < 1e-6
    # naive persistence should be no better than smart persistence (skill <= ~0)
    assert persist[0]["skill_rmse"] <= 1e-6
