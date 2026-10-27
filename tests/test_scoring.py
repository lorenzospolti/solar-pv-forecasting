"""Tests for the scoring rules (scoring v2) and the rescorer (experiments/rescore.py).

The point metrics of a probabilistic model score the mean of its quantile grid
(``pvfc.evaluation.point_forecast``), the runner stamps every row with
``scoring_version``, and the rescorer must rebuild exactly the runner's rows from
the saved predictions under either version (with the runner's masks and capacity,
never running the ETL), carry what it cannot recompute, keep rows it cannot
rescore, and add ``skill_cliper`` (the same site's CLIPER, joined on time). Also:
the known-harmless library warnings are silenced, and nothing else is.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pvfc import runner as runner_mod
from pvfc.data import ETL_VERSION, synthetic_station
from pvfc.evaluation import (
    DEFAULT_QUANTILES,
    METRIC_COLUMNS,
    SCORING_VERSION,
    point_forecast,
    point_forecast_frame,
    rmse,
)
from pvfc.models.base import Forecaster
from pvfc.models.baselines import Persistence, SmartPersistence
from pvfc.runner import append_results, run_model
from pvfc.splits import ExpandingWindowSplitter
from pvfc.utils import KNOWN_HARMLESS_WARNINGS, quiet_known_warnings, setup_logging

ROOT = Path(__file__).resolve().parents[1]
HKT = "Asia/Hong_Kong"
KEY = ["config_hash", "fold", "daytime_only"]
TIMING = ["fit_s", "predict_s", "fit_predict_s", "peak_rss_mb"]


def _load_rescore():
    spec = importlib.util.spec_from_file_location("rescore", ROOT / "experiments" / "rescore.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module's annotations here
    spec.loader.exec_module(module)
    return module


rescore = _load_rescore()


# ------------------------------------------------------------ point forecast ---
def _grid(n: int = 5, seed: int = 0) -> dict[float, np.ndarray]:
    rows = np.sort(np.random.default_rng(seed).uniform(0, 10, (len(DEFAULT_QUANTILES), n)), axis=0)
    return {tau: rows[i] for i, tau in enumerate(DEFAULT_QUANTILES)}


def test_point_forecast_is_the_grid_mean():
    q = _grid()
    expected = np.mean([q[tau] for tau in DEFAULT_QUANTILES], axis=0)
    np.testing.assert_allclose(point_forecast(q[0.5], q), expected, rtol=1e-15)
    assert not np.allclose(expected, q[0.5])  # a random grid: mean != median


def test_point_forecast_falls_back_to_y_hat():
    q, y_hat = _grid(), np.full(5, -1.0)
    for deterministic in (None, {}):
        np.testing.assert_array_equal(point_forecast(y_hat, deterministic), y_hat)
    incomplete = {tau: v for tau, v in q.items() if tau != 0.3}
    np.testing.assert_array_equal(point_forecast(y_hat, incomplete), y_hat)
    q[0.9] = q[0.9].copy()
    q[0.9][2] = np.nan  # one missing quantile value: y_hat at that timestamp only
    out = point_forecast(y_hat, q)
    assert out[2] == -1.0
    rest = [0, 1, 3, 4]
    np.testing.assert_allclose(out[rest], np.mean([q[t][rest] for t in DEFAULT_QUANTILES], axis=0))


def test_point_forecast_frame_reads_the_saved_columns():
    q = _grid()
    frame = pd.DataFrame({"y_true": 1.0, "y_hat": q[0.5], **{f"q{t}": v for t, v in q.items()},
                          "quality": 99.0})  # starts with "q" but is not a quantile
    np.testing.assert_array_equal(point_forecast_frame(frame), point_forecast(q[0.5], q))
    np.testing.assert_array_equal(point_forecast_frame(frame[["y_true", "y_hat"]]), q[0.5])


# -------------------------------------------------------------------- runner ---
class _SkewedQuantiles(Forecaster):
    """Smart persistence as the median, inside a right-skewed quantile grid (a long upper
    tail, as for PV on a broken-cloud day), so the grid mean sits above the median."""

    name = "skewed"
    supports_quantiles = True

    def fit(self, train):
        return self

    def predict(self, history, future):
        med = SmartPersistence().predict(history, future)["y_hat"].to_numpy()
        out = {"y_hat": med}
        for tau in DEFAULT_QUANTILES:
            out[f"q{tau}"] = med * (1.0 + (1.5 if tau > 0.5 else 0.3) * (tau - 0.5))
        return pd.DataFrame(out, index=future.index)


def _splitter() -> ExpandingWindowSplitter:
    return ExpandingWindowSplitter(initial_train_days=300, test_days=8, embargo_days=1,
                                   max_folds=2, day_tz=HKT)


@pytest.fixture(scope="module")
def series():
    """A synthetic station with three daytime data gaps in the first test window, so the
    finite-truth part of the mask is exercised too."""
    s = synthetic_station(days=330, seed=5)
    test_idx = next(_splitter().split(s.frame.index)).test_idx
    day_hours = test_idx[s.frame.loc[test_idx, "is_day"].to_numpy(dtype=bool)]
    frame = s.frame.copy()
    frame.loc[day_hours[[3, 4, 40]], ["power", "k"]] = np.nan
    return dataclasses.replace(s, frame=frame)


def _capacity(series):
    return lambda site, weather_mode: series.capacity


def _masked(f: pd.DataFrame, daytime_only: bool) -> np.ndarray:
    day = f["is_day"].to_numpy(dtype=bool) if daytime_only else np.ones(len(f), dtype=bool)
    return day & np.isfinite(f["y_true"].to_numpy())


def test_runner_scores_a_quantile_model_on_its_grid_mean(tmp_path, series):
    rows = run_model(_SkewedQuantiles(), series, _splitter(), predictions_dir=tmp_path)
    pred = pd.read_parquet(next(tmp_path.glob("*.parquet")))
    np.testing.assert_array_equal(pred["y_hat"], pred["q0.5"])  # saved output stays raw
    assert pred["y_true"].isna().sum() == 3
    for row in rows:
        f = pred[pred["fold"] == row["fold"]]
        mask = _masked(f, row["daytime_only"])
        y = f["y_true"].to_numpy()[mask]
        grid_mean = np.mean([f[f"q{t}"].to_numpy()[mask] for t in DEFAULT_QUANTILES], axis=0)
        median = f["y_hat"].to_numpy()[mask]
        assert row["n_test"] == mask.sum()
        assert row["rmse"] == pytest.approx(rmse(y, grid_mean), rel=1e-12)
        assert row["rmse"] != pytest.approx(rmse(y, median), rel=1e-6)
        assert row["mbe"] == pytest.approx(np.mean(grid_mean - y), rel=1e-12)
        assert row["scoring_version"] == SCORING_VERSION == 2


def test_append_results_keeps_metric_columns_float(tmp_path, series):
    """A deterministic model leaves its probabilistic metrics None; the store's metric
    columns must stay float64 (pandas 3 keeps all-NA columns when concat picks dtypes)."""
    store = tmp_path / "results.parquet"
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="The behavior of DataFrame concatenation",
                                category=FutureWarning)
        for model in (Persistence(), _SkewedQuantiles(), Persistence()):
            append_results(run_model(model, series, _splitter()), str(store))
    df = pd.read_parquet(store)
    assert all(df[c].dtype == "float64" for c in METRIC_COLUMNS)


# ------------------------------------------------------------------- rescore ---
def _run_panel(series, pred_dir: Path, store_path: Path, v1: bool = False,
               daytime_modes: tuple[bool, ...] = (True, False)) -> pd.DataFrame:
    """Two models through the runner into a store, as run.py does. ``v1=True`` is the
    runner as it was before scoring v2 (point metrics on the raw ``y_hat``): how the
    2 Oct 2026 panel was scored."""
    with pytest.MonkeyPatch.context() as mp:
        if v1:
            mp.setattr(runner_mod, "point_forecast",
                       lambda y_hat, q_pred=None: np.asarray(y_hat, dtype=float))
            mp.setattr(runner_mod, "SCORING_VERSION", 1)
        for model in (Persistence(), _SkewedQuantiles()):
            cfg = {"data": {"site_id": series.site_id, "weather_mode": "nwp"},
                   "model": {"name": model.name}}
            append_results(run_model(model, series, _splitter(), daytime_modes=daytime_modes,
                                     config=cfg, predictions_dir=pred_dir), str(store_path))
    return pd.read_parquet(store_path)


def _fake_cliper(pred_dir: Path, weather_mode: str, scale: float, chash: str,
                 folds: tuple[int, ...] = (0, 1), scramble: bool = False) -> pd.DataFrame:
    """A CLIPER predictions file: the persistence run's frame with a climatology-like
    forecast (``scale`` x clear-sky power), for the folds given. ``scramble`` writes
    the rows shuffled and on a Hong Kong time index, so that only a join on time
    lines them up with a run's hours."""
    src = pd.read_parquet(next(pred_dir.glob("synthetic__*__persistence__*.parquet")))
    fake = src[src["fold"].isin(folds)].copy()
    fake["y_hat"] = scale * fake["p_clear"]
    fake["model"], fake["weather_mode"], fake["config_hash"] = "cliper", weather_mode, chash
    if scramble:
        fake = fake.sample(frac=1.0, random_state=0).tz_convert(HKT)
    fake.to_parquet(pred_dir / f"synthetic__{weather_mode}__cliper__{chash}.parquet")
    return fake


def _other_site(pred_dir: Path, store: pd.DataFrame, chash: str = "0ther5ite001") -> pd.DataFrame:
    """The skewed run again as a second station ("other", its own config hash): writes
    its predictions file next to the first station's and returns its store rows."""
    src = pd.read_parquet(next(pred_dir.glob("synthetic__*__skewed__*.parquet")))
    src.assign(site="other", config_hash=chash).to_parquet(
        pred_dir / f"other__nwp__skewed__{chash}.parquet")
    return store[store["model"] == "skewed"].assign(site="other", config_hash=chash)


def _assert_skill_cliper(rows: pd.DataFrame, cliper: pd.DataFrame) -> None:
    """Each row's skill_cliper = 1 - RMSE / RMSE_CLIPER on the row's hours, computed from
    the CLIPER frame alone (an hour's truth and forecast share a row there, whatever
    the row order or time zone)."""
    assert len(rows)
    for _, row in rows.iterrows():
        f = cliper[cliper["fold"] == row["fold"]]
        mask = _masked(f, row["daytime_only"])
        ref = rmse(f["y_true"].to_numpy()[mask], f["y_hat"].to_numpy()[mask])
        assert row["skill_cliper"] == pytest.approx(1.0 - row["rmse"] / ref, rel=1e-12)


def _no_etl(*args, **kwargs):
    raise AssertionError("rescore must not run the ETL (pvfc.data.load_station)")


def test_rescore_knows_the_current_scoring_version():
    assert rescore.SCORING_VERSIONS == {"native": 1, "mean": SCORING_VERSION}


@pytest.mark.parametrize("point,v1", [("mean", False), ("native", True)])
def test_rescore_rebuilds_the_runner_rows_exactly(tmp_path, series, point, v1):
    store = _run_panel(series, tmp_path / "pred", tmp_path / "results.parquet", v1=v1)
    out, summary = rescore.rescore(store, tmp_path / "pred", point=point,
                                   capacity_fn=_capacity(series))
    assert summary.n_carried == len(store) == len(out) == 8
    assert not summary.unmatched and summary.kept.empty
    got, want = out.set_index(KEY).sort_index(), store.set_index(KEY).sort_index()
    pd.testing.assert_frame_equal(got[want.columns], want, check_exact=True)
    assert (got["scoring_version"] == (1 if v1 else 2)).all()
    assert got["skill_cliper"].isna().all()  # no CLIPER predictions


def test_store_rows_without_predictions_are_kept(tmp_path, series):
    store = _run_panel(series, tmp_path / "pred", tmp_path / "results.parquet")
    persistence = store[store["model"] == "persistence"]
    old = persistence.assign(config_hash="0123456789ab", site="old-site")
    old = old.drop(columns="scoring_version")  # written before the column existed
    adhoc = persistence.assign(config_hash=None)  # a run without a config (no hash)
    out, summary = rescore.rescore(pd.concat([store, old, adhoc], ignore_index=True),
                                   tmp_path / "pred", capacity_fn=_capacity(series))
    assert len(out) == len(store) + len(old) + len(adhoc)
    assert len(summary.kept) == len(old) + len(adhoc) and summary.kept_stale == 0
    kept = out[out["config_hash"] == "0123456789ab"].set_index(KEY).sort_index()
    want = old.set_index(KEY).sort_index()
    pd.testing.assert_frame_equal(kept[want.columns], want, check_exact=True)
    assert (kept["scoring_version"] == 1).all()  # the rules it was scored under
    hashless = out[out["config_hash"].isna()].reset_index(drop=True)
    pd.testing.assert_frame_equal(hashless[adhoc.columns], adhoc.reset_index(drop=True),
                                  check_exact=True)


def test_predictions_without_store_rows_are_scored_with_nan_timing(tmp_path, series):
    store = _run_panel(series, tmp_path / "pred", tmp_path / "results.parquet")
    skewed = store.loc[store["model"] == "skewed", "config_hash"].iloc[0]
    for partial in (store[store["config_hash"] != skewed], None):
        out, summary = rescore.rescore(partial, tmp_path / "pred", capacity_fn=_capacity(series))
        rows = out[out["config_hash"] == skewed].set_index(KEY).sort_index()
        assert any("skewed" in name for name in summary.unmatched)
        assert rows[TIMING].isna().all().all() and rows["env"].isna().all()
        want = store[store["config_hash"] == skewed].set_index(KEY).sort_index()
        pd.testing.assert_series_equal(rows["rmse"], want["rmse"], check_exact=True)


def test_skill_cliper_prefers_the_same_weather_else_any(tmp_path, series):
    pred_dir = tmp_path / "pred"
    store = _run_panel(series, pred_dir, tmp_path / "results.parquet")

    other = _fake_cliper(pred_dir, "none", 0.3, "c11fe5000002")  # only another weather mode
    out, summary = rescore.rescore(store, pred_dir, capacity_fn=_capacity(series))
    _assert_skill_cliper(out[out["model"] == "skewed"], other)
    assert set(summary.cliper_used.values()) == {"synthetic__none__cliper__c11fe5000002.parquet"}

    same = _fake_cliper(pred_dir, "nwp", 0.6, "c11fe5000001")  # now the same weather mode
    out, _ = rescore.rescore(store, pred_dir, capacity_fn=_capacity(series))
    _assert_skill_cliper(out[out["model"] == "skewed"], same)
    assert np.allclose(out.loc[out["model"] == "cliper", "skill_cliper"], 0.0)  # vs itself


def test_skill_cliper_is_the_same_sites_cliper_joined_on_time(tmp_path, series):
    """Another station's CLIPER never counts (on the panel all stations share one time
    index, so it would 'cover' every hour), and CLIPER's rows meet the run's by
    timestamp: the fake covers fold 1 only, shuffled, on a Hong Kong time index."""
    pred_dir = tmp_path / "pred"
    store = _run_panel(series, pred_dir, tmp_path / "results.parquet")
    _other_site(pred_dir, store)
    cliper = _fake_cliper(pred_dir, "nwp", 0.6, "c11fe5000001", folds=(1,), scramble=True)
    out, summary = rescore.rescore(store, pred_dir, capacity_fn=_capacity(series))
    assert out.loc[out["site"] == "other", "skill_cliper"].isna().all()
    assert "other__nwp__skewed__0ther5ite001.parquet" not in summary.cliper_used
    skewed = out[(out["model"] == "skewed") & (out["site"] == "synthetic")]
    assert skewed.loc[skewed["fold"] == 0, "skill_cliper"].isna().all()
    _assert_skill_cliper(skewed[skewed["fold"] == 1], cliper)


def test_skill_cliper_prefers_the_current_cliper_config(tmp_path, series):
    """When CLIPER's settings change, the old config's file stays behind: the config in
    the results store wins, then the newest file, as report.py picks, never merely the
    file name; the unused file is reported."""
    pred_dir = tmp_path / "pred"
    store = _run_panel(series, pred_dir, tmp_path / "results.parquet")
    current = _fake_cliper(pred_dir, "nwp", 0.6, "fakec1iper01")
    _fake_cliper(pred_dir, "nwp", 0.9, "0aaaa5ta1e00")  # superseded; its name sorts first
    path = {chash: pred_dir / f"synthetic__nwp__cliper__{chash}.parquet"
            for chash in ("fakec1iper01", "0aaaa5ta1e00")}
    now = time.time()

    def used(store_: pd.DataFrame, old_age_days: float) -> pd.DataFrame:
        os.utime(path["0aaaa5ta1e00"], (now - 86400 * old_age_days,) * 2)
        os.utime(path["fakec1iper01"], (now - 86400,) * 2)
        out, summary = rescore.rescore(store_, pred_dir, capacity_fn=_capacity(series))
        assert set(summary.cliper_used.values()) == {path["fakec1iper01"].name}
        assert summary.cliper_ignored == [path["0aaaa5ta1e00"].name]
        return out

    out = used(store, old_age_days=2)  # neither config in the store: the newest file
    _assert_skill_cliper(out[out["model"] == "skewed"], current)
    in_store = store[store["model"] == "persistence"].assign(model="cliper",
                                                             config_hash="fakec1iper01")
    used(pd.concat([store, in_store], ignore_index=True), old_age_days=0)  # store beats newer


def test_rescore_scores_only_the_masks_the_store_holds(tmp_path, series):
    """A config run with ``evaluation.daytime_only: false`` has all-hours rows only: the
    rebuilt store must not grow daytime rows the runner never produced. Predictions
    without store rows get both masks."""
    pred_dir = tmp_path / "pred"
    store = _run_panel(series, pred_dir, tmp_path / "results.parquet", daytime_modes=(False,))
    out, summary = rescore.rescore(store, pred_dir, capacity_fn=_capacity(series))
    assert len(out) == len(store) == summary.n_carried == 4 and not out["daytime_only"].any()
    got, want = out.set_index(KEY).sort_index(), store.set_index(KEY).sort_index()
    pd.testing.assert_frame_equal(got[want.columns], want, check_exact=True)
    alone, _ = rescore.rescore(None, pred_dir, capacity_fn=_capacity(series))
    assert len(alone) == 8 and alone["daytime_only"].sum() == 4


def test_station_capacity_reads_the_sidecars_never_the_etl(tmp_path, monkeypatch, series):
    monkeypatch.setattr(rescore.datamod, "load_station", _no_etl)
    monkeypatch.setattr(rescore.datamod, "PROCESSED_DIR", tmp_path)
    for site, mode, kw in (("SQ1", "nwp", 27.6), ("SQ1", "none", 27.5), ("LSK North", "none", 55.0)):
        (tmp_path / f"{site}__{mode}__etl{ETL_VERSION}.cap.json").write_text(
            json.dumps({"capacity": kw}))
    before = sorted(tmp_path.iterdir())
    assert rescore.station_capacity("SQ1", "nwp") == 27.6  # the run's own weather mode first
    assert rescore.station_capacity("SQ1", "none") == 27.5
    assert rescore.station_capacity("LSK North", "nwp") == 55.0  # else another mode's
    assert rescore.station_capacity("LSK North", None) == 55.0
    assert rescore.station_capacity("synthetic", "nwp") == series.capacity  # make smoke
    assert rescore.station_capacity("SQ567", "oracle") is None  # no sidecar: no ETL either
    assert sorted(tmp_path.iterdir()) == before


def test_rescore_default_capacity_rebuilds_a_synthetic_run(tmp_path, monkeypatch, series):
    """``make smoke`` saves synthetic predictions next to the panel's; the default
    capacity lookup must rescore them exactly, without the ETL."""
    monkeypatch.setattr(rescore.datamod, "load_station", _no_etl)
    monkeypatch.setattr(rescore.datamod, "PROCESSED_DIR", tmp_path / "processed")
    store = _run_panel(series, tmp_path / "pred", tmp_path / "results.parquet")
    out, summary = rescore.rescore(store, tmp_path / "pred")
    assert summary.n_carried == len(store) == len(out) and not summary.no_capacity
    got, want = out.set_index(KEY).sort_index(), store.set_index(KEY).sort_index()
    pd.testing.assert_frame_equal(got[want.columns], want, check_exact=True)


def test_runs_without_a_capacity_keep_their_store_rows(tmp_path, series):
    pred_dir = tmp_path / "pred"
    store = _run_panel(series, pred_dir, tmp_path / "results.parquet")
    other = _other_site(pred_dir, store)
    full = pd.concat([store, other], ignore_index=True)
    out, summary = rescore.rescore(
        full, pred_dir, capacity_fn=lambda site, weather_mode: None if site == "other"
        else series.capacity)
    assert summary.no_capacity == ["other__nwp__skewed__0ther5ite001.parquet"]
    assert summary.no_capacity_rows == len(other) and summary.kept.empty
    assert not summary.unmatched and summary.n_files == 2 and len(out) == len(full)
    kept = out[out["site"] == "other"].set_index(KEY).sort_index()
    pd.testing.assert_frame_equal(kept[other.columns.drop(KEY)],
                                  other.set_index(KEY).sort_index(), check_exact=True)
    assert "WARNING: no capacity" in rescore.describe(summary, out)


def test_skill_cliper_is_nan_where_cliper_misses_scored_hours(tmp_path, series):
    pred_dir = tmp_path / "pred"
    store = _run_panel(series, pred_dir, tmp_path / "results.parquet")
    _fake_cliper(pred_dir, "nwp", 0.6, "c11fe5000001", folds=(0,))
    out, _ = rescore.rescore(store, pred_dir, capacity_fn=_capacity(series))
    skewed = out[out["model"] == "skewed"]
    assert skewed.loc[skewed["fold"] == 0, "skill_cliper"].notna().all()
    assert skewed.loc[skewed["fold"] == 1, "skill_cliper"].isna().all()


def test_rescore_is_idempotent_ordered_and_keeps_the_schema(tmp_path, series):
    pred_dir = tmp_path / "pred"
    store = _run_panel(series, pred_dir, tmp_path / "results.parquet")
    _fake_cliper(pred_dir, "nwp", 0.6, "c11fe5000001")
    out, _ = rescore.rescore(store, pred_dir, capacity_fn=_capacity(series))
    path = tmp_path / "rescored.parquet"
    rescore.write_store(out, path)
    written = pd.read_parquet(path)
    again, _ = rescore.rescore(written, pred_dir, capacity_fn=_capacity(series))
    pd.testing.assert_frame_equal(again, written, check_exact=True)
    pd.testing.assert_frame_equal(written, out, check_exact=True)
    assert not out.duplicated(KEY).any()
    order = out.sort_values(["site", "weather_mode", "model", "config_hash", "fold",
                             "daytime_only"], ascending=[True] * 5 + [False])
    assert order.index.equals(out.index)
    assert list(out.columns[:len(store.columns)]) == list(store.columns)
    assert all(out[c].dtype == store[c].dtype for c in store.columns)
    assert out["scoring_version"].dtype == "int64" and out["skill_cliper"].dtype == "float64"


# ------------------------------------------------------------------ warnings ---
def test_only_the_known_harmless_warnings_are_silenced():
    harmless = [
        ("X does not have valid feature names, but LGBMRegressor was fitted with feature "
         "names", UserWarning, "sklearn.utils.validation"),
        ("'pin_memory' argument is set as true but not supported on MPS now, device pinned "
         "memory won't be used.", UserWarning, "torch.utils.data.dataloader"),
        ("`isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, "
         "TreeSpec) and treespec.is_leaf()` instead.", FutureWarning,
         "pytorch_lightning.utilities._pytree"),
        ("Total length of `list` across ranks is zero. Please make sure this was your "
         "intention.", UserWarning, "pytorch_lightning.utilities.data"),
    ]
    informative = [
        ("possible convergence problem: minimize gave code 1]", UserWarning,
         "statsforecast.arima"),
        (harmless[0][0], UserWarning, "pvfc.runner"),  # same text, someone else's module
        ("X has feature names, but LGBMRegressor was fitted without feature names",
         UserWarning, "sklearn.utils.validation"),
        (harmless[1][0], DeprecationWarning, "torch.utils.data.dataloader"),  # other category
        # the same lightning check on an empty training loader: training would not happen
        ("Total length of `DataLoader` across ranks is zero. Please make sure this was your "
         "intention.", UserWarning, "pytorch_lightning.utilities.data"),
        ("Total length of `CombinedLoader` across ranks is zero. Please make sure this was "
         "your intention.", UserWarning, "pytorch_lightning.utilities.data"),
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quiet_known_warnings()
        quiet_known_warnings()  # idempotent: the filters are not stacked twice
        n_ours = sum(f[0] == "ignore" and f[1] is not None
                     and f[1].pattern in {m for m, _, _ in KNOWN_HARMLESS_WARNINGS}
                     for f in warnings.filters)
        for message, category, module in harmless + informative:
            warnings.warn_explicit(message, category, module.replace(".", "/") + ".py", 1,
                                   module=module)
    assert n_ours == len(KNOWN_HARMLESS_WARNINGS)
    assert [str(w.message) for w in caught] == [m for m, _, _ in informative]


def test_setup_logging_installs_the_filters(monkeypatch):
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: None)
    with warnings.catch_warnings():
        warnings.resetwarnings()
        setup_logging()
        patterns = {f[1].pattern for f in warnings.filters if f[0] == "ignore" and f[1]}
    assert {m for m, _, _ in KNOWN_HARMLESS_WARNINGS} <= patterns
