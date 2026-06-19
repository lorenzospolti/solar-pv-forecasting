"""Regression tests for the Moirai-2 and TTM external-foundation wrappers (v0.5).

These tests don't need the real `uni2ts` / `granite-tsfm` libraries: they mock
the dep at the wrapper boundary (`_load_model` / `_forecast`) so CI stays
dependency-light, mirroring how `test_smoke.py` runs without darts/torch.

What we verify:
- (a) `models.build()` returns the right wrapper class for `moirai2`, `moirai2_ft`,
  `ttm`, `ttm_ft` — and the arch/label split is respected
  (`moirai2_ft` uses `arch=moirai2` and is a separate object than `moirai2`).
- (b) Calling `fit()` without the optional dep raises ImportError with the
  install instructions in the message (no traceback at predict time later).
- (c) An end-to-end Runner pass that mocks the model produces *distinct* result
  rows for the zero-shot label and the fine-tuned label — proving the
  arch/label split flows all the way through to the results table.
- (d) Mocked predict() output is reconstructed back to power via `p_clear` and
  populates quantile columns when `quantile=True`.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from pvfc.data import synthetic_station
from pvfc.models import build
from pvfc.models.external_foundation import (
    EXTERNAL_FOUNDATION_ARCHES,
    Moirai2Forecaster,
    TTMForecaster,
)
from pvfc.runner import run_model
from pvfc.splits import ExpandingWindowSplitter


def _splitter():
    return ExpandingWindowSplitter(initial_train_days=300, test_days=10,
                                   embargo_days=1, max_folds=1)


# --------------------------------------------------------------------- (a) ---
def test_external_arches_registered():
    assert EXTERNAL_FOUNDATION_ARCHES == frozenset({"moirai2", "ttm"})


def test_build_returns_correct_wrapper_class():
    m = build("moirai2", input_chunk_length=512, output_chunk_length=24)
    t = build("ttm", input_chunk_length=512, output_chunk_length=24)
    assert isinstance(m, Moirai2Forecaster) and m.arch == "moirai2"
    assert isinstance(t, TTMForecaster) and t.arch == "ttm"


def test_arch_label_split_keeps_classes_but_different_labels():
    """`moirai2_ft` with arch=moirai2 must be a Moirai2Forecaster with the FT
    label preserved — i.e. the same arch can back two different results rows."""
    zs = build("moirai2", input_chunk_length=512)
    ft = build("moirai2_ft", arch="moirai2", input_chunk_length=512, finetune=True)
    assert isinstance(zs, Moirai2Forecaster) and isinstance(ft, Moirai2Forecaster)
    assert zs.name == "moirai2" and ft.name == "moirai2_ft"
    assert zs is not ft and zs.finetune is False and ft.finetune is True

    zs_t = build("ttm", input_chunk_length=512)
    ft_t = build("ttm_ft", arch="ttm", input_chunk_length=512, finetune=True)
    assert isinstance(zs_t, TTMForecaster) and isinstance(ft_t, TTMForecaster)
    assert zs_t.name == "ttm" and ft_t.name == "ttm_ft"


# --------------------------------------------------------------------- (b) ---
@pytest.mark.parametrize(
    "name, expected_pkg",
    [("moirai2", "uni2ts"), ("ttm", "granite-tsfm")],
)
def test_missing_dep_raises_with_install_instructions(name, expected_pkg):
    """Loading without the optional dep must raise ImportError carrying the
    `pip install ...` line for the missing package."""
    model = build(name, input_chunk_length=64)

    # Patch the dep-importer to simulate "library not installed". `_load_model`
    # catches this and re-raises ImportError with the install hint.
    def boom(*args, **kwargs):
        raise ImportError("simulated missing dep")

    target_cls = "Moirai2Forecaster" if name == "moirai2" else "TTMForecaster"
    target = f"pvfc.models.external_foundation.{target_cls}._import_deps"
    train = synthetic_station(days=30, seed=0).frame
    with patch(target, new=boom), pytest.raises(ImportError) as excinfo:
        model.fit(train)
    msg = str(excinfo.value)
    assert "pip install" in msg
    assert expected_pkg in msg


# --------------------------------------------------------------------- (c) ---
class _FakeMoiraiOrTTM:
    """Minimal mock standing in for the externally-loaded model object."""


def _mock_load(self):
    return _FakeMoiraiOrTTM()


def _mock_forecast(self, context, n):
    # deterministic flat forecast in k-space; quantiles are tiny symmetric bands
    point = np.full(n, 0.5, dtype="float32")
    if self.supports_quantiles and self.quantiles:
        q = {round(float(tau), 2): point + (float(tau) - 0.5) * 0.2
             for tau in self.quantiles}
        return point, q
    return point, None


def _run_two_models(zs_name, ft_name, arch):
    series = synthetic_station(days=360, seed=11)
    sp = _splitter()
    zs = build(zs_name, input_chunk_length=128, quantile=True)
    ft = build(ft_name, arch=arch, input_chunk_length=128, quantile=True, finetune=True)
    cls = Moirai2Forecaster if arch == "moirai2" else TTMForecaster
    with patch.object(cls, "_load_model", new=_mock_load), \
            patch.object(cls, "_forecast", new=_mock_forecast):
        rows_zs = run_model(zs, series, sp)
        rows_ft = run_model(ft, series, sp)
    return rows_zs, rows_ft


def test_moirai2_arch_label_split_produces_distinct_rows():
    rows_zs, rows_ft = _run_two_models("moirai2", "moirai2_ft", "moirai2")
    assert rows_zs and rows_ft
    assert {r["model"] for r in rows_zs} == {"moirai2"}
    assert {r["model"] for r in rows_ft} == {"moirai2_ft"}
    # quantile columns from the mock should populate cov_q*
    head = rows_zs[0]
    assert any(k.startswith("cov_q") for k in head)


def test_ttm_arch_label_split_produces_distinct_rows():
    rows_zs, rows_ft = _run_two_models("ttm", "ttm_ft", "ttm")
    assert rows_zs and rows_ft
    assert {r["model"] for r in rows_zs} == {"ttm"}
    assert {r["model"] for r in rows_ft} == {"ttm_ft"}


# --------------------------------------------------------------------- (d) ---
def test_predict_reconstructs_power_and_emits_quantile_cols():
    """Predict output: y_hat = k_hat * p_clear, with q0.1..q0.9 when quantile=True."""
    series = synthetic_station(days=360, seed=22)
    sp = _splitter()
    model = build("moirai2", input_chunk_length=64, quantile=True)
    with patch.object(Moirai2Forecaster, "_load_model", new=_mock_load), \
            patch.object(Moirai2Forecaster, "_forecast", new=_mock_forecast):
        rows = run_model(model, series, sp)
    assert rows
    row = rows[0]
    # quantile reliability columns are produced -> q_pred path was hit
    qcols = [k for k in row if k.startswith("cov_q")]
    assert len(qcols) >= 3, f"expected per-quantile coverage columns, got {qcols}"
    # the mocked forecast is constant (0.5), so nRMSE is finite/non-zero
    assert np.isfinite(row["nrmse"]) and row["nrmse"] > 0


def test_predict_raises_when_called_before_fit():
    model = build("ttm", input_chunk_length=64)
    series = synthetic_station(days=30, seed=33)
    f = series.frame
    history = f.iloc[:24]
    future = f.iloc[24:48].drop(columns=["power", "k"], errors="ignore")
    with pytest.raises(RuntimeError, match="before fit"):
        model.predict(history, future)
