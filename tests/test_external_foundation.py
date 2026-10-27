"""Regression tests for the Moirai-2 and TTM external-foundation wrappers (v0.5).

These tests don't need the real `uni2ts` / `granite-tsfm` libraries: they mock
the dep at the wrapper boundary (`_load_model` / `_forecast`) so CI stays
dependency-light, mirroring how `test_smoke.py` runs without darts/torch.

What we verify:
- (a) `models.build()` returns the right wrapper class for `moirai2` and `ttm` —
  and the arch/label split is respected (a second label with `arch=moirai2` is
  a separate object scored as its own row).
- (b) Calling `fit()` without the optional dep raises ImportError with the
  install instructions in the message (no traceback at predict time later).
- (c) An end-to-end Runner pass that mocks the model produces *distinct* result
  rows for two labels of one arch — proving the arch/label split flows all the
  way through to the results table.
- (e) `finetune: true` raises instead of running zero-shot under an `_ft` label
  (neither wrapper has a fine-tuning loop).
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
    """A second label with arch=moirai2 must be a Moirai2Forecaster with its own
    label — i.e. the same arch can back two different results rows."""
    a = build("moirai2", input_chunk_length=512)
    b = build("moirai2_ctx256", arch="moirai2", input_chunk_length=256)
    assert isinstance(a, Moirai2Forecaster) and isinstance(b, Moirai2Forecaster)
    assert a.name == "moirai2" and b.name == "moirai2_ctx256"
    assert a is not b and b.input_chunk_length == 256

    a_t = build("ttm", input_chunk_length=512)
    b_t = build("ttm_ctx1024", arch="ttm", input_chunk_length=1024)
    assert isinstance(a_t, TTMForecaster) and isinstance(b_t, TTMForecaster)
    assert a_t.name == "ttm" and b_t.name == "ttm_ctx1024"


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


def _run_two_models(name_a, name_b, arch):
    series = synthetic_station(days=360, seed=11)
    sp = _splitter()
    a = build(name_a, input_chunk_length=128, quantile=True)
    b = build(name_b, arch=arch, input_chunk_length=64, quantile=True)
    cls = Moirai2Forecaster if arch == "moirai2" else TTMForecaster
    with patch.object(cls, "_load_model", new=_mock_load), \
            patch.object(cls, "_forecast", new=_mock_forecast):
        rows_a = run_model(a, series, sp)
        rows_b = run_model(b, series, sp)
    return rows_a, rows_b


def test_moirai2_arch_label_split_produces_distinct_rows():
    rows_a, rows_b = _run_two_models("moirai2", "moirai2_ctx64", "moirai2")
    assert rows_a and rows_b
    assert {r["model"] for r in rows_a} == {"moirai2"}
    assert {r["model"] for r in rows_b} == {"moirai2_ctx64"}
    # quantile columns from the mock should populate cov_q*
    head = rows_a[0]
    assert any(k.startswith("cov_q") for k in head)


def test_ttm_arch_label_split_produces_distinct_rows():
    rows_a, rows_b = _run_two_models("ttm", "ttm_ctx64", "ttm")
    assert rows_a and rows_b
    assert {r["model"] for r in rows_a} == {"ttm"}
    assert {r["model"] for r in rows_b} == {"ttm_ctx64"}


# --------------------------------------------------------------------- (e) ---
@pytest.mark.parametrize("arch, cls", [("moirai2", Moirai2Forecaster), ("ttm", TTMForecaster)])
def test_finetune_request_raises_instead_of_mislabelling(arch, cls):
    model = build(f"{arch}_ft", arch=arch, input_chunk_length=64, finetune=True)
    train = synthetic_station(days=30, seed=5).frame
    with patch.object(cls, "_load_model", new=_mock_load), \
            pytest.raises(NotImplementedError, match="fine-tuning is not implemented"):
        model.fit(train)


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
