"""The real tirex-2 package code path (TimeseriesType, ForecastModel.forecast, the quantile
buffer) through the TiRex-2 wrapper, with a tiny random-weight TiRex2: no checkpoint.

The MPS case catches the inductor regression the mocked tests cannot: tirex-2 compiles its
ResidualBlock, and on MPS that fails unless the wrapper forecasts eagerly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

tirex2 = pytest.importorskip("tirex2")
torch = pytest.importorskip("torch")

from pvfc.models import build  # noqa: E402
from pvfc.models.covariate_foundation import TiRex2Forecaster  # noqa: E402

QCOLS = [f"q{q / 10:.1f}" for q in range(1, 10)]


def _frame(days=12, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-03-01", periods=24 * days, freq="1h", tz="UTC")
    local = (idx.hour + 8) % 24
    day = (local >= 7) & (local <= 17)
    ghi = np.where(day, rng.uniform(50.0, 900.0, len(idx)), 0.0)
    k = np.where(day, 0.2 + 0.001 * ghi, 0.0)
    p_clear = np.where(day, 5.0, 0.0)
    return pd.DataFrame({"power": k * p_clear, "p_clear": p_clear, "k": k, "is_day": day,
                         "ghi": ghi, "temp_air": rng.normal(25, 3, len(idx)),
                         "wind_speed": rng.uniform(0, 8, len(idx)),
                         "rh": rng.uniform(40, 95, len(idx))}, index=idx)


def _template(kind):
    return {"time_mixer": {"model_type": kind, "embedding_dim": 128, "num_heads": 4,
                           "num_slstm_heads": 4},
            "variate_mixer": {"embedding_dim": 128, "num_heads": 4},
            "dropout": 0.0, "eps": 1e-7}


def _tiny_tirex2(device="cpu", seed=0):
    """A two-block TiRex2 with random weights (the ResidualBlock stays compiled, as in the
    package: the wrapper's eager stance is what is tested)."""
    torch.manual_seed(seed)
    model = tirex2.TiRex2(
        stack_config={"templates": {"m": _template("bi-mlstm"), "s": _template("bi-slstm")},
                      "recipe": ["m", "s"]},
        num_blocks=2, embedding_dim=128, input_patch_size=8, output_patch_size=8,
        quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        tokenizer_cfg={"output_patch_size": 8, "input_patch_size": 8, "input_patch_stride": 8},
        scaler_cfg={"use_arcsinh": True, "binaryaware": True}, h_expand=2, context_len=64,
        future_len=32, input_ff_dim=64, act_func="torch.nn.SiLU", dropout=0.0, tta_diff=True,
        stack_out_norm_config={"eps": 1e-7}, device=device)
    return model.eval()


def _check(pred, future):
    assert list(pred.columns) == ["y_hat", *QCOLS]
    assert len(pred) == 24 and np.isfinite(pred.to_numpy()).all()
    assert (pred.loc[future["p_clear"] == 0].to_numpy() == 0).all()  # night zeros
    assert (np.diff(pred[QCOLS].to_numpy(), axis=1) >= -1e-12).all()  # monotone after repair


@pytest.mark.parametrize("weather", [True, False])
def test_tirex2_wrapper_runs_the_real_package_code(monkeypatch, weather):
    fm = tirex2.ForecastModel(_tiny_tirex2())
    monkeypatch.setattr(TiRex2Forecaster, "_load_model", lambda self: fm)
    frame = _frame()
    if not weather:
        frame = frame.drop(columns=["ghi", "temp_air", "wind_speed", "rh"])
    history, future = frame.iloc[:-24], frame.iloc[-24:].drop(columns=["power", "k"])
    model = build("tirex2", input_chunk_length=24 * 10, quantile=True, device="cpu").fit(history)
    _check(model.predict(history, future), future)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_tirex2_forecasts_on_mps_and_agrees_with_cpu(monkeypatch):
    """Through TiRex2Forecaster._forecast on MPS (eager), the same weights as on CPU."""
    cpu = _tiny_tirex2("cpu")
    mps = _tiny_tirex2("mps")
    mps.load_state_dict(cpu.state_dict())
    frame = _frame()
    history, future = frame.iloc[:-24], frame.iloc[-24:].drop(columns=["power", "k"])
    preds = {}
    for device, net in (("cpu", cpu), ("mps", mps)):
        fm = tirex2.ForecastModel(net)
        monkeypatch.setattr(TiRex2Forecaster, "_load_model", lambda self, fm=fm: fm)
        model = build("tirex2", input_chunk_length=24 * 10, quantile=True,
                      device=device).fit(history)
        context, cov = model._inputs(history, future)
        raw = model._forecast(context, cov, len(future))
        assert sorted(raw) == [round(q / 10, 1) for q in range(1, 10)]
        assert all(v.shape == (24,) and np.isfinite(v).all() for v in raw.values())
        preds[device] = model.predict(history, future)
    _check(preds["mps"], future)
    np.testing.assert_allclose(preds["mps"].to_numpy(), preds["cpu"].to_numpy(), atol=1e-4)
