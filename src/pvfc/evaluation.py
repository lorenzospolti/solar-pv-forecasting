"""Evaluation: point, skill-score and probabilistic metrics.

All metrics operate on numpy arrays. Point errors are normalized by installed
AC capacity ``C`` (never by the mean, and never MAPE near night zeros — see
plan §Evaluation). The skill score is defined against a *reference* forecast
(smart persistence), following Yang et al.'s verification framework.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd

DEFAULT_QUANTILES: tuple[float, ...] = tuple(round(0.1 * i, 2) for i in range(1, 10))  # 0.1..0.9

#: Version of the scoring rules, stored with every results row. 1 = probabilistic
#: models scored on their median (y_hat); 2 = on the mean of their quantile grid.
SCORING_VERSION = 2


# --------------------------------------------------------------------------- #
# Point forecast
# --------------------------------------------------------------------------- #
def point_forecast(y_hat: np.ndarray, q_pred: dict[float, np.ndarray] | None = None) -> np.ndarray:
    """The point forecast the point metrics (RMSE, MAE, MBE, R², skill) score.

    Squared error, the headline metric, is minimised by the mean of the predictive
    distribution, not its median (Gneiting 2011), so scoring a probabilistic model
    on its median would handicap it against MSE-trained point models (LSTM). With
    the full quantile grid (tau = 0.1..0.9) the point forecast is therefore the
    average of the nine quantiles: an estimate of the mean that ignores the tails
    beyond the 10th and 90th percentiles. Wherever a quantile is missing or
    non-finite, and for deterministic models, it is the model's own ``y_hat``.
    """
    y_hat = np.asarray(y_hat, dtype=float)
    if not q_pred or not all(tau in q_pred for tau in DEFAULT_QUANTILES):
        return y_hat
    grid = np.stack([np.asarray(q_pred[tau], dtype=float) for tau in DEFAULT_QUANTILES])
    ok = np.isfinite(grid).all(axis=0)
    return np.where(ok, grid.mean(axis=0), y_hat)


def point_forecast_frame(pred: pd.DataFrame) -> np.ndarray:
    """``point_forecast`` for a predictions frame as the runner saves it (``y_hat``
    plus optional ``q0.1`` .. ``q0.9`` columns). The one definition shared by the
    runner, ``experiments/rescore.py`` and ``experiments/report.py``."""
    q_pred = {float(c[1:]): pred[c].to_numpy(dtype=float)
              for c in pred.columns if c.startswith("q") and c[1:].replace(".", "", 1).isdigit()}
    return point_forecast(pred["y_hat"].to_numpy(dtype=float), q_pred or None)


# --------------------------------------------------------------------------- #
# Point metrics
# --------------------------------------------------------------------------- #
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mbe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean bias error (positive = over-forecast)."""
    return float(np.mean(y_pred - y_true))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def normalize(value: float, capacity: float) -> float:
    return value / capacity if capacity else float("nan")


def skill_score(rmse_model: float, rmse_ref: float) -> float:
    """s = 1 - RMSE_model / RMSE_ref. s>0 beats the reference; s<0 is worse."""
    return float(1.0 - rmse_model / rmse_ref) if rmse_ref else float("nan")


# --------------------------------------------------------------------------- #
# Probabilistic metrics
# --------------------------------------------------------------------------- #
def pinball_loss(y_true: np.ndarray, q_pred: dict[float, np.ndarray]) -> float:
    """Mean pinball (quantile) loss averaged over the supplied quantile grid."""
    losses = []
    for tau, pred in q_pred.items():
        diff = y_true - pred
        losses.append(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))
    return float(np.mean(losses))


def crps_from_quantiles(y_true: np.ndarray, q_pred: dict[float, np.ndarray]) -> float:
    """CRPS approximated as 2 * mean pinball over an even quantile grid.

    For an even, dense grid the average pinball loss times 2 is a standard
    discrete CRPS estimator (it reduces to MAE for a point/median forecast).
    """
    return 2.0 * pinball_loss(y_true, q_pred)


def picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Prediction-interval coverage probability: fraction inside [lower, upper]."""
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def mean_interval_width(lower: np.ndarray, upper: np.ndarray, capacity: float = 1.0) -> float:
    """Sharpness: mean interval width (capacity-normalized)."""
    return float(np.mean(upper - lower) / capacity)


def reliability(y_true: np.ndarray, q_pred: dict[float, np.ndarray]) -> dict[float, float]:
    """Empirical coverage per nominal quantile (for a reliability diagram)."""
    return {tau: float(np.mean(y_true <= pred)) for tau, pred in q_pred.items()}


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
@dataclass
class Scores:
    """Tidy bundle of metrics for one (model, fold[, site]) run."""

    rmse: float
    mae: float
    mbe: float
    r2: float
    nrmse: float
    nmae: float
    skill_rmse: float | None = None
    pinball: float | None = None
    crps: float | None = None
    picp_80: float | None = None
    width_80: float | None = None
    extra: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        row = {k: v for k, v in self.__dict__.items() if k != "extra"}
        row.update(self.extra)
        return row


#: the metric columns every results row has (``Scores`` fields; quantile models add the
#: ``cov_q*`` reliability columns on top). Float in the store, NaN where not applicable.
METRIC_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(Scores) if f.name != "extra")


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    capacity: float,
    reference_pred: np.ndarray | None = None,
    q_pred: dict[float, np.ndarray] | None = None,
) -> Scores:
    """Compute the full metric bundle. ``q_pred`` triggers probabilistic scores."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rm = rmse(y_true, y_pred)
    scores = Scores(
        rmse=rm,
        mae=mae(y_true, y_pred),
        mbe=mbe(y_true, y_pred),
        r2=r2(y_true, y_pred),
        nrmse=normalize(rm, capacity),
        nmae=normalize(mae(y_true, y_pred), capacity),
    )
    if reference_pred is not None:
        scores.skill_rmse = skill_score(rm, rmse(y_true, np.asarray(reference_pred, dtype=float)))
    if q_pred:
        scores.pinball = pinball_loss(y_true, q_pred)
        scores.crps = normalize(crps_from_quantiles(y_true, q_pred), capacity)
        if 0.1 in q_pred and 0.9 in q_pred:
            scores.picp_80 = picp(y_true, q_pred[0.1], q_pred[0.9])
            scores.width_80 = mean_interval_width(q_pred[0.1], q_pred[0.9], capacity)
        # per-quantile empirical coverage -> tidy columns (cov_q0.1 .. cov_q0.9),
        # the data behind the thesis reliability diagram (see report.py)
        scores.extra.update({f"cov_q{tau}": cov
                             for tau, cov in reliability(y_true, q_pred).items()})
    return scores


def enforce_monotone_quantiles(q_pred: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    """Fix quantile crossing by sorting predictions across the quantile axis."""
    taus = sorted(q_pred)
    stacked = np.sort(np.stack([q_pred[t] for t in taus], axis=0), axis=0)
    return {t: stacked[i] for i, t in enumerate(taus)}
