"""Evaluation: point, skill-score and probabilistic metrics.

All metrics operate on numpy arrays. Point errors are normalized by installed
AC capacity ``C`` (never by the mean, and never MAPE near night zeros — see
plan §Evaluation). The skill score is defined against a *reference* forecast
(smart persistence), following Yang et al.'s verification framework.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_QUANTILES: tuple[float, ...] = tuple(round(0.1 * i, 2) for i in range(1, 10))  # 0.1..0.9


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
    return scores


def enforce_monotone_quantiles(q_pred: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    """Fix quantile crossing by sorting predictions across the quantile axis."""
    taus = sorted(q_pred)
    stacked = np.sort(np.stack([q_pred[t] for t in taus], axis=0), axis=0)
    return {t: stacked[i] for i, t in enumerate(taus)}
