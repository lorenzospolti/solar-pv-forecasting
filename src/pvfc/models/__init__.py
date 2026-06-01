"""Model registry. Each model is a Forecaster; the runner builds them by name."""
from __future__ import annotations

from .base import Forecaster
from .baselines import Persistence, SeasonalNaive, SmartPersistence

#: name -> factory. Heavier models (darts/foundation) are registered lazily in
#: build() to avoid importing torch/darts when only baselines are needed.
_BASE_REGISTRY: dict[str, type[Forecaster]] = {
    "persistence": Persistence,
    "smart_persistence": SmartPersistence,
    "seasonal_naive": SeasonalNaive,
}


def build(name: str, **kwargs) -> Forecaster:
    """Instantiate a forecaster by config name."""
    if name in _BASE_REGISTRY:
        return _BASE_REGISTRY[name](**kwargs)
    if name in {"sarima", "lightgbm", "lstm", "tft", "tide", "chronos2", "timesfm", "ttm", "moirai"}:
        from .darts_wrapper import build_darts_model

        return build_darts_model(name, **kwargs)
    raise KeyError(f"Unknown model '{name}'. Known: {sorted(_BASE_REGISTRY)} + darts/foundation models.")


__all__ = ["Forecaster", "Persistence", "SmartPersistence", "SeasonalNaive", "build"]
