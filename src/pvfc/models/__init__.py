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
    """Instantiate a forecaster by config name.

    Baselines come from the local registry; everything else is delegated to the
    darts wrapper, which resolves the architecture from the ``arch`` kwarg
    (default = ``name``) and validates it. This lets labels like ``chronos2_ft``
    (``arch: chronos2``) coexist with their zero-shot sibling.
    """
    if name in _BASE_REGISTRY:
        return _BASE_REGISTRY[name](**kwargs)
    from .darts_wrapper import build_darts_model

    return build_darts_model(name, **kwargs)


__all__ = ["Forecaster", "Persistence", "SmartPersistence", "SeasonalNaive", "build"]
