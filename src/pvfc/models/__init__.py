"""Model registry. Each model is a Forecaster; the runner builds them by name."""
from __future__ import annotations

from .base import Forecaster
from .baselines import CLIPER, Climatology, Persistence, SeasonalNaive, SmartPersistence

#: name -> factory. Heavier models (darts/foundation) are registered lazily in
#: build() to avoid importing torch/darts when only baselines are needed.
_BASE_REGISTRY: dict[str, type[Forecaster]] = {
    "persistence": Persistence,
    "smart_persistence": SmartPersistence,
    "seasonal_naive": SeasonalNaive,
    "climatology": Climatology,
    "cliper": CLIPER,
}


def build(name: str, **kwargs) -> Forecaster:
    """Instantiate a forecaster by config name.

    Baselines come from the local registry; everything else is delegated to a
    wrapper, which resolves the architecture from the ``arch`` kwarg (default
    = ``name``). The darts wrapper handles the natively-supported roster
    (Chronos-2, TimesFM-2.5, TiRex, plus SARIMA/LightGBM/LSTM); the external
    wrapper handles Moirai-2 (via ``uni2ts``) and TTM (via ``granite-tsfm``).
    Both wrappers respect the arch/label split so e.g. ``chronos2_ft``,
    ``moirai2_ft``, and ``ttm_ft`` coexist with their zero-shot siblings as
    separately-scored rows.
    """
    if name in _BASE_REGISTRY:
        return _BASE_REGISTRY[name](**kwargs)
    arch = kwargs.get("arch", name)
    from .external_foundation import EXTERNAL_FOUNDATION_ARCHES, build_external_foundation

    if arch in EXTERNAL_FOUNDATION_ARCHES:
        return build_external_foundation(name, **kwargs)
    from .darts_wrapper import build_darts_model

    return build_darts_model(name, **kwargs)


__all__ = ["Forecaster", "Persistence", "SmartPersistence", "SeasonalNaive", "Climatology",
           "CLIPER", "build"]
