"""Cross-cutting utilities: reproducibility, device selection, config hashing.

These handle the Apple-Silicon / MPS gotchas documented in the thesis plan:
the macOS "Tahoe" bug where ``is_built()`` is True but ``is_available()`` is
False, and the requirement to fall back to CPU rather than crash.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from typing import Any

logger = logging.getLogger("pvfc")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if present) torch — including the MPS generator.

    Note: MPS and CPU float ops are *not* bit-identical, so for stochastic
    models (LSTM) we report seed variance over several runs rather than
    promising exact reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except Exception:  # torch missing or mps query failed — fine
        pass


def get_device(prefer: str = "mps") -> str:
    """Return a usable torch device string, defending against the MPS bugs.

    Always asserts ``torch.backends.mps.is_available()`` (not just
    ``is_built()``) and falls back to CPU. Also enables the MPS CPU-fallback
    for unsupported ops. Thesis-scale models run fine on CPU, so CPU is a
    correct (if slower) answer.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    try:
        import torch
    except ImportError:
        return "cpu"

    if prefer == "cpu":
        return "cpu"
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():  # the reliable check
            return "mps"
        if torch.backends.mps.is_built():
            logger.warning(
                "MPS is built but not available (macOS Tahoe bug?). Using CPU."
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("MPS availability query failed (%s). Using CPU.", exc)
    return "cpu"


def config_hash(config: dict[str, Any], length: int = 12) -> str:
    """SHA1 over canonical, sorted JSON of a config dict.

    This hash is the run_id, the results-row key, and the artifact directory
    name — the backbone of reproducibility (see plan §Tracking).
    """
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:length]


def env_fingerprint() -> dict[str, str]:
    """Record library versions for each run's metadata."""
    import platform

    fp = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "pandas", "sklearn", "lightgbm", "statsmodels",
                "statsforecast", "torch", "pvlib", "darts"):
        try:
            m = __import__(mod)
            fp[mod] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            fp[mod] = "missing"
    return fp
