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
import warnings
from typing import Any

logger = logging.getLogger("pvfc")

#: Library warnings known to be harmless in this harness, as
#: (message regex, category, module regex). ``warnings.filterwarnings`` matches the
#: message from its start and the module it is attributed to (``stacklevel``), so each
#: filter hits one message from one place and every other warning still prints.
#: They repeat per prediction, per issued day or per fit, so overnight logs drown in them.
KNOWN_HARMLESS_WARNINGS: tuple[tuple[str, type[Warning], str], ...] = (
    # darts fits LightGBM on a DataFrame and predicts from a numpy array, so sklearn's
    # feature-name check warns on every (day x horizon x quantile) predict call:
    # 58,320 warnings, 12.7 MB of log, per 9-fold run
    ("X does not have valid feature names, but LGBMRegressor was fitted with feature names",
     UserWarning, r"sklearn\.utils\.validation\Z"),
    # torch disables pinned memory on MPS by itself; one warning per DataLoader (per day)
    ("'pin_memory' argument is set as true but not supported on MPS",
     UserWarning, r"torch\.utils\.data\.dataloader\Z"),
    # pytorch_lightning 2.5 still builds torch's deprecated LeafSpec; once per trainer call
    (r"`isinstance\(treespec, LeafSpec\)` is deprecated",
     FutureWarning, r"pytorch_lightning\.utilities\._pytree\Z"),
    # darts trains without a validation set: its data module hands lightning `[]` as the
    # validation loader. Only the `list` variant: the same check warns for an empty
    # training loader (`DataLoader`, `CombinedLoader`), and that must stay visible
    (r"Total length of `list` across ranks is zero\.",
     UserWarning, r"pytorch_lightning\.utilities\.data\Z"),
)


def quiet_known_warnings() -> None:
    """Silence exactly the ``KNOWN_HARMLESS_WARNINGS`` (idempotent).

    All four are emitted through the ``warnings`` module (lightning prints its own
    without the category name), so ``logging`` filters would not catch them.
    Deliberately NOT silenced: statsforecast's "possible convergence problem"
    (it flags the SARIMA folds whose AutoARIMA fit is worth a look).
    """
    for message, category, module in KNOWN_HARMLESS_WARNINGS:
        warnings.filterwarnings("ignore", message=message, category=category, module=module)


def setup_logging(level: int = logging.INFO) -> None:
    """Log format for every experiment script; also silences the known-harmless
    library warnings (``quiet_known_warnings``), which would otherwise bloat the logs."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    quiet_known_warnings()


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


def peak_rss_mb() -> float | None:
    """Peak resident-set size of this process in MB (None if unavailable).

    Note: ru_maxrss is a process-lifetime high-water mark, so per-row values are
    monotone within a run — read it as "RAM needed up to and including this
    model", and run heavy models in separate processes for isolated numbers.
    macOS reports bytes, Linux kilobytes.
    """
    try:
        import resource
        import sys

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(rss / (1024 ** 2 if sys.platform == "darwin" else 1024), 1)
    except Exception:  # noqa: BLE001 — e.g. Windows
        return None


def config_hash(config: dict[str, Any], length: int = 12) -> str:
    """SHA1 over canonical, sorted JSON of a config dict.

    This hash is the run_id, the results-row key, and the artifact directory
    name — the backbone of reproducibility (see plan §Tracking).
    """
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:length]


def env_fingerprint() -> dict[str, str]:
    """Record library versions for each run's metadata.

    Read from installed-distribution metadata, so nothing heavy (torch) is
    imported just to fingerprint, and packages without ``__version__`` (tirex)
    are still recorded.
    """
    import platform
    from importlib import metadata

    fp = {"python": platform.python_version(), "platform": platform.platform()}
    for key, dist in (("numpy", "numpy"), ("pandas", "pandas"), ("sklearn", "scikit-learn"),
                      ("lightgbm", "lightgbm"), ("statsmodels", "statsmodels"),
                      ("statsforecast", "statsforecast"), ("torch", "torch"),
                      ("pvlib", "pvlib"), ("darts", "darts"), ("tirex", "tirex-ts")):
        try:
            fp[key] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            fp[key] = "missing"
    return fp
