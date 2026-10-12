"""Thesis tables and figures for the Results chapter, computed from the saved predictions.

Every accuracy number is recomputed here from the per-timestamp predictions the
runner saves (``results/predictions/*.parquet``), not read from the per-fold rows
of ``results/results.parquet``: a pooled RMSE is not the mean of fold RMSEs, a
skill score is not the mean of fold skill scores, and the significance tests need
each day's error. The results store supplies only what predictions cannot: the
wall-clock time and peak memory of each fold.

Definitions (the Method chapter cites them; each function's docstring has its own):

* Scored hours. Daytime = ``is_day`` (solar zenith at the interval midpoint below
  85 deg) and finite truth: the headline. All hours = finite truth: robustness
  (appendix/all_hours).
  Every run of a station must cover the same scored hours with a finite forecast
  (``check_runs``); a run that does not is excluded, with a warning naming it.
* Point forecast. ``pvfc.evaluation.point_forecast_frame``: the mean of the nine
  quantiles for a probabilistic model (an estimate of the predictive mean, which
  minimises squared error; Gneiting 2011), ``y_hat`` for a deterministic one.
* Capacity C. The station nameplate in the ETL sidecar
  ``data/processed/<site>__<weather>__etl<ETL_VERSION>.cap.json``: the number the runner used.
* Pooled per-station metrics. Per model x weather setting x station, over the scored
  hours of all test windows. Headline = mean over stations.
* Headline reference. ``HEADLINE_REF`` (CLIPER): every main table and figure gives
  skill over it; setting it to ``"smart_persistence"`` switches them all. The
  appendix table ``master_skill_full`` gives both references.
* Intervals. 95 % percentile intervals from a moving-block bootstrap over the
  date-ordered local test days (``BLOCK_DAYS``-day blocks, ``BOOTSTRAP_RESAMPLES``
  resamples, ``BOOTSTRAP_SEED``): one matrix of day multiplicities shared by every
  station and model, so differences are paired (``Bootstrap``).
* No-covariate models. A model whose NWP and no-weather forecasts are identical at
  every station cannot use weather, so its forecast is a no-weather forecast. This
  is detected from the data (never from a list of names) and such a model is
  reported once, under "No weather", marked † in tables and "(no covariates)" in
  figures.
* Significance. The comparisons in ``COMPARISONS`` (fixed before panel v2 ran),
  tested on daily losses with Diebold-Mariano (Newey-West variance,
  Harvey-Leybourne-Newbold correction), Holm-adjusted over the family, checked by a
  moving-block bootstrap. The daily losses of all test windows form one date-ordered
  series, so the Newey-West lags and the bootstrap blocks also span the gaps between
  windows. The post hoc comparisons in ``POST_HOC`` get skill differences with
  intervals only, no tests.

Inputs. ``--predictions-dir`` and ``--results`` take one or more paths (default: the
panel and the extra runs); a missing path is skipped with a warning.

Outputs. ``--tables-dir`` gets each table as CSV (full-precision numbers) and as a
pandoc pipe table whose first line is its caption, the appendix tables in its
``appendix/`` folder; ``--figures-dir`` gets each figure as PDF and as a 200-dpi PNG.

  T1 master_skill         skill over the headline reference (95 % interval), nRMSE
  T2 chronos2_configs     the Chronos-2 configurations: skill, PICP-80, time
  T3 probabilistic        quantile score / C, PICP-80 (95 % interval), width-80 / C
  T4 compute              median fit / predict seconds, maximum peak RSS
  T5 significance         P1-P11 (skill difference, Holm p) and post hoc E1-E5
  appendix/master_skill_full    every model, skill over SP and CLIPER [min, max], nRMSE, nMBE
  appendix/significance_detail  the test statistics of P1-P11
  appendix/skill_by_fold        skill per test window
  appendix/skill_by_station     skill per station
  appendix/point_rule           skill under the mean and the median point rule
  appendix/p2_robustness        LightGBM - fine-tuned Chronos-2 per seed and window left out
  appendix/all_hours            skill vs SP on all hours (robustness; not in the thesis)
  appendix/panel_diagnostics    station sizes, scored days and hours, error correlation
  metrics_long.csv        every metric per model x setting x station x hour set
  F1 skill_by_model       F2 skill_vs_compute    F3 reliability
  F4 skill_by_fold        F5 skill_by_station    F6 weather_gain

Usage:
    python experiments/report.py
    python experiments/report.py --tables-dir /tmp/tables --figures-dir /tmp/figures
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from matplotlib.transforms import Bbox, blended_transform_factory  # noqa: E402
from scipy import stats  # noqa: E402

from pvfc.data import (  # noqa: E402
    ETL_VERSION,
    HKUST_TZ,
    PROCESSED_DIR,
    WEATHER_MODES,
    synthetic_station,
)
from pvfc.evaluation import (  # noqa: E402
    DEFAULT_QUANTILES,
    crps_from_quantiles,
    mean_interval_width,
    picp,
    point_forecast_frame,
    r2,
    reliability,
)
from pvfc.utils import setup_logging  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("pvfc.report")

# --------------------------------------------------------------------------- #
# Models, families, settings
# --------------------------------------------------------------------------- #
REFERENCE, TRAINED, FOUNDATION_ZS, FOUNDATION_FT, OTHER = (
    "reference", "trained", "foundation_zero_shot", "foundation_fine_tuned", "other")

#: (config name, display name, family) in thesis order. Unknown models go last,
#: under their raw name.
MODELS: tuple[tuple[str, str, str], ...] = (
    ("persistence", "Persistence", REFERENCE),
    ("seasonal_naive", "Seasonal naive", REFERENCE),
    ("smart_persistence", "Smart persistence", REFERENCE),
    ("climatology", "Climatology", REFERENCE),
    ("cliper", "CLIPER", REFERENCE),
    ("sarima", "SARIMA", TRAINED),
    ("lightgbm", "LightGBM", TRAINED),
    ("lstm", "LSTM", TRAINED),
    ("chronos2", "Chronos-S", FOUNDATION_ZS),
    ("chronos2_ctx1024", "Chronos-S-mid", FOUNDATION_ZS),
    ("chronos2_ctx2048", "Chronos-S-long", FOUNDATION_ZS),
    ("chronos2_base", "Chronos-B", FOUNDATION_ZS),
    ("chronos2_base_ctx2048", "Chronos-B-long", FOUNDATION_ZS),
    ("chronos2_ft", "Chronos-S-FT", FOUNDATION_FT),
    ("chronos2_ft_s1", "Chronos-S-FT, seed 1", FOUNDATION_FT),
    ("chronos2_ft_s2", "Chronos-S-FT, seed 2", FOUNDATION_FT),
    ("timesfm", "TimesFM", FOUNDATION_ZS),
    ("tirex", "TiRex", FOUNDATION_ZS),
    ("tirex_ctx2048", "TiRex-long", FOUNDATION_ZS),
)

#: Chronos-2 configuration -> (checkpoint, context in hours, training), for the
#: configuration table: the short names above carry no numbers, the table does.
CHRONOS2_SETUP: dict[str, tuple[str, int, str]] = {
    "chronos2": ("small", 336, "zero-shot"),
    "chronos2_ctx1024": ("small", 1024, "zero-shot"),
    "chronos2_ctx2048": ("small", 2048, "zero-shot"),
    "chronos2_base": ("base", 336, "zero-shot"),
    "chronos2_base_ctx2048": ("base", 2048, "zero-shot"),
    "chronos2_ft": ("small", 336, "fine-tuned"),
    "chronos2_ft_s1": ("small", 336, "fine-tuned, seed 1"),
    "chronos2_ft_s2": ("small", 336, "fine-tuned, seed 2"),
}
#: configurations run after the main run (panel v2) had been seen (configs/extras.yaml;
#: DECISIONS.md): every table and figure marks them with POST_HOC_MARK
POST_HOC_MODELS = frozenset({
    "chronos2_ctx1024", "chronos2_ctx2048", "chronos2_base", "chronos2_base_ctx2048",
    "chronos2_ft_s1", "chronos2_ft_s2", "tirex_ctx2048"})

#: the models of the main tables (T1, T3, T4) and figures, in this order
MAIN_MODELS: tuple[str, ...] = (
    "smart_persistence", "cliper", "sarima", "lightgbm", "lstm", "chronos2",
    "chronos2_base_ctx2048", "chronos2_ft", "timesfm", "tirex")
#: the Chronos-2 configurations of T2, in this order
CHRONOS2_MODELS: tuple[str, ...] = (
    "chronos2", "chronos2_ctx1024", "chronos2_ctx2048", "chronos2_base",
    "chronos2_base_ctx2048", "chronos2_ft", "chronos2_ft_s1", "chronos2_ft_s2")
#: the reference of every main table and figure ("smart_persistence" switches them all)
HEADLINE_REF = "cliper"

#: family -> (legend label, colour). The three hues are a colour-blind-safe set
#: (validated all-pairs); references wear a neutral gray and are always named.
FAMILIES: dict[str, tuple[str, str]] = {
    REFERENCE: ("Reference", "#9a9893"),
    TRAINED: ("Trained", "#2a78d6"),
    FOUNDATION_ZS: ("Foundation, zero-shot", "#eb6834"),
    FOUNDATION_FT: ("Foundation, fine-tuned", "#1baf7a"),
    OTHER: ("Other", "#52514e"),
}

#: result labels that differ from the config name
MODEL_ALIASES = {"seasonal_naive_168h": "seasonal_naive"}
#: weather settings in display order; unknown settings follow under their raw name
SETTINGS = {"nwp": "NWP", "none": "No weather"}
SP, CLIPER, CLIMATOLOGY = "smart_persistence", "cliper", "climatology"
#: references (besides SP, whose column is in every file) that skill is also measured against
EXTRA_REFERENCES = (CLIPER, CLIMATOLOGY)
NO_COV_MARK = "†"
NO_COV_LABEL = "(no covariates)"
POST_HOC_MARK = "‡"

#: The family of pairwise tests (T5 block 1), tied to the research questions. It was
#: chosen after an exploratory look at day-level tests on panel v1 (8 of those 10 pairs
#: kept, 3 added) and fixed on 2 Oct 2026, before panel v2 ran, on the same test days.
#: The Holm adjustment covers exactly the comparisons below that can run.
#: (id, (model A, setting A), (model B, setting B)); d = loss A - loss B. A side whose
#: model cannot use weather is tested where T1 reports it (``tested_side``).
COMPARISONS: tuple[tuple[str, tuple[str, str], tuple[str, str]], ...] = (
    ("P1", ("lightgbm", "nwp"), ("chronos2", "nwp")),
    ("P2", ("lightgbm", "nwp"), ("chronos2_ft", "nwp")),
    ("P3", ("chronos2_ft", "nwp"), ("sarima", "nwp")),
    ("P4", ("chronos2_ft", "nwp"), ("chronos2", "nwp")),
    ("P5", ("chronos2_ft", "none"), ("chronos2", "none")),
    ("P6", ("sarima", "none"), ("chronos2", "none")),
    ("P7", ("chronos2", "none"), ("tirex", "none")),
    ("P8", ("tirex", "none"), ("timesfm", "none")),
    ("P9", ("lightgbm", "nwp"), ("lightgbm", "none")),
    ("P10", ("chronos2", "nwp"), ("chronos2", "none")),
    ("P11", ("cliper", "nwp"), ("smart_persistence", "nwp")),
)
#: Post hoc comparisons (T5 block 2; DECISIONS.md, 4 Oct 2026): they involve the runs
#: added after panel v2, so they get a paired skill difference with its interval and
#: no test. Same layout as ``COMPARISONS``.
POST_HOC: tuple[tuple[str, tuple[str, str], tuple[str, str]], ...] = (
    ("E1", ("chronos2_ctx2048", "nwp"), ("chronos2", "nwp")),
    ("E2", ("lightgbm", "nwp"), ("chronos2_ctx2048", "nwp")),
    ("E3", ("chronos2_ft", "nwp"), ("chronos2_ctx2048", "nwp")),
    ("E4", ("lightgbm", "nwp"), ("chronos2_base_ctx2048", "nwp")),
    ("E5", ("chronos2_ft", "nwp"), ("chronos2_base_ctx2048", "nwp")),
)
#: the P2 pair whose robustness the appendix shows, and the fine-tuning seeds of its B side
P2_PAIR = ("lightgbm", "chronos2_ft")
FT_SEEDS = (("chronos2_ft", "seed 0"), ("chronos2_ft_s1", "seed 1"),
            ("chronos2_ft_s2", "seed 2"))
ALPHA = 0.05
INTERVAL = (2.5, 97.5)  # percentiles of the bootstrap distribution: a 95 % interval
DM_HORIZON = 1          # day-ahead: one daily loss per issue, so h = 1
BLOCK_DAYS = 7          # bootstrap block: one week keeps weather spells together
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0

QUANTILE_COLUMNS = tuple(f"q{tau}" for tau in DEFAULT_QUANTILES)
REQUIRED_COLUMNS = frozenset(
    {"y_true", "y_hat", "ref_hat", "is_day", "fold", "site", "model", "weather_mode"})
TIMING_COLUMNS = ("fit_s", "predict_s", "fit_predict_s", "peak_rss_mb")
HOURS = ("daytime", "all")

#: files earlier versions of this script wrote that no longer exist in the thesis
STALE_TABLES = ("master_results.csv", "master_results.md",
                "master_results_by_site.csv", "master_results_by_site.md",
                # moved to appendix/ on 4 Oct 2026
                "skill_by_station.csv", "skill_by_station.md",
                "skill_by_fold.csv", "skill_by_fold.md",
                "all_hours.csv", "all_hours.md",
                "panel_diagnostics.csv", "panel_diagnostics.md")
APPENDIX_DIR = "appendix"   # the appendix tables' folder inside --tables-dir
STALE_FIGURES = ("nrmse_by_model.png", "calibration_picp80.png")

EM_DASH = "—"
MINUS = "−"
NBSP = " "

# --------------------------------------------------------------------------- #
# Figure style
# --------------------------------------------------------------------------- #
INK, INK_2 = "#0b0b0b", "#52514e"        # primary / secondary text
GRID, AXIS = "#e6e5e0", "#c3c2b7"        # recessive grid / spines and ticks
#: F5's station dots and F3's diagonal: lighter than the reference family's gray, so
#: a station is never mistaken for a reference model's mean
STATION_DOT = "#c3c2b7"
MIDPOINT, POSITIVE, NEGATIVE = "#f0efec", "#2a78d6", "#e34948"
FULL_WIDTH, HALF_WIDTH = 6.3, 3.1        # inches: A4 text width at 2.5 cm margins, half
NO_WEATHER_ALPHA = 0.45
HATCH = "////"
MARKER = 6.0                              # marker size, points
SKILL_CMAP = LinearSegmentedColormap.from_list("skill", [NEGATIVE, MIDPOINT, POSITIVE])

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.titlecolor": INK,
    "axes.labelsize": 8.5,
    "axes.labelcolor": INK,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.facecolor": "white",
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.unicode_minus": True,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.5,
    "lines.markersize": MARKER,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "text.color": INK,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "hatch.linewidth": 0.6,
}

_MODEL_INFO = {name: (display, fam) for name, display, fam in MODELS}
_MODEL_RANK = {name: i for i, (name, _, _) in enumerate(MODELS)}

RunKey = tuple[str, str, str]  # (model, weather setting, site)


def display_name(model: str) -> str:
    return _MODEL_INFO.get(model, (model, OTHER))[0]


def family(model: str) -> str:
    return _MODEL_INFO.get(model, (model, OTHER))[1]


def colour(model: str) -> str:
    return FAMILIES[family(model)][1]


def model_order(models: Iterable[str]) -> list[str]:
    """Thesis order: references, trained, zero-shot, fine-tuned; unknown models last."""
    return sorted(set(models), key=lambda m: (_MODEL_RANK.get(m, len(_MODEL_RANK)), m))


def setting_label(setting: str) -> str:
    return SETTINGS.get(setting, setting)


def setting_order(settings: Iterable[str]) -> list[str]:
    rank = {s: i for i, s in enumerate(SETTINGS)}
    return sorted(set(settings), key=lambda s: (rank.get(s, len(rank)), s))


def skill_metric(reference: str | None = None) -> str:
    """The ``run_metrics`` column of skill over a reference (default ``HEADLINE_REF``)."""
    reference = HEADLINE_REF if reference is None else reference
    return "skill_sp" if reference == SP else f"skill_{reference}"


def reference_name(reference: str | None = None) -> str:
    """A reference as named inside a sentence: "CLIPER", "Smart persistence" (a model
    name, capitalised as in the chapters)."""
    reference = HEADLINE_REF if reference is None else reference
    return display_name(reference)


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
@dataclass
class Panel:
    """The predictions of every run, keyed (model, setting, site), and their context."""

    runs: dict[RunKey, pd.DataFrame]
    capacity: dict[str, float]
    timing: pd.DataFrame | None = None
    no_covariates: frozenset[str] = field(default_factory=frozenset)

    @cached_property
    def models(self) -> list[str]:
        return model_order(k[0] for k in self.runs)

    @cached_property
    def settings(self) -> list[str]:
        return setting_order(k[1] for k in self.runs)

    @cached_property
    def sites(self) -> list[str]:
        return sorted({k[2] for k in self.runs}, key=str.casefold)

    def has(self, model: str, setting: str) -> bool:
        return any((model, setting, site) in self.runs for site in self.sites)

    def reported_settings(self, model: str) -> list[str]:
        """Where a model's numbers appear: each setting it ran in, except that a
        no-covariate model appears once, under "none" (it uses no weather)."""
        if model in self.no_covariates:
            return ["none"]
        return [s for s in self.settings if self.has(model, s)]

    def is_quantile(self, model: str, setting: str) -> bool:
        frames = [f for (m, s, _), f in self.runs.items() if m == model and s == setting]
        return bool(frames) and all(has_quantiles(f) for f in frames)


def has_quantiles(frame: pd.DataFrame) -> bool:
    return all(c in frame.columns for c in QUANTILE_COLUMNS)


def _paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    """One path or several, as a list of distinct paths in the order given."""
    items = [paths] if isinstance(paths, (str, Path)) else list(paths)
    return list(dict.fromkeys(Path(p) for p in items))


def stored_hashes(results_paths: str | Path | Iterable[str | Path]) -> set[str]:
    """Config hashes of the rows in one or more results stores (a missing or unreadable
    store contributes none)."""
    hashes: set[str] = set()
    for path in _paths(results_paths):
        try:
            hashes |= set(pd.read_parquet(path, columns=["config_hash"])["config_hash"].dropna())
        except (OSError, KeyError, ValueError):
            continue
    return hashes


def load_predictions(predictions_dirs: str | Path | Iterable[str | Path],
                     include_synthetic: bool = False,
                     known_hashes: set[str] | None = None) -> dict[RunKey, pd.DataFrame]:
    """Read every run's predictions, from one or more folders, into
    ``{(model, setting, site): frame}``. A folder that does not exist is skipped with
    a warning, as is a file that cannot be read (e.g. one a running batch is writing).

    Each frame keeps the runner's columns plus ``y_point``, the point forecast the
    point metrics score (``point_forecast_frame``). Model labels are normalised
    (``MODEL_ALIASES``). When one run has several files (a config changed and the
    old file stayed, or a ``--max-folds`` smoke run left a partial one), the file
    covering the most test windows wins, then the one whose config hash is in the
    results store, then the newest; the others are logged and ignored.
    """
    dirs = _paths(predictions_dirs)
    files = []
    for folder in dirs:
        if not folder.is_dir():
            logger.warning("no predictions folder %s: skipped", folder)
            continue
        files += sorted(folder.glob("*.parquet"))
    where = ", ".join(str(d) for d in dirs)
    if not files:
        raise SystemExit(f"No predictions in {where}. Run experiments/run_panel.py.")
    found: dict[RunKey, list[tuple[Path, pd.DataFrame]]] = {}
    n_synthetic = 0
    for path in files:
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            logger.warning("skipping %s: unreadable (%s)", path.name, exc)
            continue
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing or frame.empty:
            logger.warning("skipping %s: %s", path.name,
                           f"missing columns {sorted(missing)}" if missing else "no rows")
            continue
        site = str(frame["site"].iloc[0])
        if site == "synthetic" and not include_synthetic:
            n_synthetic += 1
            continue
        raw_model = str(frame["model"].iloc[0])
        model = MODEL_ALIASES.get(raw_model, raw_model)
        weather = frame["weather_mode"].iloc[0]
        setting = str(weather) if pd.notna(weather) else "na"
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise SystemExit(f"{path.name}: index is not a DatetimeIndex")
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        frame = frame.sort_index()
        frame["model"] = model
        frame["y_point"] = point_forecast_frame(frame)
        found.setdefault((model, setting, site), []).append((path, frame))
    if n_synthetic:
        logger.info("excluded %d synthetic prediction files (--include-synthetic keeps them)",
                    n_synthetic)
    runs: dict[RunKey, pd.DataFrame] = {}
    for key, candidates in found.items():
        if len(candidates) > 1:
            def rank(item: tuple[Path, pd.DataFrame]) -> tuple[int, int, bool, float]:
                path, frame = item
                chash = frame["config_hash"].iloc[0] if "config_hash" in frame.columns else None
                return (frame["fold"].nunique(), len(frame), chash in (known_hashes or set()),
                        path.stat().st_mtime)
            candidates = sorted(candidates, key=rank, reverse=True)
            names = [f"{p.name} (test windows: {f['fold'].nunique()})" for p, f in candidates]
            logger.warning("%s: %d prediction files; using %s, ignoring %s", "/".join(key),
                           len(candidates), names[0], ", ".join(names[1:]))
        runs[key] = candidates[0][1]
    if not runs:
        raise SystemExit(f"No usable predictions in {where}.")
    return runs


def capacity_sidecar(site: str, setting: str, processed_dir: Path = PROCESSED_DIR) -> Path:
    """The capacity file ``pvfc.data.load_station`` writes next to its ETL cache."""
    cache = Path(processed_dir) / f"{site}__{setting}__etl{ETL_VERSION}.parquet"
    return cache.with_suffix(".cap.json")


def load_capacities(runs: dict[RunKey, pd.DataFrame],
                    processed_dir: Path = PROCESSED_DIR) -> dict[str, float]:
    """Nameplate capacity C (kW) per station, from the ETL sidecars: the denominator
    the runner normalised with. Never inferred from the metrics."""
    capacities: dict[str, float] = {}
    for site in sorted({k[2] for k in runs}):
        own = setting_order(k[1] for k in runs if k[2] == site)
        values: dict[str, float] = {}
        for setting in own + [w for w in WEATHER_MODES if w not in own]:
            path = capacity_sidecar(site, setting, processed_dir)
            if path.exists():
                values[setting] = float(json.loads(path.read_text())["capacity"])
        if not values and site == "synthetic":
            values["synthetic"] = float(
                inspect.signature(synthetic_station).parameters["capacity"].default)
        if not values:
            raise SystemExit(
                f"No capacity sidecar for {site!r} in {processed_dir} (expected e.g. "
                f"{capacity_sidecar(site, own[0], processed_dir).name}); build the ETL cache "
                "with experiments/check_data.py.")
        if len(set(values.values())) > 1:
            logger.warning("%s: capacity differs between settings %s", site, values)
        capacities[site] = next(iter(values.values()))
    return capacities


def load_timing(results_paths: str | Path | Iterable[str | Path],
                runs: dict[RunKey, pd.DataFrame],
                include_synthetic: bool = False) -> pd.DataFrame | None:
    """Per-fold timing and memory rows of the runs in use (one or more results stores,
    ``daytime_only`` rows so each fold counts once, matched on config hash). A store
    that is missing or unreadable is skipped with a warning."""
    frames = []
    for path in _paths(results_paths):
        if not path.exists():
            logger.warning("no results store at %s: skipped", path)
            continue
        try:
            frames.append(pd.read_parquet(path))
        except (OSError, ValueError) as exc:
            logger.warning("results store %s unreadable (%s): skipped", path, exc)
    if not frames:
        logger.warning("no results store: the compute table is empty and F2 is skipped")
        return None
    df = pd.concat(frames, ignore_index=True)
    if "daytime_only" in df.columns:
        df = df[df["daytime_only"].astype(bool)]
    if not include_synthetic:
        df = df[df["site"] != "synthetic"]
    weather = (df["weather_mode"] if "weather_mode" in df.columns
               else pd.Series(None, index=df.index))
    df = df.assign(model=df["model"].replace(MODEL_ALIASES),
                   weather_mode=weather.fillna("na").astype(str))
    run_hash = {key: (f["config_hash"].iloc[0] if "config_hash" in f.columns else None)
                for key, f in runs.items()}
    hashes = df["config_hash"] if "config_hash" in df.columns else pd.Series(None, index=df.index)
    keep = [key in run_hash and (run_hash[key] is None or h is None or h == run_hash[key])
            for key, h in zip(zip(df["model"], df["weather_mode"], df["site"]), hashes)]
    columns = ["model", "weather_mode", "site", "fold"] + [c for c in TIMING_COLUMNS
                                                            if c in df.columns]
    return df.loc[keep, columns].reset_index(drop=True)


def _same_forecast(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    quantiles = {c for c in a.columns if c in QUANTILE_COLUMNS}
    if quantiles != {c for c in b.columns if c in QUANTILE_COLUMNS} \
            or not a.index.equals(b.index):
        return False
    return all(np.array_equal(a[c].to_numpy(float), b[c].to_numpy(float), equal_nan=True)
               for c in ["y_hat", *sorted(quantiles)])


def detect_no_covariates(runs: dict[RunKey, pd.DataFrame]) -> frozenset[str]:
    """Models whose ``nwp`` and ``none`` forecasts are identical (y_hat and every
    quantile, ``np.array_equal``) at every station they ran at: they use no
    covariates, so the weather setting cannot change them."""
    found = set()
    for model in {k[0] for k in runs}:
        sites = {k[2] for k in runs if k[0] == model}
        pairs = [(runs.get((model, "nwp", s)), runs.get((model, "none", s))) for s in sites]
        if all(a is not None and b is not None and _same_forecast(a, b) for a, b in pairs):
            found.add(model)
    return frozenset(found)


def _scored_hours(frame: pd.DataFrame) -> tuple[bytes, bytes]:
    """A run's scored hours (finite truth) as hashable (UTC ns, fold) arrays."""
    mask = scored_mask(frame, "all")
    return (frame.index[mask].asi8.tobytes(),
            frame["fold"].to_numpy()[mask].astype(np.int64).tobytes())


def check_runs(runs: dict[RunKey, pd.DataFrame]) -> dict[RunKey, pd.DataFrame]:
    """Exclude the runs that would not be scored on the same hours as the rest.

    Pooled metrics compare models only if every run of a station covers the same
    scored hours (the (time, fold) pairs with finite truth) with a finite forecast on
    each: a ``--max-folds`` or interrupted run would pool fewer test windows, and one
    missing forecast would leave the station's RMSE undefined. The reference is the
    scored-hour set most of the station's runs share (the largest on a tie). A run
    that differs from it, or has a non-finite point forecast, quantile or SP forecast
    on a scored hour, is excluded with a WARNING naming it; the captions then name
    the means that cover fewer stations.
    """
    def pairs(h: tuple[bytes, bytes]) -> set[tuple[int, int]]:
        return set(zip(np.frombuffer(h[0], np.int64).tolist(),
                       np.frombuffer(h[1], np.int64).tolist()))

    hours = {key: _scored_hours(frame) for key, frame in runs.items()}
    references = {}
    for site in {k[2] for k in runs}:
        counts = Counter(h for k, h in hours.items() if k[2] == site)
        references[site] = max(counts, key=lambda h: (counts[h], len(h[0])))
    kept: dict[RunKey, pd.DataFrame] = {}
    for key, frame in runs.items():   # input order: the same sums as without the check
        problems, reference = [], references[key[2]]
        if hours[key] != reference:
            own, ref = pairs(hours[key]), pairs(reference)
            windows = {f for _, f in own & ref}
            problems.append(
                f"scores {len(own & ref)} of the station's {len(ref)} scored hours (in "
                f"{len(windows)} of {len({f for _, f in ref})} test windows)"
                + (f" and {len(own - ref)} others" if own - ref else ""))
        columns = ["y_point", "ref_hat", *(c for c in QUANTILE_COLUMNS if c in frame.columns)]
        values = frame[columns].to_numpy(float)[scored_mask(frame, "all")]
        n_bad = int((~np.isfinite(values)).any(axis=1).sum())
        if n_bad:
            problems.append(f"non-finite forecast on scored hours: {n_bad}")
        if problems:
            logger.warning("excluded %s: %s", "/".join(key), "; ".join(problems))
        else:
            kept[key] = frame
    if not kept:
        raise SystemExit("No run passed the coverage check (see the warnings above).")
    return kept


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def scored_mask(frame: pd.DataFrame, hours: str = "daytime") -> np.ndarray:
    """Scored hours: daytime = ``is_day`` & finite truth (headline); all = finite truth."""
    finite = np.isfinite(frame["y_true"].to_numpy(float))
    if hours == "all":
        return finite
    if hours != "daytime":
        raise ValueError(f"hours must be one of {HOURS}, got {hours!r}")
    return finite & frame["is_day"].to_numpy(bool)


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else math.nan


def _rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err ** 2))) if err.size else math.nan


def _skill(score: float, reference: float) -> float:
    return 1.0 - score / reference if reference and np.isfinite(reference) else math.nan


def run_metrics(frame: pd.DataFrame, capacity: float, hours: str = "daytime",
                references: dict[str, pd.Series] | None = None) -> dict[str, float]:
    """Pooled metrics of one run (model x setting x station) over its scored hours.

    Point metrics score ``y_point``; errors are forecast - truth (MBE > 0 is an
    over-forecast) and the n-prefixed ones are divided by the capacity C. Skill vs
    a reference R is 1 - RMSE_model / RMSE_R on the same hours: R = smart
    persistence is the ``ref_hat`` column of the same file; R = CLIPER or
    climatology is that model's point forecast for the station, joined on time (on
    the scored hours where it exists). CRPS ~= 2 x mean pinball loss over the nine
    quantiles; a deterministic forecast's CRPS is its MAE (``quantile`` = False
    flags it). CRPS skill vs SP = 1 - CRPS / MAE_SP. PICP-80 = share of truths in
    [q0.1, q0.9]; width-80 is divided by C; ``cov_q<tau>`` = observed share of
    truths <= q_tau.
    """
    mask = scored_mask(frame, hours)
    y = frame["y_true"].to_numpy(float)[mask]
    y_point = frame["y_point"].to_numpy(float)[mask]
    err = y_point - y
    err_sp = frame["ref_hat"].to_numpy(float)[mask] - y
    rmse, mae, mbe = _rmse(err), _mean(np.abs(err)), _mean(err)
    mae_sp = _mean(np.abs(err_sp))
    out: dict[str, float] = {
        "n_hours": int(mask.sum()),
        "rmse": rmse, "mae": mae, "mbe": mbe,
        "nrmse": rmse / capacity, "nmae": mae / capacity, "nmbe": mbe / capacity,
        "r2": r2(y, y_point) if y.size else math.nan,
        "rmse_sp": _rmse(err_sp),
        "skill_sp": _skill(rmse, _rmse(err_sp)),
    }
    for name, series in (references or {}).items():
        ref = series.reindex(frame.index).to_numpy(float)[mask]
        ok = np.isfinite(ref)
        out[f"skill_{name}"] = _skill(_rmse(err[ok]), _rmse(ref[ok] - y[ok]))
    quantile = has_quantiles(frame)
    if quantile:
        q = {tau: frame[col].to_numpy(float)[mask]
             for tau, col in zip(DEFAULT_QUANTILES, QUANTILE_COLUMNS)}
        crps = crps_from_quantiles(y, q)
        out["picp_80"] = picp(y, q[0.1], q[0.9])
        out["nwidth_80"] = mean_interval_width(q[0.1], q[0.9], capacity)
        out.update({f"cov_q{tau}": cov for tau, cov in reliability(y, q).items()})
    else:
        crps = mae
    out.update(quantile=quantile, crps=crps, ncrps=crps / capacity,
               crps_skill_sp=_skill(crps, mae_sp))
    return out


def reference_forecast(panel: Panel, reference: str, site: str,
                       setting: str) -> pd.Series | None:
    """A reference model's point forecast for a station: the same setting's run if
    there is one, else any setting's (CLIPER and climatology ignore weather)."""
    for s in [setting] + [x for x in panel.settings if x != setting]:
        frame = panel.runs.get((reference, s, site))
        if frame is not None:
            return frame["y_point"]
    return None


def compute_station_metrics(panel: Panel) -> pd.DataFrame:
    """One row per model x setting x station x hour set, with every ``run_metrics``."""
    rows = []
    for (model, setting, site), frame in panel.runs.items():
        references = {}
        for name in EXTRA_REFERENCES:
            series = reference_forecast(panel, name, site, setting)
            if series is not None:
                references[name] = series
        for hours in HOURS:
            rows.append({"model": model, "setting": setting, "site": site, "hours": hours,
                         **run_metrics(frame, panel.capacity[site], hours, references)})
    return pd.DataFrame(rows)


class Summary:
    """Across-station statistics (mean, min, max) of the pooled per-station metrics."""

    KEYS = ["model", "setting", "hours"]

    def __init__(self, station: pd.DataFrame) -> None:
        self.station = station
        grouped = station.drop(columns=["site"]).groupby(self.KEYS)
        self._stats = {"mean": grouped.mean(numeric_only=True),
                       "min": grouped.min(numeric_only=True),
                       "max": grouped.max(numeric_only=True),
                       "count": grouped.count()}
        self._by_site = station.set_index(["model", "setting", "site", "hours"])

    def n_stations(self, model: str, setting: str, metric: str = "skill_sp",
                   hours: str = "daytime") -> int:
        """How many stations the mean of a metric covers (those with a finite value)."""
        value = self.stat(model, setting, metric, hours, "count")
        return int(value) if math.isfinite(value) else 0

    def stat(self, model: str, setting: str, metric: str, hours: str = "daytime",
             how: str = "mean") -> float:
        table = self._stats[how]
        if metric not in table.columns or (model, setting, hours) not in table.index:
            return math.nan
        value = table.at[(model, setting, hours), metric]
        return float(value) if pd.notna(value) else math.nan

    def triple(self, model: str, setting: str, metric: str,
               hours: str = "daytime") -> tuple[float, float, float]:
        return tuple(self.stat(model, setting, metric, hours, how)  # type: ignore[return-value]
                     for how in ("mean", "min", "max"))

    def at_site(self, model: str, setting: str, site: str, metric: str,
                hours: str = "daytime") -> float:
        key = (model, setting, site, hours)
        if metric not in self._by_site.columns or key not in self._by_site.index:
            return math.nan
        value = self._by_site.at[key, metric]
        return float(value) if pd.notna(value) else math.nan


def compute_fold_skill(panel: Panel) -> pd.DataFrame:
    """Skill per station-fold: 1 - RMSE_model / RMSE_R over that station's scored
    daytime hours in that test window, for R = SP (``skill_sp``) and each reference of
    ``EXTRA_REFERENCES`` with predictions (``skill_<name>``, on the hours where the
    reference has a forecast)."""
    metrics = ["skill_sp"] + [skill_metric(r) for r in EXTRA_REFERENCES]
    rows = []
    for (model, setting, site), frame in panel.runs.items():
        sub = frame.loc[scored_mask(frame), ["fold", "y_true", "y_point", "ref_hat"]]
        for name in EXTRA_REFERENCES:
            series = reference_forecast(panel, name, site, setting)
            sub[name] = math.nan if series is None else series.reindex(sub.index).to_numpy(float)
        for fold, g in sub.groupby("fold"):
            y = g["y_true"].to_numpy(float)
            err = g["y_point"].to_numpy(float) - y
            row = {"model": model, "setting": setting, "site": site, "fold": int(fold),
                   "skill_sp": _skill(_rmse(err), _rmse(g["ref_hat"].to_numpy(float) - y))}
            for name in EXTRA_REFERENCES:
                ref = g[name].to_numpy(float)
                ok = np.isfinite(ref)
                row[skill_metric(name)] = (_skill(_rmse(err[ok]), _rmse(ref[ok] - y[ok]))
                                           if ok.any() else math.nan)
            rows.append(row)
    return pd.DataFrame(rows, columns=["model", "setting", "site", "fold", *metrics])


def fold_labels(panel: Panel) -> dict[int, str]:
    """Each test window labelled by the local month holding its middle ("Jul 2022"):
    the month most of its days fall in, which its start month need not be (a window
    from 30 Jun to 29 Jul is a July window). By its start date when two windows share
    that month (windows shorter than a month)."""
    first: dict[int, pd.Timestamp] = {}
    last: dict[int, pd.Timestamp] = {}
    for frame in panel.runs.values():
        times = frame.index.to_series().groupby(frame["fold"].to_numpy())
        for fold, ts in times.min().items():
            first[int(fold)] = min(first.get(int(fold), ts), ts)
        for fold, ts in times.max().items():
            last[int(fold)] = max(last.get(int(fold), ts), ts)
    local = {f: (first[f].tz_convert(HKUST_TZ), last[f].tz_convert(HKUST_TZ))
             for f in sorted(first)}
    labels = {f: (a + (b - a) / 2).strftime("%b %Y") for f, (a, b) in local.items()}
    if len(set(labels.values())) < len(labels):
        labels = {f: f"{a.day} {a:%b %Y}" for f, (a, _) in local.items()}
    return labels


def window_days(panel: Panel) -> int:
    """Length of a test window in local days (the longest window of any run)."""
    days = 0
    for frame in panel.runs.values():
        local = frame.index.tz_convert(HKUST_TZ).tz_localize(None).normalize()
        days = max(days, int(pd.Series(local).groupby(frame["fold"].to_numpy()).nunique().max()))
    return days


# --------------------------------------------------------------------------- #
# Significance
# --------------------------------------------------------------------------- #
def station_day_losses(frame: pd.DataFrame, capacity: float) -> pd.Series:
    """Daily loss of one run: for each local (Asia/Hong_Kong) calendar day, the mean
    of ((y_point - y_true) / C)^2 over that day's scored daytime hours."""
    mask = scored_mask(frame)
    err = (frame["y_point"].to_numpy(float) - frame["y_true"].to_numpy(float))[mask] / capacity
    days = frame.index[mask].tz_convert(HKUST_TZ).tz_localize(None).normalize()
    return pd.Series(err ** 2, index=days).groupby(level=0).mean().rename_axis("date")


def paired_daily_losses(panel: Panel, a: tuple[str, str], b: tuple[str, str],
                        cache: dict[RunKey, pd.Series] | None = None) -> pd.DataFrame:
    """Daily losses of two models, ``loss_a`` and ``loss_b``, on the days both have.

    Each day's loss is the mean over the stations available that day for both
    models (equal weight per station) of the station-day losses, so the two
    columns always average the same stations.
    """
    cache = {} if cache is None else cache

    def losses(key: RunKey) -> pd.Series:
        if key not in cache:
            cache[key] = station_day_losses(panel.runs[key], panel.capacity[key[2]])
        return cache[key]

    parts = []
    for site in panel.sites:
        ka, kb = (a[0], a[1], site), (b[0], b[1], site)
        if ka in panel.runs and kb in panel.runs:
            parts.append(pd.concat({"loss_a": losses(ka), "loss_b": losses(kb)},
                                   axis=1, join="inner"))
    if not parts:
        return pd.DataFrame(columns=["loss_a", "loss_b"])
    return pd.concat(parts).groupby(level=0).mean().sort_index()


def _floor_cbrt(n: int) -> int:
    """floor(n ** (1/3)) exactly (the float power rounds 64 ** (1/3) down to 3)."""
    root = round(n ** (1.0 / 3.0))
    while root ** 3 > n:
        root -= 1
    while (root + 1) ** 3 <= n:
        root += 1
    return root


def newey_west_variance(d: Sequence[float] | np.ndarray, lags: int | None = None) -> float:
    """Newey-West (HAC) long-run variance of ``d``: gamma_0 + 2 sum_l w_l gamma_l,
    Bartlett weights w_l = 1 - l / (L + 1), L = floor(n^(1/3)) by default, and
    autocovariances gamma_l = sum_t (d_t - mean)(d_{t-l} - mean) / n. A lag counts
    positions in ``d``: on the daily losses of several test windows it also pairs the
    last days of one window with the first days of the next."""
    d = np.asarray(d, dtype=float)
    n = d.size
    if n == 0:
        return math.nan
    n_lags = _floor_cbrt(n) if lags is None else int(lags)
    e = d - d.mean()
    variance = float(e @ e) / n
    for lag in range(1, min(n_lags, n - 1) + 1):
        variance += 2.0 * (1.0 - lag / (n_lags + 1.0)) * float(e[lag:] @ e[:-lag]) / n
    return variance


@dataclass(frozen=True)
class DMResult:
    statistic: float  # HLN-corrected; < 0 means A has the lower loss
    p_value: float    # two-sided, Student's t with n - 1 df
    variance: float   # Newey-West long-run variance of d
    n: int


def diebold_mariano(loss_a: Sequence[float] | np.ndarray, loss_b: Sequence[float] | np.ndarray,
                    h: int = DM_HORIZON) -> DMResult:
    """Diebold-Mariano test of equal expected loss on d = loss_a - loss_b.

    DM = mean(d) / sqrt(V / n) with V the Newey-West long-run variance of d, times
    the Harvey-Leybourne-Newbold small-sample factor sqrt((n + 1 - 2h + h(h-1)/n) / n);
    the p-value is two-sided from Student's t with n - 1 degrees of freedom. A zero
    variance means d is constant: statistic 0 and p = 1 if it is 0 everywhere,
    otherwise an infinite statistic and p = 0.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = d.size
    if n < 2:
        return DMResult(math.nan, math.nan, math.nan, n)
    mean = float(d.mean())
    variance = newey_west_variance(d)
    if not variance > 0:
        if mean == 0:
            return DMResult(0.0, 1.0, max(variance, 0.0), n)
        return DMResult(math.copysign(math.inf, mean), 0.0, max(variance, 0.0), n)
    hln = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    statistic = mean / math.sqrt(variance / n) * hln
    return DMResult(statistic, float(2.0 * stats.t.sf(abs(statistic), df=n - 1)), variance, n)


def block_indices(n: int, block: int = BLOCK_DAYS, n_boot: int = BOOTSTRAP_RESAMPLES,
                  seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """The moving-block resamples of a series of length ``n``: an (n_boot, n) array of
    positions. Each resample joins ceil(n / block) blocks of ``block`` consecutive
    positions, their starts drawn uniformly (``numpy.random.default_rng(seed)``),
    and is cut to length n. The one scheme of the bootstrap p-value and the intervals."""
    size = min(block, n)
    n_blocks = math.ceil(n / size)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - size + 1, size=(n_boot, n_blocks))
    return (starts[:, :, None] + np.arange(size)).reshape(n_boot, n_blocks * size)[:, :n]


def day_weights(n_days: int, block: int = BLOCK_DAYS, n_boot: int = BOOTSTRAP_RESAMPLES,
                seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """W: (n_boot, n_days) integer multiplicities, W[b, j] = how often resample b draws
    day j (``block_indices``); every row sums to n_days. A resampled pooled statistic
    is then a matrix product: (W @ per-day sums) / (W @ per-day counts)."""
    index = block_indices(n_days, block, n_boot, seed)
    offsets = (np.arange(n_boot) * n_days)[:, None]
    return np.bincount((index + offsets).ravel(),
                       minlength=n_boot * n_days).reshape(n_boot, n_days)


def block_bootstrap_p(d: Sequence[float] | np.ndarray, block: int = BLOCK_DAYS,
                      n_boot: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> float:
    """Moving-block bootstrap p-value of mean(d) = 0, the check on the DM test.

    Resamples overlapping blocks of ``block`` consecutive entries of the date-ordered
    series (``numpy.random.default_rng(seed)``) into ``n_boot`` series of the
    original length; a block that straddles the gap between two test windows joins
    days about a month apart. Two-sided, null-centred:
    p = (1 + #{|mean(d*) - mean(d)| >= |mean(d)|}) / (n_boot + 1).
    """
    d = np.asarray(d, dtype=float)
    n = d.size
    if n == 0:
        return math.nan
    means = d[block_indices(n, block, n_boot, seed)].mean(axis=1)
    observed = float(d.mean())
    exceed = int(np.count_nonzero(np.abs(means - observed) >= abs(observed)))
    return (1 + exceed) / (n_boot + 1)


def holm(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (monotone step-down): sorted ascending,
    p_(i) -> max_{j <= i} min(1, (m - j + 1) p_(j)), returned in input order."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out = np.empty(m)
    out[order] = adjusted
    return out


def tested_side(panel: Panel, model: str, setting: str) -> tuple[str, str]:
    """The runs one side of a comparison is tested on. A no-covariate model forecasts
    the same in both settings, so it is tested on its No-weather runs, where T1 reports
    it; CLIPER and climatology ignore weather by construction, so when they lack the
    named setting their other setting stands in (No weather first). Any other model
    keeps its setting: there the setting is part of the hypothesis."""
    if model in panel.no_covariates:
        return model, "none"
    if model in EXTRA_REFERENCES and not panel.has(model, setting):
        for other in ["none", *panel.settings]:
            if panel.has(model, other):
                return model, other
    return model, setting


def compute_significance(panel: Panel) -> pd.DataFrame:
    """Run the comparisons fixed before panel v2 (``COMPARISONS``) on daily losses.

    Per pair: days n, mean daily losses, relative difference mean(d) / mean(L_B),
    the Diebold-Mariano statistic and p-value, the Holm-adjusted p (over the pairs
    that could be run), the block-bootstrap p, and the verdict at Holm p < ALPHA.
    Each side runs on ``tested_side``'s setting, which the ``setting_*`` columns
    record. A pair with a model that has no predictions is kept as "not available".
    The daily losses are taken in date order across the test windows, so the
    Newey-West lags and the bootstrap blocks also pair days on either side of the
    gap between two windows (as specified; the caption says so).
    """
    cache: dict[RunKey, pd.Series] = {}
    rows = []
    for pid, a, b in COMPARISONS:
        a, b = tested_side(panel, *a), tested_side(panel, *b)
        row: dict = {"id": pid, "model_a": a[0], "setting_a": a[1],
                     "model_b": b[0], "setting_b": b[1]}
        missing = [f"{m}/{s}" for m, s in (a, b) if not panel.has(m, s)]
        daily = pd.DataFrame() if missing else paired_daily_losses(panel, a, b, cache)
        if missing or daily.empty:
            logger.info("%s not available: no predictions for %s", pid,
                        ", ".join(missing) or "common days")
            rows.append({**row, "available": False})
            continue
        d = (daily["loss_a"] - daily["loss_b"]).to_numpy()
        dm = diebold_mariano(daily["loss_a"], daily["loss_b"])
        rows.append({**row, "available": True, "n_days": len(daily),
                     "loss_a": daily["loss_a"].mean(), "loss_b": daily["loss_b"].mean(),
                     "rel_diff_pct": 100.0 * d.mean() / daily["loss_b"].mean(),
                     "dm_stat": dm.statistic, "p_dm": dm.p_value,
                     "p_bootstrap": block_bootstrap_p(d)})
    df = pd.DataFrame(rows)
    for col in ("n_days", "loss_a", "loss_b", "rel_diff_pct", "dm_stat", "p_dm",
                "p_bootstrap"):
        if col not in df.columns:
            df[col] = math.nan
    df["p_holm"] = math.nan
    tested = df["available"] & df["p_dm"].notna()
    df.loc[tested, "p_holm"] = holm(df.loc[tested, "p_dm"].to_numpy())

    def verdict(r: pd.Series) -> str:
        if not r["available"]:
            return "not available"
        if r["p_holm"] < ALPHA:
            return "A better" if r["rel_diff_pct"] < 0 else "B better"
        return "no significant difference"

    df["verdict"] = df.apply(verdict, axis=1)
    return df


# --------------------------------------------------------------------------- #
# Bootstrap intervals
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Estimate:
    """A statistic: its value on the test days (``point``) and on each bootstrap
    resample of them (``samples``, empty when it cannot be computed)."""

    point: float
    samples: np.ndarray

    @property
    def available(self) -> bool:
        return _finite(self.point) and self.samples.size > 0

    @property
    def interval(self) -> tuple[float, float]:
        """The 95 % percentile interval (``INTERVAL``) of the resampled values."""
        if not self.available:
            return math.nan, math.nan
        lo, hi = np.percentile(self.samples, INTERVAL)
        return float(lo), float(hi)

    def __sub__(self, other: "Estimate") -> "Estimate":
        """The paired difference: resample by resample (both on the same weights)."""
        if not (self.available and other.available) or self.samples.shape != other.samples.shape:
            return NOT_AVAILABLE
        return Estimate(self.point - other.point, self.samples - other.samples)


NOT_AVAILABLE = Estimate(math.nan, np.empty(0))


class Bootstrap:
    """Moving-block bootstrap over the date-ordered local test days, shared by every
    station and model.

    The days are the local (Asia/Hong_Kong) calendar days with a scored daytime hour
    at any station. One weight matrix W (``day_weights``: ``BLOCK_DAYS``-day blocks,
    ``BOOTSTRAP_RESAMPLES`` resamples, ``BOOTSTRAP_SEED``) gives each resample's day
    multiplicities; the same W serves every station and model, so the difference of
    two statistics, resample by resample, is paired. Per run (and reference) the
    per-day sums of squared error, scored hours and hours inside [q0.1, q0.9] are
    kept, so a resampled per-station pooled statistic is (W @ sums) / (W @ counts),
    e.g. MSE; skill per station is 1 - sqrt(MSE_model / MSE_reference) on the same
    hours, and the statistic is its mean over the stations. With all-ones weights
    these are exactly the point estimates of ``run_metrics`` and ``Summary``.
    """

    def __init__(self, panel: Panel, block: int = BLOCK_DAYS,
                 n_boot: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> None:
        self.panel, self.block, self.n_boot, self.seed = panel, block, n_boot, seed
        fold_of: dict[pd.Timestamp, int] = {}
        for frame in panel.runs.values():
            mask = scored_mask(frame)
            days = _local_days(frame.index[mask])
            for day, fold in zip(days, frame["fold"].to_numpy()[mask]):
                fold_of.setdefault(day, int(fold))
        #: the test days in date order, and the test window each belongs to
        self.days = pd.DatetimeIndex(sorted(fold_of))
        self.fold = np.array([fold_of[d] for d in self.days], dtype=int)
        self._weights: dict[int, np.ndarray] = {}
        self._sums: dict[tuple[RunKey, str | None], dict[str, np.ndarray] | None] = {}

    def weights(self, n_days: int) -> np.ndarray:
        """W for a series of ``n_days`` days (float, for the matrix products)."""
        if n_days not in self._weights:
            self._weights[n_days] = day_weights(n_days, self.block, self.n_boot,
                                                self.seed).astype(float)
        return self._weights[n_days]

    def _run_sums(self, key: RunKey, reference: str | None) -> dict[str, np.ndarray] | None:
        """Per-day sums of one run over its scored daytime hours. With a reference:
        hours ``n``, squared error of the point forecast ``sse`` (mean rule) and of
        ``y_hat`` ``sse_median`` (median rule), and of the reference ``sse_ref``, on
        the hours where the reference has a forecast. Without: hours ``n`` and hours
        inside [q0.1, q0.9] ``inside`` (quantile runs). None if there is no reference."""
        if (key, reference) in self._sums:
            return self._sums[(key, reference)]
        model, setting, site = key
        frame = self.panel.runs[key]
        mask = scored_mask(frame)
        y = frame["y_true"].to_numpy(float)[mask]
        pos = self.days.get_indexer(_local_days(frame.index[mask]))
        n_days = len(self.days)

        def per_day(weights: np.ndarray | None, ok: np.ndarray) -> np.ndarray:
            return np.bincount(pos[ok], weights=None if weights is None else weights[ok],
                               minlength=n_days).astype(float)

        out: dict[str, np.ndarray] | None
        if reference is None:
            everywhere = np.ones(y.size, dtype=bool)
            out = {"n": per_day(None, everywhere)}
            if has_quantiles(frame):
                lower, upper = (frame[c].to_numpy(float)[mask] for c in ("q0.1", "q0.9"))
                out["inside"] = per_day(((y >= lower) & (y <= upper)).astype(float), everywhere)
        else:
            if reference == SP:
                ref = frame["ref_hat"].to_numpy(float)[mask]
            else:
                series = reference_forecast(self.panel, reference, site, setting)
                ref = None if series is None else series.reindex(frame.index).to_numpy(float)[mask]
            if ref is None:
                out = None
            else:
                ok = np.isfinite(ref)
                out = {"n": per_day(None, ok),
                       "sse": per_day((frame["y_point"].to_numpy(float)[mask] - y) ** 2, ok),
                       "sse_median": per_day((frame["y_hat"].to_numpy(float)[mask] - y) ** 2, ok),
                       "sse_ref": per_day((ref - y) ** 2, ok)}
        self._sums[(key, reference)] = out
        return out

    def sites(self, model: str, setting: str, reference: str | None = None) -> list[str]:
        """The stations where the model ran in the setting and, unless ``reference`` is
        None, the reference has a forecast."""
        return [s for s in self.panel.sites if (model, setting, s) in self.panel.runs
                and self._run_sums((model, setting, s), reference) is not None]

    def _stack(self, model: str, setting: str, reference: str | None, sites: Sequence[str],
               column: str, days: np.ndarray | None) -> np.ndarray:
        """(days, stations) matrix of one per-day sum."""
        matrix = np.column_stack([self._run_sums((model, setting, s), reference)[column]
                                  for s in sites])
        return matrix if days is None else matrix[days]

    def skill(self, model: str, setting: str, reference: str | None = None,
              rule: str = "mean", sites: Sequence[str] | None = None,
              days: np.ndarray | None = None) -> Estimate:
        """Skill over ``reference`` (default ``HEADLINE_REF``), mean over ``sites``
        (default: every station with both runs), optionally on a subset of the days
        (``days``: a boolean mask over ``self.days``, resampled with its own W of the
        same seed). ``rule`` "median" scores ``y_hat`` (the median of a quantile
        model) instead of the point forecast."""
        reference = HEADLINE_REF if reference is None else reference
        sites = self.sites(model, setting, reference) if sites is None else list(sites)
        if not sites:
            return NOT_AVAILABLE
        error = "sse" if rule == "mean" else "sse_median"
        sse, sse_ref, n = (self._stack(model, setting, reference, sites, c, days)
                           for c in (error, "sse_ref", "n"))

        def resampled(w: np.ndarray) -> np.ndarray:
            with np.errstate(divide="ignore", invalid="ignore"):
                counts = w @ n
                mse, mse_ref = (w @ sse) / counts, (w @ sse_ref) / counts
                return (1.0 - np.sqrt(mse / mse_ref)).mean(axis=1)

        n_days = sse.shape[0]
        return Estimate(float(resampled(np.ones((1, n_days)))[0]),
                        resampled(self.weights(n_days)))

    def skill_difference(self, a: tuple[str, str], b: tuple[str, str],
                         reference: str | None = None, rule: str = "mean",
                         days: np.ndarray | None = None) -> Estimate:
        """Paired skill difference A - B, both over the stations they share."""
        reference = HEADLINE_REF if reference is None else reference
        common = [s for s in self.sites(*a, reference) if s in self.sites(*b, reference)]
        if not common:
            return NOT_AVAILABLE
        return (self.skill(*a, reference=reference, rule=rule, sites=common, days=days)
                - self.skill(*b, reference=reference, rule=rule, sites=common, days=days))

    def picp(self, model: str, setting: str) -> Estimate:
        """PICP-80 (share of scored hours inside [q0.1, q0.9]), mean over stations."""
        sites = [s for s in self.sites(model, setting)
                 if "inside" in self._run_sums((model, setting, s), None)]
        if not sites or not self.panel.is_quantile(model, setting):
            return NOT_AVAILABLE
        inside, n = (self._stack(model, setting, None, sites, c, None) for c in ("inside", "n"))

        def resampled(w: np.ndarray) -> np.ndarray:
            return ((w @ inside) / (w @ n)).mean(axis=1)

        return Estimate(float(resampled(np.ones((1, inside.shape[0])))[0]),
                        resampled(self.weights(inside.shape[0])))


def _local_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Each UTC timestamp's local (Asia/Hong_Kong) calendar day, as naive midnight."""
    return index.tz_convert(HKUST_TZ).tz_localize(None).normalize()


def complete(panel: Panel, model: str, setting: str) -> bool:
    """Whether a model has a (checked) run in the setting at every station."""
    return all((model, setting, s) in panel.runs for s in panel.sites)


def add_skill_differences(sig: pd.DataFrame, boot: Bootstrap) -> pd.DataFrame:
    """The comparisons' paired skill difference A - B over ``HEADLINE_REF``, on the
    tested sides' settings: ``skill_diff`` and its 95 % interval ``skill_diff_lo``,
    ``skill_diff_hi`` (NaN where not available)."""
    out = sig.copy()
    values = []
    for r in out.to_dict("records"):
        est = (boot.skill_difference((r["model_a"], r["setting_a"]),
                                     (r["model_b"], r["setting_b"]))
               if r["available"] else NOT_AVAILABLE)
        values.append((est.point, *est.interval))
    out[["skill_diff", "skill_diff_lo", "skill_diff_hi"]] = (
        np.array(values, dtype=float).reshape(len(values), 3))
    return out


def compute_post_hoc(panel: Panel, boot: Bootstrap) -> pd.DataFrame:
    """The post hoc comparisons (``POST_HOC``): paired skill difference A - B over
    ``HEADLINE_REF`` with its 95 % interval, no test. A comparison is "not available"
    unless both sides have runs at every station (an incomplete run is never compared
    on fewer stations)."""
    rows = []
    for pid, a, b in POST_HOC:
        a, b = tested_side(panel, *a), tested_side(panel, *b)
        row = {"id": pid, "model_a": a[0], "setting_a": a[1], "model_b": b[0],
               "setting_b": b[1]}
        est = (boot.skill_difference(a, b)
               if complete(panel, *a) and complete(panel, *b) else NOT_AVAILABLE)
        if not est.available:
            logger.warning("%s not available: %s/%s or %s/%s has no run at some station",
                           pid, *a, *b)
        rows.append({**row, "available": est.available, "skill_diff": est.point,
                     "skill_diff_lo": est.interval[0], "skill_diff_hi": est.interval[1]})
    return pd.DataFrame(rows)


@dataclass
class Diagnostics:
    table: pd.DataFrame   # one row per station
    corr: pd.DataFrame    # station x station correlation of hourly |SP error| / C
    panel_days: int       # local days with a scored daytime hour at any station


def compute_panel_diagnostics(panel: Panel) -> Diagnostics:
    """Per station: capacity, scored days (local days with a scored daytime hour),
    scored hours, mean |SP error| / C, and the Pearson correlation of hourly
    |SP error| / C with the other stations, each pair on its common scored daytime
    hours (SP = the ``ref_hat`` column, the same in every file of a station)."""
    errors, rows, all_days = {}, [], set()
    for site in panel.sites:
        keys = sorted(k for k in panel.runs if k[2] == site)
        key = next((k for k in keys if k[0] == SP), keys[0])
        frame, cap = panel.runs[key], panel.capacity[site]
        day = scored_mask(frame, "daytime")
        abs_err = (frame["ref_hat"] - frame["y_true"]).abs() / cap
        errors[site] = abs_err[day]
        days = set(frame.index[day].tz_convert(HKUST_TZ).tz_localize(None).normalize())
        all_days |= days
        rows.append({"site": site, "capacity_kw": cap, "days": len(days),
                     "daytime_hours": int(day.sum()),
                     "all_hours": int(scored_mask(frame, "all").sum()),
                     "mean_abs_sp_error_c": float(abs_err[day].mean())})
    corr = pd.DataFrame(errors).corr(method="pearson")
    table = pd.DataFrame(rows)
    n = len(corr)
    table["mean_corr_others"] = ((corr.sum(axis=1) - 1.0) / (n - 1)).to_numpy() if n > 1 \
        else math.nan
    return Diagnostics(table, corr, len(all_days))


def pairwise_correlations(corr: pd.DataFrame) -> np.ndarray:
    upper = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    return corr.to_numpy()[upper]


#: compute summary columns: medians over station-folds, ``window_s`` = fit_s + predict_s
#: of each station-fold, and the maximum peak RSS
COMPUTE_COLUMNS = (*TIMING_COLUMNS, "window_s", "max_peak_rss_mb", "n")


def compute_medians(panel: Panel) -> pd.DataFrame:
    """Per model x setting, over its station-folds (``n``): the median fit, predict and
    fit + predict seconds and peak RSS, the median of fit_s + predict_s per
    station-fold (``window_s``), and the maximum peak RSS (``max_peak_rss_mb``)."""
    cols = list(COMPUTE_COLUMNS)
    if panel.timing is None or panel.timing.empty:
        return pd.DataFrame(columns=cols)
    timing = panel.timing
    if {"fit_s", "predict_s"} <= set(timing.columns):
        timing = timing.assign(window_s=timing["fit_s"] + timing["predict_s"])
    grouped = timing.groupby(["model", "weather_mode"])
    medians = grouped[[c for c in (*TIMING_COLUMNS, "window_s")
                       if c in timing.columns]].median()
    if "peak_rss_mb" in timing.columns:
        medians["max_peak_rss_mb"] = grouped["peak_rss_mb"].max()
    medians["n"] = grouped.size()
    return medians.reindex(columns=cols)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
@dataclass
class Table:
    stem: str
    label: str
    caption: str
    display: pd.DataFrame
    numeric: pd.DataFrame
    appendix: bool = False   # written to <tables-dir>/appendix/


def _finite(x: object) -> bool:
    try:
        return x is not None and math.isfinite(float(x))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _num(x: object, decimals: int = 3) -> str:
    """Fixed decimals, a true minus sign, no "-0.000", an em dash when not applicable."""
    if not _finite(x):
        return EM_DASH
    text = f"{float(x):.{decimals}f}"  # type: ignore[arg-type]
    if text.startswith("-"):
        text = text[1:] if float(text) == 0 else MINUS + text[1:]
    return text


def _num_range(mean: float, lo: float, hi: float, decimals: int = 3) -> str:
    """The cell "mean [min, max]". A no-break space keeps "[min, max]" whole, so a
    narrow column breaks the cell only between the mean and its range."""
    if not _finite(mean):
        return EM_DASH
    return f"{_num(mean, decimals)} [{_num(lo, decimals)},{NBSP}{_num(hi, decimals)}]"


def _interval_cell(est: Estimate, decimals: int = 3) -> str:
    """The cell "point [lo, hi]" of a bootstrap estimate (its 95 % interval)."""
    return _num_range(est.point, *est.interval, decimals) if est.available else EM_DASH


def _p(p: object) -> str:
    if not _finite(p):
        return EM_DASH
    return "<0.001" if float(p) < 0.001 else f"{float(p):.3f}"  # type: ignore[arg-type]


def _int(x: object) -> str:
    return f"{round(float(x)):d}" if _finite(x) else EM_DASH  # type: ignore[arg-type]


def _post_hoc(model: str) -> str:
    """The name with the post hoc mark, if the configuration was added after the main run."""
    return display_name(model) + (f" {POST_HOC_MARK}" if model in POST_HOC_MODELS else "")


def _model_cell(panel: Panel, model: str) -> str:
    return _post_hoc(model) + (f" {NO_COV_MARK}" if model in panel.no_covariates else "")


def _identity(panel: Panel, model: str) -> dict:
    return {"model": model, "display_name": display_name(model), "family": family(model),
            "no_covariates": model in panel.no_covariates}


def _no_cov_note(panel: Panel, models: Iterable[str]) -> str:
    """The caption notes of the marks that ``models`` carry: † (no covariates) and ‡
    (post hoc)."""
    models = set(models)
    note = ""
    if panel.no_covariates & models:
        note += (f" {NO_COV_MARK} No covariates: the forecasts are identical in both weather "
                 "settings, so the model is reported once, under No weather.")
    if POST_HOC_MODELS & models:
        note += (f" {POST_HOC_MARK} Post hoc: added after the results of the main run had "
                 "been seen.")
    return note


def _station_note(panel: Panel, summary: Summary, models: Iterable[str]) -> str:
    """Names the means over stations that cover fewer than all of them (a run that is
    missing or failed ``check_runs``), so a caption never overstates a mean."""
    short = [f"{display_name(m)}, {setting_label(s)} ({n})"
             for m in models for s in panel.reported_settings(m)
             if 0 < (n := summary.n_stations(m, s, skill_metric())) < len(panel.sites)]
    if not short:
        return ""
    return (f" Means over fewer than the {len(panel.sites)} stations (number in brackets): "
            + "; ".join(short) + ".")


#: Glyph widths of the manuscript font (Latin Modern Roman) in digit widths, rounded:
#: a minus sign or "<" is about 1.55 digits wide, a "." 0.56, an "m" 1.67.
_GLYPH_WIDTH = {ch: width for chars, width in (
    ("ijlf.,:;!'[]|", 0.58), ("- " + NBSP, 0.67), ("rtI()", 0.77), ("sJ", 0.8),
    ("cez†", 0.89), ("abgov/–0123456789", 1.0), ("kqxy", 1.06), ("bdhnpuS", 1.11),
    ("FLZ", 1.26), ("BCEPT", 1.42), ("ADGHKNOQRUVXYw", 1.5), (MINUS + "<>=+", 1.55),
    ("m%", 1.67), ("M", 1.83), ("W" + EM_DASH, 2.0)) for ch in chars}
#: separator dashes per digit width (fine-grained relative column widths)
DASHES_PER_DIGIT = 3


def _print_width(text: str) -> float:
    """Approximate printed width of ``text`` in digit widths (``_GLYPH_WIDTH``)."""
    return sum(_GLYPH_WIDTH.get(ch, 1.4 if ch.isupper() else 1.0) for ch in text)


def pipe_table(frame: pd.DataFrame) -> str:
    """A pandoc pipe table. Columns holding words are left-aligned, numbers right.

    Pandoc fits a wide table to the text width with relative column widths taken
    from the dashes of the separator line, and LaTeX wraps each cell within its
    column. A column's share is proportional to the widest thing it must print on
    one line, measured in glyph widths rather than characters (``_print_width``):
    a number's longest unbreakable piece (a "[min, max]" range is one), and the
    longest word of a text cell or header, since text wraps between words (a header
    also after a hyphen: "PICP-" / "80"; a name such as "Chronos-2" stays whole). A
    text column claims no more than that word, so in a crowded table the numbers keep
    their width and the names wrap.
    """
    def keep_units(text: str) -> str:
        # a number and its unit stay on one line: "2,048 h)", never "2,048" / "h)"
        return re.sub(r"(\d) (h\b|s\b)", rf"\1{NBSP}\2", text)

    columns = [keep_units(str(c)) for c in frame.columns]
    cells = [[keep_units(str(v)).replace("|", "\\|") for v in row]
             for row in frame.itertuples(index=False, name=None)]
    words = [any(any(ch.isalpha() for ch in r[i]) for r in cells if r[i] != EM_DASH)
             for i in range(len(columns))]

    def longest(text: str) -> float:
        return max(_print_width(piece) for piece in text.split(" "))

    rule_widths = [max(3, round(DASHES_PER_DIGIT * max([longest(c.replace("-", "- ")),
                                                        *(longest(r[i]) for r in cells)])))
                   for i, c in enumerate(columns)]
    widths = [max([len(c), *(len(r[i]) for r in cells)]) for i, c in enumerate(columns)]

    def line(values: list[str]) -> str:
        return "| " + " | ".join(v.ljust(w) if left else v.rjust(w)
                                 for v, w, left in zip(values, widths, words)) + " |"

    # pandoc counts the colon as part of a column's separator width
    rule = "|" + "|".join(":" + "-" * (w - 1) if left else "-" * (w - 1) + ":"
                          for w, left in zip(rule_widths, words)) + "|"
    return "\n".join([line(columns), rule, *(line(r) for r in cells)])


def write_table(table: Table, tables_dir: Path) -> list[Path]:
    """CSV (numbers at full precision) and Markdown: caption line, blank line, table;
    an appendix table goes to the ``APPENDIX_DIR`` folder of ``tables_dir``."""
    if table.appendix:
        tables_dir = tables_dir / APPENDIX_DIR
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv, md = tables_dir / f"{table.stem}.csv", tables_dir / f"{table.stem}.md"
    table.numeric.to_csv(csv, index=False)
    md.write_text(f": {table.caption} {{#tbl:{table.label}}}\n\n{pipe_table(table.display)}\n",
                  encoding="utf-8")
    return [csv, md]


def _present(panel: Panel, models: Iterable[str]) -> list[str]:
    """The models of a set that have predictions, in the set's order."""
    return [m for m in models if m in panel.models]


def _interval_note() -> str:
    """The interval method, as captions state it."""
    return (f"95 % interval from a {BLOCK_DAYS}-day moving-block bootstrap over the test days "
            f"({BOOTSTRAP_RESAMPLES:,} resamples, the same days for all stations)")


def _skill_cells(panel: Panel, boot: Bootstrap, model: str) -> tuple[dict, dict]:
    """A model's skill over ``HEADLINE_REF`` per setting where it is reported: display
    cells ("Skill (<setting>)": "0.280 [0.231, 0.326]", the reference's own row "0")
    and numeric columns (skill_<s>, skill_<s>_lo, skill_<s>_hi)."""
    shown = panel.reported_settings(model)
    cells, values = {}, {}
    for s in panel.settings:
        est = boot.skill(model, s) if s in shown else NOT_AVAILABLE
        lo, hi = est.interval
        own = model == HEADLINE_REF and est.available
        cells[f"Skill ({setting_label(s)})"] = "0" if own else _interval_cell(est)
        values.update({f"skill_{s}": est.point, f"skill_{s}_lo": lo, f"skill_{s}_hi": hi})
    return cells, values


def table_master_skill(panel: Panel, summary: Summary, boot: Bootstrap) -> Table:
    """T1: per main model, skill over ``HEADLINE_REF`` with its 95 % interval and the
    mean nRMSE over stations, per setting."""
    models = _present(panel, MAIN_MODELS)
    display, numeric = [], []
    for model in models:
        shown = panel.reported_settings(model)
        cells, values = _skill_cells(panel, boot, model)
        drow = {"Model": _model_cell(panel, model), **cells}
        nrow = {**_identity(panel, model), "reference": HEADLINE_REF, **values}
        for s in panel.settings:
            nrmse = summary.stat(model, s, "nrmse") if s in shown else math.nan
            drow[f"nRMSE ({setting_label(s)})"] = _num(nrmse)
            nrow[f"nrmse_{s}"] = nrmse
            nrow[f"n_stations_{s}"] = (summary.n_stations(model, s, skill_metric())
                                       if s in shown else 0)
        display.append(drow)
        numeric.append(nrow)
    ref, other = reference_name(), reference_name(CLIPER if HEADLINE_REF == SP else SP)
    caption = (
        f"Accuracy on daytime hours: skill over {ref} (1 {MINUS} RMSE / RMSE of {ref}), "
        "each station pooled over all test days, mean over the "
        f"{len(panel.sites)} stations, with its {_interval_note()}; nRMSE = RMSE / "
        "capacity, mean over stations. Probabilistic models are scored on the mean of "
        f"their nine quantiles. The full table, with skill over {other} and the range "
        "across stations, is @tbl:master-skill-full."
        + _no_cov_note(panel, models) + _station_note(panel, summary, models))
    return Table("master_skill", "master-skill", caption, pd.DataFrame(display),
                 pd.DataFrame(numeric))


#: the setting whose PICP-80 and time T2 shows
PRIMARY_SETTING = "nwp"


def table_chronos2_configs(panel: Panel, summary: Summary, boot: Bootstrap,
                           medians: pd.DataFrame) -> Table:
    """T2: the Chronos-2 configurations (``CHRONOS2_MODELS``): skill over
    ``HEADLINE_REF`` with its interval per setting, PICP-80 and the median fit +
    predict seconds per test window in the NWP setting. A configuration without a
    run at every station in each setting it ran in is left out (with a warning and a
    caption note), never averaged over fewer stations."""
    display, numeric, left_out = [], [], []
    for model in _present(panel, CHRONOS2_MODELS):
        ran = [s for s in panel.settings if panel.has(model, s)]
        if not all(complete(panel, model, s) for s in ran):
            logger.warning("T2: %s left out: not every station has a run", model)
            left_out.append(display_name(model))
            continue
        cells, values = _skill_cells(panel, boot, model)
        primary = PRIMARY_SETTING in panel.reported_settings(model)
        picp = summary.stat(model, PRIMARY_SETTING, "picp_80") \
            if primary and panel.is_quantile(model, PRIMARY_SETTING) else math.nan
        key = (model, PRIMARY_SETTING)
        seconds = float(medians.at[key, "window_s"]) \
            if primary and key in medians.index and pd.notna(medians.at[key, "window_s"]) \
            else math.nan
        checkpoint, context_h, _ = CHRONOS2_SETUP.get(model, ("", 0, ""))
        display.append({"Configuration": _model_cell(panel, model),
                        "Checkpoint": checkpoint,
                        "Context (h)": f"{context_h:,}" if context_h else EM_DASH, **cells,
                        f"PICP-80 ({setting_label(PRIMARY_SETTING)})": _num(picp),
                        "Time per window (s)": _num(seconds, 1)})
        numeric.append({**_identity(panel, model), "reference": HEADLINE_REF, **values,
                        f"picp_80_{PRIMARY_SETTING}": picp,
                        f"median_window_s_{PRIMARY_SETTING}": seconds})
    columns = ["Configuration", "Checkpoint", "Context (h)",
               *(f"Skill ({setting_label(s)})" for s in panel.settings),
               f"PICP-80 ({setting_label(PRIMARY_SETTING)})", "Time per window (s)"]
    missing = (" Not shown (a run is missing at some station): " + "; ".join(left_out) + "."
               if left_out else "")
    shown = [m for m in _present(panel, CHRONOS2_MODELS) if display_name(m) not in left_out]
    caption = (
        "The Chronos-2 configurations. Names: S and B for the small and base checkpoints, "
        "-mid and -long for contexts longer than the one fixed before the main run, -FT for "
        "fine-tuning on each training window (repeated with seeds 1 and 2 in the NWP "
        f"setting only). Skill over {reference_name()} on daytime hours as in "
        f"@tbl:master-skill, with its {_interval_note()}. PICP-80: share of hours inside "
        "the 10–90 % interval (nominal 0.8). Time: median over the station-folds of fit "
        f"+ predict seconds per {window_days(panel)}-day test window." + missing
        + _no_cov_note(panel, shown))
    return Table("chronos2_configs", "chronos2-configs", caption,
                 pd.DataFrame(display, columns=columns), pd.DataFrame(numeric))


def _setting_blocks(panel: Panel, models: Iterable[str]) -> list[tuple[str, str, bool]]:
    """(setting, model, first row of its block) rows for the long tables: a block per
    setting, models in the given order, each no-covariate model once (No weather)."""
    models = list(models)
    rows = []
    for s in panel.settings:
        block = [m for m in models if s in panel.reported_settings(m)]
        rows += [(s, m, i == 0) for i, m in enumerate(block)]
    return rows


def _quantile_blocks(panel: Panel, models: Iterable[str]) -> list[tuple[str, str, bool]]:
    """``_setting_blocks`` restricted to the runs that issue the nine quantiles."""
    rows = [(s, m) for s, m, _ in _setting_blocks(panel, models) if panel.is_quantile(m, s)]
    return [(s, m, i == 0 or rows[i - 1][0] != s) for i, (s, m) in enumerate(rows)]


def table_probabilistic(panel: Panel, summary: Summary, boot: Bootstrap) -> Table:
    """T3: per probabilistic main model and setting, the quantile score / C (mean
    pinball loss over the nine levels / C = the store column ``crps`` / 2), PICP-80
    with its 95 % interval and the mean width-80 / C (means over stations)."""
    display, numeric = [], []
    models = _present(panel, MAIN_MODELS)
    for s, model, first in _quantile_blocks(panel, models):
        qs = summary.stat(model, s, "ncrps") / 2.0
        picp = boot.picp(model, s)
        display.append({"Setting": setting_label(s) if first else "",
                        "Model": _model_cell(panel, model),
                        "QS / C": _num(qs, 4),
                        "PICP-80": _interval_cell(picp),
                        "Width-80 / C": _num(summary.stat(model, s, "nwidth_80"))})
        numeric.append({"setting": s, **_identity(panel, model), "qs_c": qs,
                        "picp_80": picp.point, "picp_80_lo": picp.interval[0],
                        "picp_80_hi": picp.interval[1],
                        "width_80_c": summary.stat(model, s, "nwidth_80")})
    caption = (
        "Probabilistic accuracy on daytime hours of the models that issue quantiles, mean "
        "over the stations of each station's value pooled over all test days. QS / C: the "
        "quantile score, the mean pinball loss over the nine quantile levels 0.1, …, 0.9, "
        "divided by capacity C (lower is better); twice the QS is a common approximation of "
        "the CRPS. PICP-80: share of hours inside the 10–90 % interval (nominal 0.8), with "
        f"its {_interval_note()}. Width-80 / C: the interval's mean width divided by C."
        + _no_cov_note(panel, models) + _station_note(panel, summary, models))
    return Table("probabilistic", "probabilistic", caption,
                 pd.DataFrame(display, columns=["Setting", "Model", "QS / C", "PICP-80",
                                                "Width-80 / C"]),
                 pd.DataFrame(numeric))


#: bytes per GB in the compute table: the store's "MB" are MiB (ru_maxrss / 1024²)
MB_PER_GB = 1024.0


def table_compute(panel: Panel, medians: pd.DataFrame) -> Table:
    """T4: per main model (references left out) and setting, the median fit and
    predict seconds over its station-folds and the maximum peak RSS, in GB."""
    display, numeric = [], []
    models = [m for m in _present(panel, MAIN_MODELS) if family(m) != REFERENCE]
    for s, model, first in _setting_blocks(panel, models):
        rec = medians.loc[(model, s)] if (model, s) in medians.index else None

        def get(col: str, rec: pd.Series | None = rec) -> float:
            return float(rec[col]) if rec is not None and pd.notna(rec[col]) else math.nan

        display.append({"Setting": setting_label(s) if first else "",
                        "Model": _model_cell(panel, model),
                        "Fit (s)": _num(get("fit_s"), 1),
                        "Predict (s)": _num(get("predict_s"), 1),
                        "Max memory (GB)": _num(get("max_peak_rss_mb") / MB_PER_GB, 1)})
        numeric.append({"setting": s, **_identity(panel, model),
                        **{f"median_{c}": get(c) for c in (*TIMING_COLUMNS, "window_s")},
                        "max_peak_rss_mb": get("max_peak_rss_mb"),
                        "max_peak_rss_gb": get("max_peak_rss_mb") / MB_PER_GB,
                        "n_station_folds": get("n")})
    caption = (
        f"Compute cost per test window of {window_days(panel)} days: median wall-clock "
        "seconds over the station-folds to fit the model and to issue the window's "
        "day-ahead forecasts, and the maximum over the station-folds of the process's peak "
        "resident set size (one process per run, so it includes the data and the "
        "framework; 1 GB = 1,024³ bytes). Memory does not count GPU (Metal) memory, so it "
        "is a lower bound for the neural models." + _no_cov_note(panel, models))
    return Table("compute", "compute", caption,
                 pd.DataFrame(display, columns=["Setting", "Model", "Fit (s)", "Predict (s)",
                                                "Max memory (GB)"]),
                 pd.DataFrame(numeric))


#: verdicts as printed in the detail table (the CSV keeps the words); the caption explains them
VERDICT_SHORT = {"no significant difference": "n.s.", "not available": "n/a"}


def _side(panel: Panel, model: str, setting: str) -> str:
    """One side named as in T1: a no-covariate model with †, any other with its setting."""
    if model in panel.no_covariates:
        return _model_cell(panel, model)
    return f"{display_name(model)}, {setting_label(setting)}"


def _comparison(panel: Panel, r: dict) -> str:
    """"Setting: A vs B" when both sides run in one setting, "Model: setting A vs
    setting B" when one model is compared across settings, else both sides in full."""
    (ma, sa), (mb, sb) = (r["model_a"], r["setting_a"]), (r["model_b"], r["setting_b"])
    if sa == sb:
        return f"{setting_label(sa)}: {_model_cell(panel, ma)} vs {_model_cell(panel, mb)}"
    if ma == mb:
        return f"{display_name(ma)}: {setting_label(sa)} vs {setting_label(sb)}"
    return f"{_side(panel, ma, sa)} vs {_side(panel, mb, sb)}"


def _difference_cell(r: dict) -> str:
    if not r["available"] or not _finite(r["skill_diff"]):
        return EM_DASH
    return _num_range(r["skill_diff"], r["skill_diff_lo"], r["skill_diff_hi"])


def table_significance(panel: Panel, sig: pd.DataFrame, post_hoc: pd.DataFrame) -> Table:
    """T5: the comparisons fixed before panel v2 (P1-P11: paired skill difference over
    ``HEADLINE_REF`` with its interval, Holm p, verdict) and the post hoc ones (E1-E5:
    skill difference with its interval only, no test). ``sig`` comes from
    ``add_skill_differences``; the test statistics are in ``significance_detail``."""
    columns = ["ID", "Comparison (A vs B)", f"Skill difference (A {MINUS} B)", "Holm p",
               "Verdict"]
    display = []
    for r in sig.to_dict("records"):
        ok = bool(r["available"])
        display.append(dict(zip(columns, [
            r["id"], _comparison(panel, r), _difference_cell(r), _p(r["p_holm"]),
            VERDICT_SHORT.get(r["verdict"], r["verdict"]) if ok else "not available"])))
    for r in post_hoc.to_dict("records"):
        display.append(dict(zip(columns, [
            r["id"], _comparison(panel, r), _difference_cell(r), EM_DASH,
            "post hoc" if r["available"] else "not available"])))
    n_tested = int(sig["p_holm"].notna().sum())
    first = f"{sig['id'].iloc[0]}–{sig['id'].iloc[-1]}" if len(sig) else "P"
    later = (f"{post_hoc['id'].iloc[0]}–{post_hoc['id'].iloc[-1]}"
             if len(post_hoc) else "E")
    ref = reference_name()
    caption = (
        f"Pairwise comparisons. Skill difference: skill over {ref} of A minus that of B "
        f"(positive: A more accurate), with its {_interval_note()}. {first} were fixed "
        "before the main run, after a first run on the same test days had been examined: "
        "Diebold-Mariano test on daily squared-error losses, Holm-adjusted over the "
        f"{n_tested} comparisons; verdict at Holm p < {ALPHA} (n.s.: no significant "
        f"difference). {later} are post hoc (they involve configurations added after the "
        "main run) and are reported with intervals only. Full test statistics: "
        "@tbl:significance-detail."
        + _no_cov_note(panel, set(sig["model_a"]) | set(sig["model_b"])
                       | set(post_hoc.get("model_a", [])) | set(post_hoc.get("model_b", []))))
    numeric = pd.concat([sig.assign(block="fixed before panel v2"),
                         post_hoc.assign(block="post hoc")], ignore_index=True)
    return Table("significance", "significance", caption,
                 pd.DataFrame(display, columns=columns), numeric)


# --------------------------------------------------------------------------- #
# Appendix tables
# --------------------------------------------------------------------------- #
def table_master_skill_full(panel: Panel, summary: Summary) -> Table:
    """Every model: skill over SP (and over CLIPER when present) as mean [min, max]
    over stations, and the mean over stations of nRMSE and nMBE, per setting."""
    blocks = [("skill_sp", "Skill vs SP", True)]
    if CLIPER in panel.models:
        blocks.append(("skill_cliper", "Skill vs CLIPER", True))
    blocks += [("nrmse", "nRMSE", False), ("nmbe", "nMBE", False)]
    display, numeric = [], []
    for model in panel.models:
        shown = panel.reported_settings(model)
        drow, nrow = {"Model": _model_cell(panel, model)}, _identity(panel, model)
        for s in panel.settings:
            nrow[f"n_stations_{s}"] = summary.n_stations(model, s) if s in shown else 0
        for metric, label, with_range in blocks:
            for s in panel.settings:
                mean, lo, hi = summary.triple(model, s, metric) if s in shown \
                    else (math.nan,) * 3
                drow[f"{label} ({setting_label(s)})"] = (
                    _num_range(mean, lo, hi) if with_range else _num(mean))
                nrow.update({f"{metric}_{s}_mean": mean, f"{metric}_{s}_min": lo,
                             f"{metric}_{s}_max": hi})
        display.append(drow)
        numeric.append(nrow)
    vs_cliper = " and over CLIPER" if CLIPER in panel.models else ""
    caption = (
        f"Accuracy of every model run on daytime hours: skill over Smart persistence (SP)"
        f"{vs_cliper} as the mean over the {len(panel.sites)} stations [min, max across "
        "stations: the spread between stations, not an uncertainty interval], and "
        "capacity-normalised RMSE and mean bias error (nMBE > 0: over-forecast), mean over "
        "stations. Each station's metrics pool the hours of all test windows; probabilistic "
        "models are scored on the mean of their nine quantiles."
        + _no_cov_note(panel, panel.models) + _station_note(panel, summary, panel.models))
    return Table("master_skill_full", "master-skill-full", caption, pd.DataFrame(display),
                 pd.DataFrame(numeric), appendix=True)


def table_significance_detail(panel: Panel, sig: pd.DataFrame) -> Table:
    """The test statistics of the comparisons fixed before panel v2 (T5 block 1).

    Laid out to fit the text width at the manuscript's 11 pt: A and B share one cell,
    the verdicts are short (explained in the caption), and the day count goes in the
    caption unless the comparisons cover different numbers of days.
    """
    days = sorted({int(n) for n in sig.loc[sig["available"], "n_days"].dropna()})
    display = []
    for r in sig.to_dict("records"):
        ok = bool(r["available"])
        row = {"ID": r["id"], "Comparison (A vs B)": _comparison(panel, r)}
        if len(days) > 1:
            row["Days"] = _int(r["n_days"]) if ok else EM_DASH
        display.append({
            **row,
            "Loss A": _num(r["loss_a"], 4) if ok else EM_DASH,
            "Loss B": _num(r["loss_b"], 4) if ok else EM_DASH,
            "Rel. diff. (%)": _num(r["rel_diff_pct"], 1) if ok else EM_DASH,
            "DM": _num(r["dm_stat"], 2) if ok else EM_DASH,
            "p": _p(r["p_dm"]), "Holm p": _p(r["p_holm"]),
            "Bootstrap p": _p(r["p_bootstrap"]),
            "Verdict": VERDICT_SHORT.get(r["verdict"], r["verdict"]),
        })
    n_tested = int(sig["p_holm"].notna().sum())
    span = f" over the same {days[0]} days" if len(days) == 1 else ""
    missing = (" n/a: not available (a model has no predictions)."
               if (~sig["available"].astype(bool)).any() else "")
    caption = (
        f"Test statistics of the comparisons of @tbl:significance fixed before the main run, "
        f"on daily losses{span}: each local day's loss "
        f"is the station average of the mean ((forecast {MINUS} truth) / C)² over the day's "
        "scored daytime hours; Loss A and B are their means over the days, and Rel. diff. "
        f"is the mean of d = loss A {MINUS} loss B relative to loss B (negative: A better). "
        "DM: Diebold-Mariano statistic with Newey-West variance and the "
        "Harvey-Leybourne-Newbold correction, two-sided p from Student's t; Holm p adjusts "
        f"over the {n_tested} comparisons that could be run; Bootstrap p: moving-block "
        f"bootstrap ({BLOCK_DAYS}-day blocks, {BOOTSTRAP_RESAMPLES:,} resamples). Newey-West "
        "lags and bootstrap blocks run over the days in date order, so they also span the "
        f"gaps between test windows. Verdict at Holm p < {ALPHA}; n.s.: no significant "
        "difference." + missing
        + _no_cov_note(panel, set(sig["model_a"]) | set(sig["model_b"])))
    return Table("significance_detail", "significance-detail", caption, pd.DataFrame(display),
                 sig.copy(), appendix=True)


def _appendix_models(panel: Panel) -> list[str]:
    """The rows of the per-station and per-window tables: the main models but the
    headline reference (its skill is 0 everywhere)."""
    return [m for m in _present(panel, MAIN_MODELS) if m != HEADLINE_REF]


def table_skill_by_station(panel: Panel, summary: Summary) -> Table:
    """Skill over ``HEADLINE_REF`` per main model x station (and its mean), a block
    per setting."""
    display, numeric = [], []
    metric = skill_metric()
    models = _appendix_models(panel)
    for s, model, first in _setting_blocks(panel, models):
        values = {site: summary.at_site(model, s, site, metric) for site in panel.sites}
        mean = summary.stat(model, s, metric)
        display.append({"Setting": setting_label(s) if first else "",
                        "Model": _model_cell(panel, model),
                        **{site: _num(v) for site, v in values.items()}, "Mean": _num(mean)})
        numeric.append({"setting": s, **_identity(panel, model), "reference": HEADLINE_REF,
                        **values, "mean": mean})
    caption = (f"Skill over {reference_name()} per station (daytime hours, all test windows "
               "pooled) and its mean over stations." + _no_cov_note(panel, models)
               + _station_note(panel, summary, models))
    return Table("skill_by_station", "skill-by-station", caption, pd.DataFrame(display),
                 pd.DataFrame(numeric), appendix=True)


def fold_means(folds: pd.DataFrame, metric: str | None = None) -> pd.DataFrame:
    """Mean over stations of the per-station-fold skill (default: over
    ``HEADLINE_REF``), (model, setting) x fold."""
    metric = skill_metric() if metric is None else metric
    if folds.empty or metric not in folds.columns:
        return pd.DataFrame()
    return folds.groupby(["model", "setting", "fold"])[metric].mean().unstack("fold")


def table_skill_by_fold(panel: Panel, summary: Summary, folds: pd.DataFrame,
                        labels: dict[int, str]) -> Table:
    """Skill over ``HEADLINE_REF`` per test window: per station-fold 1 - RMSE ratio,
    then the mean over stations; columns are labelled by ``fold_labels``."""
    means = fold_means(folds)
    display, numeric = [], []
    models = _appendix_models(panel)
    for s, model, first in _setting_blocks(panel, models):
        row = means.loc[(model, s)] if (model, s) in means.index else pd.Series(dtype=float)
        values = {f: float(row.get(f, math.nan)) for f in labels}
        display.append({"Setting": setting_label(s) if first else "",
                        "Model": _model_cell(panel, model),
                        **{labels[f]: _num(v) for f, v in values.items()}})
        numeric.append({"setting": s, **_identity(panel, model), "reference": HEADLINE_REF,
                        **{f"fold_{f}": v for f, v in values.items()}})
    columns = ("each window's start date" if any(t[:1].isdigit() for t in labels.values())
               else "the month holding the middle of each window")
    caption = (f"Skill over {reference_name()} per {window_days(panel)}-day test window "
               f"(columns: {columns}), computed per station and window on daytime hours and "
               "averaged over stations." + _no_cov_note(panel, models)
               + _station_note(panel, summary, models))
    return Table("skill_by_fold", "skill-by-fold", caption, pd.DataFrame(display),
                 pd.DataFrame(numeric), appendix=True)


def median_is_y_hat(panel: Panel, model: str, setting: str) -> bool:
    """Whether a quantile run's ``y_hat`` equals its 0.5 quantile on every scored
    hour (the median rule scores ``y_hat``)."""
    for site in panel.sites:
        frame = panel.runs.get((model, setting, site))
        if frame is None:
            continue
        mask = scored_mask(frame)
        if not np.allclose(frame["y_hat"].to_numpy(float)[mask],
                           frame["q0.5"].to_numpy(float)[mask], rtol=0, atol=1e-9):
            return False
    return True


def table_point_rule(panel: Panel, boot: Bootstrap) -> Table:
    """Skill over ``HEADLINE_REF`` of the probabilistic main models under the point
    rule used (the mean of the nine quantiles) and under the median rule (``y_hat``,
    checked to equal q0.5), and their difference."""
    display, numeric, differs = [], [], []
    models = _present(panel, MAIN_MODELS)
    for s, model, first in _quantile_blocks(panel, models):
        if not median_is_y_hat(panel, model, s):
            logger.warning("point rule: %s/%s has y_hat != q0.5 on some scored hour", model, s)
            differs.append(f"{display_name(model)}, {setting_label(s)}")
        mean_rule = boot.skill(model, s).point
        median_rule = boot.skill(model, s, rule="median").point
        display.append({"Setting": setting_label(s) if first else "",
                        "Model": _model_cell(panel, model),
                        "Skill, mean rule": _num(mean_rule),
                        "Skill, median rule": _num(median_rule),
                        "Difference": _num(mean_rule - median_rule)})
        numeric.append({"setting": s, **_identity(panel, model), "reference": HEADLINE_REF,
                        "skill_mean_rule": mean_rule, "skill_median_rule": median_rule,
                        "difference": mean_rule - median_rule})
    note = (" For these runs y_hat is not the 0.5 quantile: " + "; ".join(differs) + "."
            if differs else "")
    caption = (
        f"Sensitivity to the point-forecast rule: skill over {reference_name()} on daytime "
        "hours (mean over stations) when a probabilistic model is scored on the mean of its "
        "nine quantiles (the rule used throughout) and on its median (the 0.5 quantile), and "
        "the difference (mean rule minus median rule)." + note + _no_cov_note(panel, models))
    return Table("point_rule", "point-rule", caption,
                 pd.DataFrame(display, columns=["Setting", "Model", "Skill, mean rule",
                                                "Skill, median rule", "Difference"]),
                 pd.DataFrame(numeric), appendix=True)


def table_p2_robustness(panel: Panel, boot: Bootstrap, labels: dict[int, str]) -> Table:
    """The P2 skill difference (LightGBM minus fine-tuned Chronos-2, NWP, over
    ``HEADLINE_REF``) with its interval: on all windows with each fine-tuning seed,
    and with seed 0 leaving out each test window in turn (the interval then resamples
    the remaining days)."""
    a_model, b_model = P2_PAIR
    a = (a_model, PRIMARY_SETTING)
    rows = []   # (label, window left out or None, model B, days used, estimate)
    for model, seed in FT_SEEDS:
        if not (complete(panel, *a) and complete(panel, model, PRIMARY_SETTING)):
            logger.warning("P2 robustness: %s left out: not every station has a run", model)
            continue
        label = ("All windows (seed 0)" if model == b_model
                 else f"All windows, {seed} {POST_HOC_MARK}")
        rows.append((label, None, model, len(boot.days),
                     boot.skill_difference(a, (model, PRIMARY_SETTING))))
    if complete(panel, *a) and complete(panel, b_model, PRIMARY_SETTING):
        for fold, name in labels.items():
            days = boot.fold != fold
            if not days.all():
                rows.append((f"Without {name}", fold, b_model, int(days.sum()),
                             boot.skill_difference(a, (b_model, PRIMARY_SETTING), days=days)))
    column = f"{display_name(a_model)} {MINUS} {display_name(b_model)}"
    display = [{"Test days": label, column: _interval_cell(est)}
               for label, _, _, _, est in rows]
    numeric = [{"variant": label, "fold_left_out": fold, "model_a": a_model, "model_b": model,
                "setting": PRIMARY_SETTING, "reference": HEADLINE_REF, "n_days": n_days,
                "skill_diff": est.point, "skill_diff_lo": est.interval[0],
                "skill_diff_hi": est.interval[1]} for label, fold, model, n_days, est in rows]
    caption = (
        f"Robustness of comparison P2 (@tbl:significance): skill over {reference_name()} of "
        f"{display_name(a_model)} minus that of {display_name(b_model)}, "
        f"{setting_label(PRIMARY_SETTING)} setting (positive: {display_name(a_model)} more "
        f"accurate), with its {_interval_note()}: on all test windows with each fine-tuning "
        "seed, and with seed 0 leaving out one test window (named by the month holding its "
        "middle) at a time, the interval then resampling the remaining days."
        + (_no_cov_note(panel, [m for _, _, m, _, _ in rows])))
    return Table("p2_robustness", "p2-robustness", caption,
                 pd.DataFrame(display, columns=["Test days", column]),
                 pd.DataFrame(numeric), appendix=True)


def table_all_hours(panel: Panel, summary: Summary) -> Table:
    """The skill vs SP scored on all hours (night included), and the all-hours nRMSE;
    the caption quantifies both changes against daytime scoring."""
    display, numeric, skill_shift, nrmse_ratio = [], [], [], []
    for model in panel.models:
        shown = panel.reported_settings(model)
        drow, nrow = {"Model": _model_cell(panel, model)}, _identity(panel, model)
        for metric, label in (("skill_sp", "Skill vs SP, all hours"),
                              ("nrmse", "nRMSE, all hours")):
            for s in panel.settings:
                values = {hours: summary.triple(model, s, metric, hours) if s in shown
                          else (math.nan,) * 3 for hours in HOURS}
                mean, lo, hi = values["all"]
                drow[f"{label} ({setting_label(s)})"] = (
                    _num_range(mean, lo, hi) if metric == "skill_sp" else _num(mean))
                for hours, triple in values.items():
                    nrow.update({f"{metric}_{hours}_{s}_{how}": v
                                 for how, v in zip(("mean", "min", "max"), triple)})
                day = values["daytime"][0]
                if _finite(mean) and _finite(day):
                    if metric == "skill_sp":
                        skill_shift.append(abs(mean - day))
                    elif day > 0:
                        nrmse_ratio.append(mean / day)
        display.append(drow)
        numeric.append(nrow)
    shift = (f" No model's mean skill moves by more than {max(skill_shift):.5f} from its "
             "daytime value (@tbl:master-skill-full): at night both truth and forecasts are "
             "close to zero for every model, so the night adds almost nothing to either RMSE "
             "in the skill ratio." if skill_shift else "")
    dilution = (f" The night zeros do lower nRMSE, by {100 * (1 - np.mean(nrmse_ratio)):.0f} % "
                "on average, which is why accuracy is reported on daytime hours."
                if nrmse_ratio else "")
    caption = ("Robustness of the headline to the scoring mask: skill over Smart persistence "
               "and nRMSE scored on all hours with valid truth (night included); skill as the "
               "mean over stations [min, max], nRMSE as the mean." + shift + dilution
               + _no_cov_note(panel, panel.models)
               + _station_note(panel, summary, panel.models))
    return Table("all_hours", "all-hours", caption, pd.DataFrame(display),
                 pd.DataFrame(numeric), appendix=True)


def table_panel_diagnostics(diagnostics: Diagnostics) -> Table:
    """Per-station scored data and the correlation of SP errors across stations."""
    diag, pairs = diagnostics.table, pairwise_correlations(diagnostics.corr)
    display = [{"Station": r["site"], "Capacity (kW)": _num(r["capacity_kw"], 2),
                "Days": _int(r["days"]), "Daytime hours": _int(r["daytime_hours"]),
                "All hours": _int(r["all_hours"]),
                "Mean abs. SP error / C": _num(r["mean_abs_sp_error_c"]),
                "Corr. with other stations": _num(r["mean_corr_others"])}
               for r in diag.to_dict("records")]
    display.append({"Station": "Panel", "Capacity (kW)": EM_DASH,
                    "Days": _int(diagnostics.panel_days),
                    "Daytime hours": _int(diag["daytime_hours"].sum()),
                    "All hours": _int(diag["all_hours"].sum()),
                    "Mean abs. SP error / C": _num(diag["mean_abs_sp_error_c"].mean()),
                    "Corr. with other stations": _num_range(
                        pairs.mean(), pairs.min(), pairs.max(), 2) if pairs.size else EM_DASH})
    numeric = diag.copy()
    numeric.loc[len(numeric)] = {"site": "Panel", "capacity_kw": math.nan,
                                 "days": diagnostics.panel_days,
                                 "daytime_hours": diag["daytime_hours"].sum(),
                                 "all_hours": diag["all_hours"].sum(),
                                 "mean_abs_sp_error_c": diag["mean_abs_sp_error_c"].mean(),
                                 "mean_corr_others": pairs.mean() if pairs.size else math.nan}
    caption = (
        "The station panel: nameplate capacity, scored local days and hours (daytime "
        "hours score the headline; all hours with valid truth score the robustness check), "
        "the mean absolute smart-persistence error / C, and the Pearson correlation of "
        "hourly absolute SP errors / C with the other stations (each pair on its common "
        f"daytime hours). Panel row: totals, and the mean of the {pairs.size} pairwise "
        "correlations [min, max]. The stations share one campus and one weather regime, so "
        "their errors are far from independent.")
    return Table("panel_diagnostics", "panel-diagnostics", caption, pd.DataFrame(display),
                 numeric, appendix=True)


def metrics_long(panel: Panel, station: pd.DataFrame) -> pd.DataFrame:
    """Every metric per model x setting x station x hour set, one row per value."""
    ids = ["model", "setting", "site", "hours", "quantile"]
    values = [c for c in station.columns if c not in ids]
    long = station.melt(id_vars=ids, value_vars=values, var_name="metric",
                        value_name="value").dropna(subset=["value"])
    long = long.rename(columns={"setting": "weather_mode", "quantile": "probabilistic"})
    long.insert(1, "display_name", long["model"].map(display_name))
    long.insert(2, "family", long["model"].map(family))
    long.insert(3, "no_covariates", long["model"].isin(panel.no_covariates))
    long.insert(6, "capacity_kw", long["site"].map(panel.capacity))

    def thesis_order(col: pd.Series) -> pd.Series:
        if col.name == "model":
            return col.map({m: i for i, m in enumerate(panel.models)})
        if col.name == "weather_mode":
            return col.map({s: i for i, s in enumerate(panel.settings)})
        return col

    return long.sort_values(["model", "weather_mode", "site", "hours", "metric"],
                            key=thesis_order).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save(fig: plt.Figure, figures_dir: Path, stem: str) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    pdf, png = figures_dir / f"{stem}.pdf", figures_dir / f"{stem}.png"
    fig.savefig(pdf, metadata={"CreationDate": None})  # no timestamp: reruns are identical
    fig.savefig(png, dpi=200)
    plt.close(fig)
    return [pdf, png]


def _figure_label(panel: Panel, model: str) -> str:
    return _post_hoc(model) + (f" {NO_COV_LABEL}" if model in panel.no_covariates else "")


def _zero_line(ax: plt.Axes, axis: str = "y") -> None:
    """The zero reference on a skill axis (skill 0 = the headline reference)."""
    (ax.axhline if axis == "y" else ax.axvline)(0.0, color=INK_2, linewidth=0.8, zorder=1.5)


def _bar_style(model: str, setting: str) -> dict:
    """NWP: solid family hue; No weather: the hue at 45 % with a //// hatch."""
    hue = colour(model)
    if setting == "none":
        return {"facecolor": to_rgba(hue, NO_WEATHER_ALPHA), "edgecolor": hue,
                "hatch": HATCH, "linewidth": 0}
    return {"facecolor": hue, "edgecolor": hue, "linewidth": 0}


def _marker_style(model: str | None, setting: str, size: float = MARKER) -> dict:
    """NWP: filled with a white ring; No weather: hollow (white face, hue edge)."""
    hue = colour(model) if model else INK_2
    if setting == "none":
        return {"marker": "o", "markersize": size, "markerfacecolor": "white",
                "markeredgecolor": hue, "markeredgewidth": 1.3}
    return {"marker": "o", "markersize": size, "markerfacecolor": hue,
            "markeredgecolor": "white", "markeredgewidth": 0.8}


def _blank() -> Line2D:
    return Line2D([], [], linestyle="none", label="")


def _family_handles(models: Iterable[str], kind: str = "patch") -> list:
    families = []
    for m in models:
        if family(m) not in families:
            families.append(family(m))
    families.sort(key=list(FAMILIES).index)
    if kind == "patch":
        return [Patch(facecolor=FAMILIES[f][1], linewidth=0, label=FAMILIES[f][0])
                for f in families]
    return [Line2D([], [], linestyle="none", marker="o", markersize=MARKER,
                   markerfacecolor=FAMILIES[f][1], markeredgecolor="white",
                   markeredgewidth=0.8, label=FAMILIES[f][0]) for f in families]


def _setting_handles(settings: Iterable[str], kind: str = "patch") -> list:
    handles = []
    for s in setting_order(settings):
        if kind == "patch":
            style = _bar_style("", s) | {"facecolor": to_rgba(INK_2, NO_WEATHER_ALPHA)
                                         if s == "none" else INK_2, "edgecolor": INK_2}
            handles.append(Patch(**style, label=setting_label(s)))
        else:
            handles.append(Line2D([], [], linestyle="none", label=setting_label(s),
                                  **_marker_style(None, s)))
    return handles


def _legend_below(fig: plt.Figure, groups: list[list], max_cols: int | None = None) -> None:
    """One legend under the plot with a row per group of handles (e.g. families, then
    settings). Groups wrap at ``max_cols``, and at fewer columns while the legend is
    wider than the figure (so more models never push it off the page). Matplotlib
    fills legend columns top to bottom, so the padded grid goes in column by column."""
    groups = [g for g in groups if g]
    cols = max_cols or max(len(g) for g in groups)
    renderer = fig.canvas.get_renderer()
    while True:
        rows = [g[i:i + cols] for g in groups for i in range(0, len(g), cols)]
        ncol = max(len(r) for r in rows)
        grid = [r + [_blank() for _ in range(ncol - len(r))] for r in rows]
        ordered = [grid[r][c] for c in range(ncol) for r in range(len(grid))]
        legend = fig.legend(handles=ordered, labels=[h.get_label() for h in ordered],
                            loc="outside lower center", ncol=ncol, handlelength=1.8,
                            columnspacing=1.6, handletextpad=0.6)
        if cols == 1 or legend.get_window_extent(renderer).width <= 0.98 * fig.bbox.width:
            return
        legend.remove()
        cols -= 1


def figure_models(panel: Panel) -> list[str]:
    """The models of the per-model figures: the main models but the headline
    reference, which is their zero line."""
    return [m for m in _present(panel, MAIN_MODELS) if m != HEADLINE_REF]


def _entries(panel: Panel, summary: Summary, models: Iterable[str],
             metric: str | None = None) -> list[tuple[str, list[tuple[str, float]]]]:
    """(model, [(setting, mean over stations), ...]) of the given models, sorted by the
    NWP mean, best first (a model without an NWP value sorts by its other one).
    ``metric`` defaults to skill over ``HEADLINE_REF``."""
    metric = skill_metric() if metric is None else metric
    entries = []
    for model in models:
        values = [(s, summary.stat(model, s, metric)) for s in panel.reported_settings(model)]
        values = [v for v in values if _finite(v[1])]
        if values:
            entries.append((model, values))

    def key(entry: tuple[str, list[tuple[str, float]]]) -> float:
        by_setting = dict(entry[1])
        return -by_setting.get("nwp", entry[1][0][1])

    return sorted(entries, key=key)


def figure_order(panel: Panel, summary: Summary) -> list[str]:
    """The one model order of every per-model figure (F1, F3-F6): by NWP skill over
    ``HEADLINE_REF``, best first, a model without an NWP value by its No-weather one
    (``_entries``); models without a skill value follow in thesis order."""
    ranked = [m for m, _ in _entries(panel, summary, panel.models)]
    return ranked + [m for m in panel.models if m not in ranked]


def fig_skill_by_model(panel: Panel, summary: Summary, boot: Bootstrap,
                       figures_dir: Path) -> list[Path]:
    """F1: skill over ``HEADLINE_REF`` per main model, an NWP and a No-weather bar
    with its 95 % interval; a no-covariate model has one bar."""
    entries = [(model, [(s, boot.skill(model, s)) for s, _ in values])
               for model, values in _entries(panel, summary, figure_models(panel))]
    if not entries:
        logger.warning("F1 skipped: no model to plot")
        return []
    n = len(entries)
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 1.35 + 0.36 * n), layout="constrained")
    height, gap = 0.34, 0.05
    for i, (model, values) in enumerate(entries):
        offsets = [0.0] if len(values) == 1 else [-(height + gap) / 2, (height + gap) / 2]
        for (setting, est), off in zip(values, offsets):
            ax.barh(i + off, est.point, height=height, zorder=2, **_bar_style(model, setting))
            lo, hi = est.interval
            if _finite(lo) and _finite(hi):
                ax.errorbar(est.point, i + off, xerr=[[est.point - lo], [hi - est.point]],
                            fmt="none", ecolor=INK_2, elinewidth=0.8, capsize=2,
                            capthick=0.8, zorder=3)
    ax.set_yticks(range(n), labels=[_figure_label(panel, m) for m, _ in entries])
    ax.set_ylim(n - 0.5, -0.5)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x")
    _zero_line(ax, "x")
    ax.set_xlabel(f"Skill vs {reference_name()} (daytime, mean over stations)")
    settings = {v[0] for _, values in entries for v in values}
    whisker = Line2D([], [], color=INK_2, linewidth=0.8, marker="|", markersize=6,
                     markeredgewidth=0.8, label="95 % interval")
    _legend_below(fig, [_family_handles(m for m, _ in entries),
                        _setting_handles(settings) + [whisker]])
    return _save(fig, figures_dir, "skill_by_model")


#: preferred label side per model in F2 (tried first; the placer moves a label that
#: would collide). Sides: right, left, above, below, and the four diagonals.
F2_LABEL_SIDE: dict[str, str] = {}
_SIDES = {
    "right": ((7, 0), "left", "center"), "left": ((-7, 0), "right", "center"),
    "above": ((0, 7), "center", "bottom"), "below": ((0, -7), "center", "top"),
    "above right": ((6, 6), "left", "bottom"), "below right": ((6, -6), "left", "top"),
    "above left": ((-6, 6), "right", "bottom"), "below left": ((-6, -6), "right", "top"),
}


def _place_labels(ax: plt.Axes, labels: list[tuple[str, float, float, str]],
                  points: list[tuple[float, float]], avoid: list[Bbox],
                  lines: Sequence[Sequence[tuple[float, float]]] = ()) -> None:
    """Greedy direct labelling: each label takes the first side (its preferred one
    first) where it stays inside the axes and clears every marker, every joining line
    (``lines``: polylines in data coordinates) and every label so far."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    frame = ax.get_window_extent(renderer)
    taken = list(avoid)
    for x, y in points:
        cx, cy = ax.transData.transform((x, y))
        r = (MARKER / 2 + 1.0) * px
        taken.append(Bbox([[cx - r, cy - r], [cx + r, cy + r]]))
    tracks = []  # each line segment as display-space points about 1 pt apart
    for line in lines:
        xy = ax.transData.transform(np.asarray(line, dtype=float))
        for p, q in zip(xy[:-1], xy[1:]):
            n = int(np.hypot(*(q - p)) / px) + 2
            tracks.append(p + np.linspace(0.0, 1.0, n)[:, None] * (q - p))
    for model, x, y, text in labels:
        sides = list(_SIDES)
        if model in F2_LABEL_SIDE:
            sides.remove(F2_LABEL_SIDE[model])
            sides.insert(0, F2_LABEL_SIDE[model])
        chosen = None
        for side in sides:
            (dx, dy), ha, va = _SIDES[side]
            ann = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                              ha=ha, va=va, color=INK, fontsize=8, annotation_clip=False)
            ann.set_in_layout(False)
            box = ann.get_window_extent(renderer).padded(1.0 * px)
            inside = (frame.x0 <= box.x0 and box.x1 <= frame.x1
                      and frame.y0 <= box.y0 and box.y1 <= frame.y1)
            if inside and not any(box.overlaps(t) for t in taken) \
                    and not any(box.count_contains(t) for t in tracks):
                chosen = box
                break
            ann.remove()
        if chosen is None:
            (dx, dy), ha, va = _SIDES[sides[0]]
            ann = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                              ha=ha, va=va, color=INK, fontsize=8, annotation_clip=False)
            ann.set_in_layout(False)
            chosen = ann.get_window_extent(renderer)
            logger.warning("F2: no free spot for the label %r; it may overlap", text)
        taken.append(chosen)


def fig_skill_vs_compute(panel: Panel, summary: Summary, medians: pd.DataFrame,
                         figures_dir: Path) -> list[Path]:
    """F2: skill over ``HEADLINE_REF`` against the median fit + predict seconds per
    test window (log x), for the main models without the references."""
    points: dict[str, list[tuple[str, float, float]]] = {}
    for model in figure_models(panel):
        if family(model) == REFERENCE:
            continue
        for s in panel.reported_settings(model):
            x = float(medians.at[(model, s), "window_s"]) \
                if (model, s) in medians.index else math.nan
            y = summary.stat(model, s, skill_metric())
            if _finite(x) and x > 0 and _finite(y):
                points.setdefault(model, []).append((s, x, y))
    if not points:
        logger.warning("F2 skipped: no model has both a skill and a timing")
        return []
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 3.8), layout="constrained")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    for model, pts in points.items():
        if len(pts) > 1:  # ties each hollow (No weather) point to its labelled NWP point
            ax.plot([p[1] for p in pts], [p[2] for p in pts], color=colour(model),
                    alpha=0.65, linewidth=1.0, zorder=2)
        for s, x, y in pts:
            ax.plot(x, y, linestyle="none", zorder=3, **_marker_style(model, s))
    xs = [p[1] for pts in points.values() for p in pts]
    ys = [p[2] for pts in points.values() for p in pts]
    # room on both sides for direct labels as long as "Chronos-2 (fine-tuned, 336 h)"
    ax.set_xlim(min(xs) / 16, max(xs) * 12)
    lo, hi = min(min(ys), 0.0), max(max(ys), 0.0)
    pad = 0.14 * ((hi - lo) or 1.0)
    ax.set_ylim(lo - pad, hi + pad)
    _zero_line(ax, "y")
    zero_text = ax.text(0.995, 0.0, reference_name(), ha="right", va="bottom",
                        color=INK_2, fontsize=8,
                        transform=blended_transform_factory(ax.transAxes, ax.transData))
    ax.grid(axis="y")
    ax.set_xlabel("Median fit + predict time per test window (s, log scale)")
    ax.set_ylabel(f"Skill vs {reference_name()}")
    settings = {p[0] for pts in points.values() for p in pts}
    _legend_below(fig, [_family_handles(points, "marker"), _setting_handles(settings, "marker")])
    fig.canvas.draw()                # settle the layout, then freeze it for labelling
    fig.set_layout_engine("none")
    renderer = fig.canvas.get_renderer()
    labels = []
    for model, pts in points.items():
        s, x, y = next((p for p in pts if p[0] == "nwp"), pts[0])
        labels.append((model, x, y, _figure_label(panel, model)))
    labels.sort(key=lambda lab: -lab[2])
    _place_labels(ax, labels, [(p[1], p[2]) for pts in points.values() for p in pts],
                  [zero_text.get_window_extent(renderer)],
                  [[(p[1], p[2]) for p in pts] for pts in points.values() if len(pts) > 1])
    return _save(fig, figures_dir, "skill_vs_compute")


def fig_reliability(panel: Panel, summary: Summary, figures_dir: Path) -> list[Path]:
    """F3: reliability (observed P(y <= q_tau) vs tau), a panel per quantile main
    model, in ``figure_order``."""
    main = set(MAIN_MODELS)
    models = [m for m in figure_order(panel, summary) if m in main
              and any(panel.is_quantile(m, s) for s in panel.reported_settings(m))]
    if not models:
        logger.warning("F3 skipped: no quantile model")
        return []
    ncols = min(3, len(models))
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(FULL_WIDTH, 0.95 + 1.85 * nrows),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    taus = list(DEFAULT_QUANTILES)
    used_settings: set[str] = set()
    for ax, model in zip(axes.flat, models):
        ax.plot([0, 1], [0, 1], color=STATION_DOT, linewidth=0.9, linestyle=(0, (3, 2)),
                zorder=1)
        for s in panel.reported_settings(model):
            if not panel.is_quantile(model, s):
                continue
            cov = [summary.stat(model, s, f"cov_q{tau}") for tau in taus]
            ax.plot(taus, cov, color=colour(model), linestyle="-" if s == "nwp" else "--",
                    zorder=3, **_marker_style(model, s, size=4.5))
            used_settings.add(s)
        ax.set_title(_post_hoc(model), loc="left")
        if model in panel.no_covariates:
            ax.text(0.04, 0.96, "no covariates", transform=ax.transAxes, ha="left",
                    va="top", color=INK_2, fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1], labels=["0", "0.5", "1"])
        ax.set_yticks([0, 0.5, 1], labels=["0", "0.5", "1"])
        ax.grid(axis="y")
        ax.set_aspect("equal")
    for ax in axes.flat[len(models):]:
        ax.set_visible(False)
    for column in axes.T:  # a panel above an empty slot still needs its x tick labels
        [ax for ax in column if ax.get_visible()][-1].tick_params(labelbottom=True)
    # the shared x label sits on the bottom-middle panel: a figure-level supxlabel would
    # share the bottom margin with the legend and collide with it
    bottom = [ax for ax in axes[-1] if ax.get_visible()]
    bottom[len(bottom) // 2].set_xlabel("Nominal quantile level τ")
    fig.supylabel("Observed share of hours with y ≤ the τ-quantile", fontsize=8.5)
    handles = [Line2D([], [], color=INK_2, linestyle="-" if s == "nwp" else "--",
                      label=setting_label(s), **_marker_style(None, s, size=4.5))
               for s in setting_order(used_settings)]
    handles.append(Line2D([], [], color=STATION_DOT, linewidth=0.9, linestyle=(0, (3, 2)),
                          label="Perfect calibration"))
    _legend_below(fig, [handles])
    return _save(fig, figures_dir, "reliability")


def _ink_on(rgba: tuple[float, ...]) -> str:
    """White or primary ink, whichever contrasts more with the cell (WCAG)."""
    def lum(c: tuple[float, ...]) -> float:
        lin = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c[:3]]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    cell, ink = lum(rgba), lum(to_rgba(INK))
    return "white" if 1.05 / (cell + 0.05) > (cell + 0.05) / (ink + 0.05) else INK


def fig_skill_by_fold(panel: Panel, summary: Summary, folds: pd.DataFrame,
                      labels: dict[int, str], figures_dir: Path) -> list[Path]:
    """F4: heatmaps of skill over ``HEADLINE_REF`` per test window (mean over
    stations), one per setting; rows are the main models but the headline reference,
    in ``figure_order``."""
    means = fold_means(folds)
    shown = set(figure_models(panel))
    models = [m for m in figure_order(panel, summary) if m in shown]
    blocks = []
    for s in panel.settings:
        rows = [m for m in models if s in panel.reported_settings(m) and (m, s) in means.index]
        if rows:
            blocks.append((s, rows, means.loc[[(m, s) for m in rows], list(labels)].to_numpy()))
    if not blocks:
        logger.warning("F4 skipped: no fold skill to plot")
        return []
    values = np.concatenate([b[2].ravel() for b in blocks])
    vmax = max(0.05, math.ceil(np.nanmax(np.abs(values)) / 0.05) * 0.05)
    norm = Normalize(-vmax, vmax)
    n_rows = sum(len(b[1]) for b in blocks)
    # the year on a second line ("Jul" / "2022"): nine one-line labels overrun the columns
    column_labels = [f"{head}\n{year}" if head else year
                     for head, _, year in (t.rpartition(" ") for t in labels.values())]
    fig, axes = plt.subplots(len(blocks), 1, squeeze=False, layout="constrained",
                             figsize=(FULL_WIDTH, 0.85 + 0.27 * n_rows + 0.3 * len(blocks)),
                             gridspec_kw={"height_ratios": [len(b[1]) for b in blocks]})
    mesh = None
    for k, (ax, (s, rows, matrix)) in enumerate(zip(axes[:, 0], blocks)):
        mesh = ax.pcolormesh(np.ma.masked_invalid(matrix), cmap=SKILL_CMAP, norm=norm,
                             edgecolors="white", linewidth=1.5)
        for (i, j), v in np.ndenumerate(matrix):
            if np.isfinite(v):
                ax.text(j + 0.5, i + 0.5, _num(v, 2), ha="center", va="center", fontsize=8,
                        color=_ink_on(SKILL_CMAP(norm(v))))
        ax.set_ylim(len(rows), 0)
        ax.set_yticks(np.arange(len(rows)) + 0.5,
                      labels=[_figure_label(panel, m) for m in rows])
        ax.set_xticks(np.arange(len(labels)) + 0.5, labels=column_labels)
        ax.tick_params(length=0, labelbottom=k == len(blocks) - 1)
        ax.tick_params(axis="y", labelcolor=INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(setting_label(s), loc="left")
    cbar = fig.colorbar(mesh, ax=axes[:, 0].tolist(), location="right", shrink=0.85,
                        aspect=30, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=2, color=AXIS, labelcolor=INK_2)
    cbar.set_label(f"Skill vs {reference_name()}", color=INK, fontsize=8.5)
    return _save(fig, figures_dir, "skill_by_fold")


def fig_skill_by_station(panel: Panel, summary: Summary, figures_dir: Path) -> list[Path]:
    """F5: per main model (but the headline reference), each station's skill over
    ``HEADLINE_REF`` (small gray dots) and their mean (a larger marker in the family
    hue), a panel per setting."""
    metric = skill_metric()
    order = [m for m, _ in _entries(panel, summary, figure_models(panel))]
    blocks = [(s, [m for m in order if s in panel.reported_settings(m)])
              for s in panel.settings]
    blocks = [b for b in blocks if b[1]]
    if not blocks:
        logger.warning("F5 skipped: no model to plot")
        return []
    n_rows = sum(len(rows) for _, rows in blocks)
    fig, axes = plt.subplots(len(blocks), 1, sharex=True, squeeze=False, layout="constrained",
                             figsize=(FULL_WIDTH, 1.05 + 0.27 * n_rows + 0.3 * len(blocks)),
                             gridspec_kw={"height_ratios": [len(rows) for _, rows in blocks]})
    sites = panel.sites
    jitter = np.linspace(-0.16, 0.16, len(sites)) if len(sites) > 1 else np.zeros(1)
    for ax, (s, rows) in zip(axes[:, 0], blocks):
        for i, model in enumerate(rows):
            values = np.array([summary.at_site(model, s, site, metric) for site in sites])
            ok = np.isfinite(values)
            ax.plot(values[ok], i + jitter[ok], linestyle="none", marker="o", markersize=3.5,
                    markerfacecolor=STATION_DOT, markeredgewidth=0, alpha=0.9, zorder=2)
            ax.plot(summary.stat(model, s, metric), i, linestyle="none", zorder=3,
                    **_marker_style(model, s, size=7.5))
        ax.set_yticks(range(len(rows)), labels=[_figure_label(panel, m) for m in rows])
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.tick_params(axis="y", length=0, labelcolor=INK)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x")
        _zero_line(ax, "x")
        ax.set_title(setting_label(s), loc="left")
    axes[-1, 0].set_xlabel(f"Skill vs {reference_name()} (daytime)")
    station = Line2D([], [], linestyle="none", marker="o", markersize=3.5,
                     markerfacecolor=STATION_DOT, markeredgewidth=0, label="One station")
    means = [Line2D([], [], linestyle="none", label=f"Mean over stations, {setting_label(s)}",
                    **_marker_style(None, s, size=7.5)) for s, _ in blocks]
    _legend_below(fig, [[station, *means], _family_handles(order, "marker")])
    return _save(fig, figures_dir, "skill_by_station")


def fig_weather_gain(panel: Panel, summary: Summary, figures_dir: Path) -> list[Path]:
    """F6: per covariate-using main model, No-weather skill (hollow) to NWP skill
    (filled) over ``HEADLINE_REF``, in ``figure_order``."""
    rows = []
    shown, metric = set(figure_models(panel)), skill_metric()
    for model in figure_order(panel, summary):
        if model not in shown or model in panel.no_covariates \
                or not {"nwp", "none"} <= set(panel.settings):
            continue
        nwp, none = summary.stat(model, "nwp", metric), summary.stat(model, "none", metric)
        if _finite(nwp) and _finite(none):
            rows.append((model, none, nwp))
    if not rows:
        logger.warning("F6 skipped: no model ran in both weather settings")
        return []
    fig, ax = plt.subplots(figsize=(HALF_WIDTH, 1.45 + 0.3 * len(rows)), layout="constrained")
    for i, (model, none, nwp) in enumerate(rows):
        ax.plot([none, nwp], [i, i], color=AXIS, linewidth=1.2, zorder=2)
        ax.plot(none, i, linestyle="none", zorder=3, **_marker_style(model, "none"))
        ax.plot(nwp, i, linestyle="none", zorder=3, **_marker_style(model, "nwp"))
    ax.set_yticks(range(len(rows)), labels=[_post_hoc(m) for m, _, _ in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x")
    _zero_line(ax, "x")
    ax.set_xlabel(f"Skill vs {reference_name()}")
    _legend_below(fig, [_setting_handles(["none", "nwp"], "marker")[::-1],  # reading order
                        _family_handles((m for m, _, _ in rows), "marker")], max_cols=2)
    return _save(fig, figures_dir, "weather_gain")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _log_inputs(panel: Panel) -> None:
    logger.info("%d runs: %d models x %d settings x %d stations; no covariates: %s",
                len(panel.runs), len(panel.models), len(panel.settings), len(panel.sites),
                ", ".join(model_order(panel.no_covariates)) or "none")
    for model in panel.models:
        if model not in _MODEL_INFO:
            logger.warning("unknown model %r: listed last under its raw name", model)
        for s in panel.reported_settings(model):
            n = sum((model, s, site) in panel.runs for site in panel.sites)
            if n < len(panel.sites):
                logger.warning("%s/%s has %d of %d stations: its means cover only those",
                               model, s, n, len(panel.sites))
    if HEADLINE_REF not in panel.models:
        logger.warning("no %s predictions: skill over the headline reference is not "
                       "available (main tables and figures)", HEADLINE_REF)
    if CLIPER not in panel.models:
        logger.info("no CLIPER predictions: skipped the skill-vs-CLIPER columns of "
                    "master_skill_full")
    if CLIMATOLOGY not in panel.models:
        logger.info("no climatology predictions: skipped skill vs climatology "
                    "(metrics_long.csv)")


def _remove_stale(tables_dir: Path, figures_dir: Path) -> None:
    for path in [tables_dir / n for n in STALE_TABLES] + [figures_dir / n for n in STALE_FIGURES]:
        if path.exists():
            path.unlink()
            logger.info("removed obsolete %s", path)


def main(argv: Sequence[str] | None = None) -> list[Path]:
    ap = argparse.ArgumentParser(description="Build the Results-chapter tables and figures "
                                             "from the saved predictions.")
    ap.add_argument("--results", nargs="+",
                    default=[str(ROOT / "results" / "results.parquet"),
                             str(ROOT / "results" / "extra" / "results.parquet")],
                    help="results stores (timing and memory only); missing ones are skipped")
    ap.add_argument("--predictions-dir", nargs="+",
                    default=[str(ROOT / "results" / "predictions"),
                             str(ROOT / "results" / "extra" / "predictions")],
                    help="folders of per-run prediction parquets (the source of every "
                         "accuracy number); missing ones are skipped")
    ap.add_argument("--tables-dir", default=str(ROOT / "thesis" / "tables"))
    ap.add_argument("--figures-dir", default=str(ROOT / "thesis" / "figures"))
    ap.add_argument("--processed-dir", default=str(PROCESSED_DIR),
                    help="ETL cache holding the capacity sidecars (*.cap.json)")
    ap.add_argument("--include-synthetic", action="store_true",
                    help="keep site=='synthetic' runs (harness checks; excluded by default)")
    args = ap.parse_args(argv)
    setup_logging()
    logging.getLogger("fontTools").setLevel(logging.WARNING)  # PDF font subsetting chatter

    results_paths = [Path(p) for p in args.results]
    runs = check_runs(load_predictions([Path(p) for p in args.predictions_dir],
                                       args.include_synthetic, stored_hashes(results_paths)))
    panel = Panel(runs=runs, capacity=load_capacities(runs, Path(args.processed_dir)))
    panel.no_covariates = detect_no_covariates(runs)
    panel.timing = load_timing(results_paths, runs, args.include_synthetic)
    _log_inputs(panel)

    station = compute_station_metrics(panel)
    summary = Summary(station)
    folds = compute_fold_skill(panel)
    labels = fold_labels(panel)
    medians = compute_medians(panel)
    boot = Bootstrap(panel)
    significance = add_skill_differences(compute_significance(panel), boot)
    post_hoc = compute_post_hoc(panel, boot)
    diagnostics = compute_panel_diagnostics(panel)

    tables_dir, figures_dir = Path(args.tables_dir), Path(args.figures_dir)
    tables = [table_master_skill(panel, summary, boot),
              table_chronos2_configs(panel, summary, boot, medians),
              table_probabilistic(panel, summary, boot), table_compute(panel, medians),
              table_significance(panel, significance, post_hoc),
              # appendix
              table_master_skill_full(panel, summary),
              table_significance_detail(panel, significance),
              table_skill_by_fold(panel, summary, folds, labels),
              table_skill_by_station(panel, summary), table_point_rule(panel, boot),
              table_p2_robustness(panel, boot, labels),
              table_all_hours(panel, summary), table_panel_diagnostics(diagnostics)]
    written: list[Path] = []
    for table in tables:
        written += write_table(table, tables_dir)
    long_path = tables_dir / "metrics_long.csv"
    metrics_long(panel, station).to_csv(long_path, index=False)
    written.append(long_path)

    with plt.rc_context(STYLE):
        written += fig_skill_by_model(panel, summary, boot, figures_dir)
        written += fig_skill_vs_compute(panel, summary, medians, figures_dir)
        written += fig_reliability(panel, summary, figures_dir)
        written += fig_skill_by_fold(panel, summary, folds, labels, figures_dir)
        written += fig_skill_by_station(panel, summary, figures_dir)
        written += fig_weather_gain(panel, summary, figures_dir)
    _remove_stale(tables_dir, figures_dir)

    for model in panel.models:
        logger.info("skill vs %s %-38s %s", reference_name(), display_name(model), "  ".join(
            f"{setting_label(s)} {_num(summary.stat(model, s, skill_metric()))}"
            for s in panel.reported_settings(model)))
    pairs = pairwise_correlations(diagnostics.corr)
    if pairs.size:
        logger.info("mean inter-station correlation of |SP error| / C: %.3f (%d pairs)",
                    pairs.mean(), pairs.size)
    for path in written:
        logger.info("wrote %s", path)
    return written


if __name__ == "__main__":
    main()
