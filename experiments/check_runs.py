"""Checks of a batch's predictions and logs (DEFINITIVE_RUN.md sections 5-6).

The plan: the runs a config asks for, models x weather settings x stations (``--config``,
default configs/experiment.yaml: 13 x 2 x 6 = 156 runs; ``--models``, ``--weather-modes``
and ``--sites`` narrow it, as in run_panel.py). Every planned run needs one prediction file
and, with ``--logs-dir``, a last ledger status ok; a planned run without either fails, and
so does a file outside the plan. Synthetic-station files are left out with a note, as
report.py does, unless ``--sites synthetic``.

Structure, for every prediction file: rows and test windows (default: the config's),
missing forecasts (y_hat and the quantiles), quantile order, zero forecasts where the
clear-sky power is zero (the files' ``is_day`` mask is zenith < 85 deg, so its "night"
hours can carry output), and NWP vs No weather per model and station: identical for the
references, different for every model that reads the weather; a run without its partner
in the other setting fails.

``--reference-root R``: the time index, test windows, truth, ``p_clear`` and ``is_day`` of
each file equal the earlier panel's file for the same station and setting
(``R/predictions``, any model). ``--reproduce``: each model's forecasts against the earlier
run of identical params (``REPRODUCE``), max |diff| of y_hat and the quantiles in kW; a
value missing on one side only is a difference.

``--logs-dir``: the ledgers (``panel_ledger.csv``; a resumed batch writes a new folder).
Each planned run's last row that is not "skipped" gives its status and names its log
(earlier attempts are listed, not checked). In that log: warning words (the statsforecast
convergence warning of SARIMA's AutoARIMA is counted, not failed), and the "MPS driver
memory" lines: each at most ``--max-mps-gb``, one per 30 forecasts of every TimesFM-3 and
TiRex-2 run on MPS (9 in a 270-day run). Not ``check_data__*.log``, whose QC lines name
"meter errors". ``--no-ledger``: the logs of one run.py process (P1); every run log is
checked, the memory lines counted per model.

No skill score is printed: these are test days. ``--dev-skill`` (development days only,
configs/dev_check.yaml; refused for files that reach 4 Jun 2022) adds skill over CLIPER
and PICP-80 per model and setting, and the P3 gates of DEFINITIVE_RUN.md section 5.

Exit code 1 if a structural check, a ledger or log check or (with ``--dev-skill``) a P3
gate fails; a reproduction difference is reported, not an error (the LSTM and fine-tuning
on MPS may differ; section 6.5 says what follows from it).

Usage (the main run, after a resume; P3; P1):
    python experiments/check_runs.py --predictions-dir results/predictions \\
        --logs-dir results/logs results/logs_resume1 --expect-files 156 \\
        --reference-root --reproduce
    python experiments/check_runs.py --config configs/dev_check.yaml \\
        --predictions-dir results/dev_check/predictions --logs-dir results/dev_check/logs \\
        --expect-files 48 --dev-skill
    python experiments/check_runs.py --predictions-dir $P1/predictions --logs-dir $P1/logs \\
        --no-ledger --sites synthetic --weather-modes nwp --expect-folds 1 --expect-files 13
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "experiment.yaml"
#: the earlier checkout's results (panel v2 and the covariate runs), read-only
OLD_RESULTS = Path.home() / "Thesis" / "solar-pv-forecasting" / "results"

#: references: they read no weather, so NWP and No weather must give the same forecast
REFERENCES = ("persistence", "smart_persistence", "seasonal_naive", "climatology", "cliper")
#: references that copy measured power, which can be non-zero where p_clear is 0
COPIES_POWER = ("persistence", "seasonal_naive")
#: new model (file label) -> (folder under the old results root, the old run's label):
#: the earlier run with identical params (DEFINITIVE_RUN.md section 6.5)
REPRODUCE = {
    **{m: ("predictions", m) for m in ("persistence", "smart_persistence", "seasonal_naive_168h",
                                        "climatology", "cliper", "sarima", "lightgbm", "lstm",
                                        "chronos2_ft")},
    "chronos2_small": ("predictions", "chronos2"),
    "chronos2": ("covariates/predictions", "chronos2_base_ctx8192"),
}
REQUIRED = ("y_true", "y_hat", "p_clear", "is_day", "fold", "site", "model", "weather_mode")
LOG_WORDS = re.compile(r"warning|traceback|error|inductorerror|fall back to run on the cpu|"
                       r"capped at", re.IGNORECASE)
#: counted, not failed: AutoARIMA's optimiser on hard folds (statsforecast), which
#: utils.quiet_known_warnings leaves visible; the line after it echoes the source line
ALLOWED_WARNINGS = (re.compile(r"statsforecast[/\\]arima\.py:\d+: UserWarning: possible "
                               r"convergence problem"),)
WARN_ECHO = re.compile(r"^\s*warnings\.warn\($")
#: archs whose wrapper (pvfc.models.covariate_foundation) logs "MPS driver memory" on MPS
MEMORY_LOGGED = ("timesfm3", "tirex2")
#: forecasts per memory line (_NativeCovariateForecaster.MEMORY_LOG_EVERY)
MEMORY_LOG_EVERY = 30
MEM_LINE = re.compile(r"(\S+): MPS driver memory\b.*?([0-9]+(?:\.[0-9]+)?)\s*GB")
GB = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*GB")
TZ = "Asia/Hong_Kong"
#: --dev-skill refuses files that reach this local day (the first inner window's embargo
#: day; the test days start 5 Jul 2022)
DEV_END = pd.Timestamp("2022-06-04", tz=TZ)
#: P3 gates (DEFINITIVE_RUN.md section 5)
DEV_MIN_SKILL = 0.10
DEV_PICP = (0.5, 0.95)
TOL = 1e-9


def family_name(model: str) -> str:
    """The config name behind a file label (``seasonal_naive_168h`` -> ``seasonal_naive``)."""
    return "seasonal_naive" if model.startswith("seasonal_naive") else model


def slug(text: str) -> str:
    """As ``pvfc.runner.predictions_path`` writes station names into file names."""
    return re.sub(r"[^A-Za-z0-9.-]+", "_", str(text)).strip("_")


def quantile_columns(frame: pd.DataFrame) -> list[str]:
    cols = [c for c in frame.columns if c.startswith("q") and c[1:].replace(".", "", 1).isdigit()]
    return sorted(cols, key=lambda c: float(c[1:]))


def forecast_columns(frame: pd.DataFrame) -> list[str]:
    return ["y_hat", *quantile_columns(frame)]


def load_plan(config: Path, models: str | None = None, modes: str = "nwp,none",
              sites: str | None = None) -> dict:
    """The planned runs of a config, (config name, setting, station), as run_panel.py
    builds them; ``mps``: the planned models that log their MPS memory."""
    cfg = yaml.safe_load(Path(config).read_text())
    params = {m["name"]: m.get("params") or {} for m in cfg["models"]}
    names = [m.strip() for m in models.split(",") if m.strip()] if models else list(params)
    unknown = sorted(set(names) - set(params))
    if unknown:
        raise SystemExit(f"--models: not in {config}: {unknown}")
    settings = [w.strip() for w in modes.split(",") if w.strip()]
    stations = [s.strip() for s in sites.split(",")] if sites else list(cfg["data"]["panel"])
    mps = {n for n in names
           if params[n].get("arch") in MEMORY_LOGGED and str(params[n].get("device")) == "mps"}
    return {"config": Path(config), "models": names, "settings": settings, "sites": stations,
            "mps": mps, "splits": cfg["splits"],
            "runs": [(m, w, s) for m in names for w in settings for s in stations]}


def read_dir(folder: Path) -> tuple[dict[tuple[str, str, str], tuple[Path, pd.DataFrame]],
                                    list[str]]:
    """Every file of a predictions folder, keyed (model, setting, site); and the problems
    that make a file unusable (unreadable, columns missing, several runs or files per run)."""
    runs: dict[tuple[str, str, str], tuple[Path, pd.DataFrame]] = {}
    problems: list[str] = []
    for path in sorted(folder.glob("*.parquet")):
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            problems.append(f"{path.name}: unreadable ({exc})")
            continue
        missing = [c for c in REQUIRED if c not in frame.columns]
        if missing or frame.empty:
            problems.append(f"{path.name}: {'missing ' + str(missing) if missing else 'no rows'}")
            continue
        labels = [frame[c].astype(str).unique() for c in ("model", "weather_mode", "site")]
        if any(len(v) != 1 for v in labels):
            problems.append(f"{path.name}: more than one model, setting or station")
            continue
        key = (labels[0][0], labels[1][0], labels[2][0])
        if key in runs:
            problems.append(f"{path.name}: a second file for {'/'.join(key)} "
                            f"(first: {runs[key][0].name})")
            continue
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            problems.append(f"{path.name}: the index is not a tz-aware DatetimeIndex")
            continue
        runs[key] = (path, frame)
    return runs, problems


def match_plan(runs: dict, plan: dict) -> tuple[dict, list[tuple], list[str], list[str]]:
    """The files of the planned runs; the planned runs with no file; the files outside the
    plan (failures); the synthetic-station files left out."""
    cells = set(plan["runs"])
    planned: dict = {}
    seen: dict[tuple, str] = {}
    outside, synthetic = [], []
    for key, (path, frame) in sorted(runs.items()):
        cell = (family_name(key[0]), key[1], key[2])
        if cell in cells and cell not in seen:
            planned[key], seen[cell] = (path, frame), path.name
        elif cell in cells:
            outside.append(f"{path.name}: a second file for {'/'.join(cell)} (first: "
                           f"{seen[cell]})")
        elif key[2] == "synthetic":
            synthetic.append(path.name)
        else:
            outside.append(f"{path.name}: not in the plan of {plan['config'].name}")
    return planned, [c for c in plan["runs"] if c not in seen], outside, synthetic


def check_file(frame: pd.DataFrame, model: str, rows: int, folds: int) -> dict:
    """Structural numbers of one file and the list of what fails."""
    cols = forecast_columns(frame)
    values = frame[cols].to_numpy(float)
    q = frame[quantile_columns(frame)].to_numpy(float)
    dark = frame["p_clear"].to_numpy(float) == 0
    out = {
        "rows": len(frame),
        "folds": frame["fold"].nunique(),
        "missing": int((~np.isfinite(values)).sum()),
        "unsorted": int((np.diff(q, axis=1) < -TOL).sum()) if q.shape[1] > 1 else 0,
        "night_max": float(np.abs(values[dark]).max()) if dark.any() else 0.0,
        "quantiles": q.shape[1],
    }
    fail = []
    if out["rows"] != rows:
        fail.append(f"{out['rows']} rows, expected {rows}")
    if out["folds"] != folds:
        fail.append(f"{out['folds']} test windows, expected {folds}")
    elif frame.groupby("fold").size().nunique() != 1:
        fail.append("test windows of different lengths")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        fail.append("time index duplicated or unsorted")
    if out["missing"]:
        fail.append(f"{out['missing']} missing or infinite forecasts")
    if out["unsorted"]:
        fail.append(f"{out['unsorted']} quantile pairs out of order")
    if out["night_max"] > TOL and family_name(model) not in COPIES_POWER:
        fail.append(f"forecast up to {out['night_max']:.3g} kW where p_clear is 0")
    out["fail"] = fail
    return out


def forecast_gap(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """max |a - b| over y_hat and the shared quantiles (kW); nan if the indexes differ."""
    if not a.index.equals(b.index):
        return math.nan
    cols = [c for c in forecast_columns(a) if c in b.columns]
    return float(np.nanmax(np.abs(a[cols].to_numpy(float) - b[cols].to_numpy(float))))


def old_file(folder: Path, site: str, setting: str, model: str | None) -> Path | None:
    """The earlier run's file for a station and setting (one model, or any)."""
    pattern = f"{slug(site)}__{setting}__{model or '*'}__*.parquet"
    found = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return found[0] if found else None


def same(a: pd.Series, b: pd.Series) -> bool:
    """Exactly equal, NaN where NaN."""
    x, y = a.to_numpy(), b.to_numpy()
    if x.dtype.kind == "f" or y.dtype.kind == "f":
        return bool(np.array_equal(x.astype(float), y.astype(float), equal_nan=True))
    return bool(np.array_equal(x, y))


# --------------------------------------------------------------------------- #
def report_plan(folder: Path, plan: dict, n_planned: int, missing: list[tuple],
                outside: list[str], synthetic: list[str], expect_files: int | None) -> int:
    """Every planned run has one file, and no file lies outside the plan."""
    n_runs = len(plan["runs"])
    print(f"{folder}: {n_planned} of {n_runs} planned runs have a file (plan: "
          f"{len(plan['models'])} models x {len(plan['settings'])} settings x "
          f"{len(plan['sites'])} stations, {plan['config'].name})")
    failures = len(missing) + len(outside)
    if expect_files is not None and n_runs != expect_files:
        failures += 1
        print(f"  FAILED the plan has {n_runs} runs, expected {expect_files}")
    by_run: dict[tuple[str, str], list[str]] = defaultdict(list)
    for model, setting, site in missing:
        by_run[(model, setting)].append(site)
    for (model, setting), sites in by_run.items():
        print(f"  FAILED no file: {model} / {setting}: {', '.join(sites)}")
    for line in outside:
        print(f"  FAILED {line}")
    if synthetic:
        print(f"  {len(synthetic)} synthetic-station files left out, as report.py does "
              "(--sites synthetic checks them)")
    return failures


def report_structure(runs: dict, rows: int, folds: int) -> int:
    by_run: dict[tuple[str, str], list] = defaultdict(list)
    failures = 0
    print(f"\nStructure (expected {rows} rows, {folds} test windows per file):")
    print(f"  {'model':22s} {'setting':7s} {'files':>5s} {'rows':>6s} {'folds':>5s} "
          f"{'missing':>7s} {'q':>2s} {'unsorted':>8s} {'night max kW':>12s}  status")
    for (model, setting, site), (path, frame) in sorted(runs.items()):
        res = check_file(frame, model, rows, folds)
        by_run[(model, setting)].append((site, path, res))
    for (model, setting), items in sorted(by_run.items()):
        res = [r for _, _, r in items]
        bad = [(site, path, r) for site, path, r in items if r["fail"]]
        failures += len(bad)
        rows_shown = "/".join(sorted({str(r["rows"]) for r in res}))
        folds_shown = "/".join(sorted({str(r["folds"]) for r in res}))
        night = max(r["night_max"] for r in res)
        note = " (copies measured power)" if family_name(model) in COPIES_POWER and night else ""
        print(f"  {model:22s} {setting:7s} {len(items):5d} {rows_shown:>6s} {folds_shown:>5s} "
              f"{sum(r['missing'] for r in res):7d} {max(r['quantiles'] for r in res):2d} "
              f"{sum(r['unsorted'] for r in res):8d} {night:12.3g}  "
              f"{'FAILED' if bad else 'ok'}{note}")
        for site, path, r in bad:
            print(f"      {site}: {'; '.join(r['fail'])} ({path.name})")
    return failures


def report_weather_gap(runs: dict, plan: dict) -> int:
    """NWP vs No weather per planned model and station: 0 for the references, > 0 for the
    others; a run without its partner in the other setting fails."""
    if not {"nwp", "none"} <= set(plan["settings"]):
        print("\nNWP vs No weather: not compared (the plan has one weather setting)")
        return 0
    cell = {(family_name(m), w, s): item for (m, w, s), item in runs.items()}
    sites = plan["sites"]
    failures = 0
    print("\nNWP vs No weather, max |difference| of the forecasts (kW):")
    print(f"  {'model':22s} " + " ".join(f"{s[:10]:>10s}" for s in sites) + "  status")
    for model in plan["models"]:
        cells, bad = [], []
        reads_weather = model not in REFERENCES
        for site in sites:
            a, b = cell.get((model, "nwp", site)), cell.get((model, "none", site))
            if a is None or b is None:
                cells.append(f"{'-':>10s}")
                lacking = " and ".join(w for w, x in (("NWP", a), ("No-weather", b)) if x is None)
                bad.append(f"{site}: no {lacking} file")
                continue
            gap = forecast_gap(a[1], b[1])
            cells.append(f"{gap:10.3f}")
            if not np.isfinite(gap):
                bad.append(f"{site}: the two settings have different time indexes")
            elif reads_weather and gap <= TOL:
                bad.append(f"{site}: identical forecasts, but {model} reads the weather")
            elif not reads_weather and gap > TOL:
                bad.append(f"{site}: the settings differ, but {model} reads no weather")
        failures += len(bad)
        expect = "differ" if reads_weather else "equal"
        print(f"  {model:22s} " + " ".join(cells)
              + f"  {'FAILED' if bad else 'ok'} (expect {expect})")
        for line in bad:
            print(f"      {line}")
    return failures


def report_reference(runs: dict, root: Path) -> int:
    """Index, test windows, truth and the daytime mask equal the earlier panel's."""
    folder = root / "predictions"
    cache: dict[tuple[str, str], pd.DataFrame | None] = {}
    failures, checked = 0, 0
    print(f"\nAgainst the earlier panel ({folder}): time index, folds, y_true, p_clear, is_day")
    for (model, setting, site), (path, frame) in sorted(runs.items()):
        key = (site, setting)
        if key not in cache:
            ref = old_file(folder, site, setting, "smart_persistence") or \
                old_file(folder, site, setting, None)
            cache[key] = pd.read_parquet(ref) if ref is not None else None
            print(f"  {site} / {setting}: {ref.name if ref is not None else 'NO FILE'}")
        ref = cache[key]
        if ref is None:
            failures += 1
            print(f"      {path.name}: no earlier file for this station and setting")
            continue
        checked += 1
        if not frame.index.equals(ref.index):
            diff = f"{len(frame.index.difference(ref.index))} new, " \
                   f"{len(ref.index.difference(frame.index))} missing timestamps"
            failures += 1
            print(f"      {path.name}: time index differs ({diff})")
            continue
        bad = [c for c in ("fold", "y_true", "p_clear", "is_day") if not same(frame[c], ref[c])]
        if bad:
            failures += 1
            print(f"      {path.name}: differs in {', '.join(bad)}")
    print(f"  {checked} files compared, {failures} differ or have no earlier file")
    return failures


def _max(diff: np.ndarray) -> float:
    """max over the finite cells (0 if none)."""
    finite = diff[np.isfinite(diff)]
    return float(finite.max()) if finite.size else 0.0


def report_reproduction(runs: dict, root: Path) -> None:
    """Each new run against the earlier run of identical params (kW). A value missing on
    one side only counts as a differing cell (column "one NaN")."""
    print(f"\nReproduction against {root} (max |difference|, kW):")
    print(f"  {'model':22s} {'setting':7s} {'site':18s} {'y_hat':>10s} {'quantiles':>10s} "
          f"{'n differ':>8s} {'one NaN':>7s}  earlier run")
    #: per model: (max |diff|, or nan if not comparable; cells with a NaN on one side)
    per_model: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for (model, setting, site), (path, frame) in sorted(runs.items()):
        if model not in REPRODUCE:
            per_model.setdefault(model, [])
            continue
        sub, old_label = REPRODUCE[model]
        ref_path = old_file(root / sub, site, setting, old_label)
        if ref_path is None:
            print(f"  {model:22s} {setting:7s} {site:18s} {'':>10s} {'':>10s} {'':>8s} "
                  f"{'':>7s}  NO FILE {sub}/{old_label}")
            per_model[model].append((math.nan, 0))
            continue
        ref = pd.read_parquet(ref_path)
        if not frame.index.equals(ref.index):
            print(f"  {model:22s} {setting:7s} {site:18s} time index differs ({ref_path.name})")
            per_model[model].append((math.nan, 0))
            continue
        qcols = [c for c in quantile_columns(frame) if c in ref.columns]
        if set(qcols) != set(quantile_columns(frame)) | set(quantile_columns(ref)):
            print(f"  {model:22s} {setting:7s} {site:18s} quantile columns differ "
                  f"({ref_path.name})")
            per_model[model].append((math.nan, 0))
            continue
        new = frame[["y_hat", *qcols]].to_numpy(float)
        old = ref[["y_hat", *qcols]].to_numpy(float)
        diff = np.abs(new - old)
        one_nan = np.isnan(new) != np.isnan(old)
        m_hat, m_q = _max(diff[:, 0]), _max(diff[:, 1:])
        n_nan = int(one_nan.sum())
        n_diff = int((diff > 0).sum()) + n_nan
        per_model[model].append((max(m_hat, m_q), n_nan))
        shown_q = f"{m_q:10.3g}" if qcols else f"{'-':>10s}"
        print(f"  {model:22s} {setting:7s} {site:18s} {m_hat:10.3g} {shown_q} {n_diff:8d} "
              f"{n_nan:7d}  {sub}/{ref_path.name}")
    print("  per model:")
    differs = []
    for model, items in sorted(per_model.items()):
        if model not in REPRODUCE:
            print(f"    {model:22s} no earlier run of identical params")
            continue
        finite = [g for g, _ in items if np.isfinite(g)]
        n_off = len(items) - len(finite)
        n_nan = sum(1 for g, k in items if np.isfinite(g) and k)
        if items and not n_off and not n_nan and all(g == 0 for g in finite):
            verdict = "identical"
        else:
            parts = [f"DIFFERS (max {max(finite):.3g} kW)"] if finite and max(finite) > 0 else []
            parts += [f"{n_nan} files with a NaN on one side only"] if n_nan else []
            parts += [f"{n_off} files NOT COMPARABLE"] if n_off else []
            verdict = "; ".join(parts)
            differs.append(model)
        print(f"    {model:22s} {len(items):3d} files  {verdict}")
    print(f"Reproduction: {'identical' if not differs else 'differs: ' + ', '.join(differs)}")


# --------------------------------------------------------------------------- #
def read_ledgers(dirs: list[Path]) -> pd.DataFrame:
    """Every row of the ledgers in the logs folders, oldest first, with its folder."""
    frames = [pd.read_csv(d / "panel_ledger.csv", dtype=str).assign(ledger_dir=str(d))
              for d in dirs if (d / "panel_ledger.csv").exists()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("started", kind="stable")


def ledger_log(row: pd.Series) -> Path | None:
    """The run log a ledger row names: in the ledger's own folder (run_panel.py writes both
    there), else at the path as written."""
    named = Path(str(row["log"]))
    for path in (Path(row["ledger_dir"]) / named.name, named):
        if path.is_file():
            return path
    return None


def report_ledgers(ledger: pd.DataFrame, plan: dict) -> tuple[int, dict, list[pd.Series]]:
    """The last status of every planned run: its last ledger row that is not "skipped" (a
    resumed batch's later row wins). Returns the failures, that row per planned run, and
    the earlier attempts it supersedes."""
    key = ["model", "weather_mode", "site"]
    ran = ledger[ledger["status"] != "skipped"]
    current: dict[tuple, pd.Series] = {}
    earlier: list[pd.Series] = []
    for cell, group in ran.groupby(key, sort=False):
        current[tuple(cell)] = group.iloc[-1]
        earlier += [row for _, row in group.iloc[:-1].iterrows()]
    listed = set(ledger[key].itertuples(index=False, name=None))
    counts: Counter = Counter()
    lines = []
    for cell in plan["runs"]:
        row = current.get(cell)
        if row is None:
            counts["missing"] += 1
            why = "only skipped rows (its batch's logs folder not given?)" if cell in listed \
                else "no ledger row"
            lines.append(f"    {why}: {' / '.join(cell)}")
        else:
            counts[row["status"]] += 1
            if row["status"] != "ok":
                lines.append(f"    {row['status']}: {' / '.join(cell)} ({row['log']})")
    print(f"\nLedgers (panel_ledger.csv in {len(ledger['ledger_dir'].unique())} folders), "
          "last status per planned run:")
    print("  " + ", ".join(f"{k} {v}" for k, v in counts.items())
          + f" (of {len(plan['runs'])} planned)")
    for line in lines:
        print(line)
    cells = set(plan["runs"])
    outside = [c for c in current if c not in cells]
    if outside:
        print(f"  {len(outside)} ledger runs outside the plan, not checked")
    return (len(lines), {c: r for c, r in current.items() if c in cells},
            [r for r in earlier if tuple(r[key]) in cells])


def scan_log(path: Path) -> dict:
    """A run log's warning-word lines (not the allowed ones), its allowed warnings, the
    MPS memory values (GB) per model name, and whether run.py skipped a model there."""
    hits: list[str] = []
    mem: dict[str, list[float]] = defaultdict(list)
    allowed, after_allowed, skip = 0, False, False
    for line in path.read_text(errors="replace").splitlines():
        found = MEM_LINE.search(line)
        if found:
            mem[found.group(1)].append(float(found.group(2)))
        elif "MPS driver memory" in line and GB.findall(line):
            mem["?"].append(float(GB.findall(line)[-1]))
        skip = skip or line.startswith("SKIP ")
        if any(p.search(line) for p in ALLOWED_WARNINGS):
            allowed, after_allowed = allowed + 1, True
            continue
        if after_allowed and WARN_ECHO.match(line):
            after_allowed = False
            continue
        after_allowed = False
        if LOG_WORDS.search(line):
            hits.append(line.strip())
    return {"hits": hits, "allowed": allowed, "mem": mem, "skip": skip}


def report_logs(dirs: list[Path], current: dict | None, earlier: list, plan: dict,
                per_run: int, max_gb: float) -> int:
    """Warning words and the MPS memory lines in the run logs. With the ledgers'
    ``current`` rows: the log of each planned run, and ``per_run`` memory lines in every ok
    run of a model in ``plan["mps"]``; without (``None``): every run log, and ``per_run``
    lines per planned run of those models, counted over all logs."""
    failures = 0
    all_logs = sorted(p for d in dirs for p in d.glob("*__*.log")
                      if not p.name.startswith("check_data__"))
    checked: dict[Path, tuple | None] = {}
    print("\nRun logs (not check_data__*.log):")
    if current is None:
        checked = {p: None for p in all_logs}
        print(f"  no ledger: every run log of {', '.join(map(str, dirs))} ({len(all_logs)})")
    else:
        for cell, row in sorted(current.items()):
            path = ledger_log(row)
            if path is None:
                failures += 1
                print(f"    {' / '.join(cell)}: no log {row['log']}")
            else:
                checked[path] = cell
        print(f"  the logs of the last attempt of {len(checked)} planned runs "
              f"({len(all_logs)} run logs in {len(dirs)} folders)")
    peak, peak_file, n_mem, n_allowed, n_allowed_logs = -1.0, "", 0, 0, 0
    per_model: Counter = Counter()
    for path, cell in checked.items():
        res = scan_log(path)
        bad = []
        if res["hits"]:
            bad.append(f"{len(res['hits'])} lines, e.g. {res['hits'][0][:160]}")
            bad += [f"    {line[:160]}" for line in res["hits"][1:3]]
        for name, values in res["mem"].items():
            per_model[name] += len(values)
            n_mem += len(values)
            if max(values) > peak:
                peak, peak_file = max(values), path.name
        if cell is not None and cell[0] in plan["mps"] and current[cell]["status"] == "ok":
            n = len(res["mem"].get(cell[0], []))
            if n < per_run:
                hint = " (the log holds a SKIP: a later batch reused this logs folder)" \
                    if res["skip"] else ""
                bad.append(f"{n} MPS driver memory lines for {cell[0]}, expected {per_run}"
                           + hint)
        n_allowed += res["allowed"]
        n_allowed_logs += bool(res["allowed"])
        if bad:
            failures += 1
            print(f"    {path.name}: {bad[0]}")
            for line in bad[1:]:
                print(f"        {line}")
    if current is None:
        for model in sorted(plan["mps"]):
            n_runs = sum(1 for m, _, _ in plan["runs"] if m == model)
            if per_model[model] < per_run * n_runs:
                failures += 1
                print(f"    {model}: {per_model[model]} MPS driver memory lines, expected "
                      f"{per_run * n_runs} ({n_runs} runs x {per_run})")
    if n_mem:
        over = peak > max_gb
        failures += int(over)
        print(f"  MPS driver memory: {n_mem} lines ("
              + ", ".join(f"{k} {v}" for k, v in sorted(per_model.items()))
              + f"), max {peak:.2f} GB ({peak_file}) {'ABOVE' if over else 'within'} "
              f"{max_gb:g} GB")
    else:
        print("  MPS driver memory: no lines"
              + (f" (expected {per_run} per run of {', '.join(sorted(plan['mps']))})"
                 if plan["mps"] and per_run else ""))
    if n_allowed:
        print(f"  allowed, not failed: {n_allowed} statsforecast convergence warnings "
              f"(SARIMA's AutoARIMA) in {n_allowed_logs} logs")
    if earlier:
        print(f"  {len(earlier)} earlier attempts, superseded by a later ledger row (not "
              "checked):")
        for row in earlier:
            print(f"    {row['status']} {row['model']} / {row['weather_mode']} / {row['site']}"
                  f" ({row['started']}): {row['log']}")
    print(f"  {len(checked)} run logs checked, {failures} failures")
    return failures


def report_run_logs(dirs: list[Path], plan: dict, per_run: int, max_gb: float,
                    no_ledger: bool) -> int:
    """The ledgers, then the logs of the last attempt of each planned run."""
    if no_ledger:
        return report_logs(dirs, None, [], plan, per_run, max_gb)
    ledger = read_ledgers(dirs)
    if ledger.empty:
        print(f"\nLedgers: FAILED, no panel_ledger.csv rows in {', '.join(map(str, dirs))} "
              "(--no-ledger for the logs of one run.py process)")
        return 1 + report_logs(dirs, None, [], plan, per_run, max_gb)
    failures, current, earlier = report_ledgers(ledger, plan)
    return failures + report_logs(dirs, current, earlier, plan, per_run, max_gb)


# --------------------------------------------------------------------------- #
def _report_module():
    """experiments/report.py, or None if it does not import (it is a script, not a package)."""
    name = "_check_runs_report"
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / "report.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up there
    try:
        spec.loader.exec_module(module)
        return module if hasattr(module, "run_metrics") else None
    except Exception as exc:  # noqa: BLE001 - fall back to the direct computation
        print(f"  (report.py does not import: {exc}; direct computation)")
        return None


def _direct_metrics(frame: pd.DataFrame, ref: pd.DataFrame | None) -> dict[str, float]:
    """As report.run_metrics: daytime hours with a finite truth (and a CLIPER forecast)."""
    mask = frame["is_day"].to_numpy(bool) & np.isfinite(frame["y_true"].to_numpy(float))
    y = frame["y_true"].to_numpy(float)[mask]
    out = {"skill_cliper": math.nan, "picp_80": math.nan}
    if ref is not None:
        r = ref["y_point"].reindex(frame.index).to_numpy(float)[mask]
        ok = np.isfinite(r)
        err = frame["y_point"].to_numpy(float)[mask][ok] - y[ok]
        rmse_ref = np.sqrt(np.mean((r[ok] - y[ok]) ** 2))
        out["skill_cliper"] = float(1 - np.sqrt(np.mean(err ** 2)) / rmse_ref)
    if {"q0.1", "q0.9"} <= set(frame.columns):
        lo, hi = frame["q0.1"].to_numpy(float)[mask], frame["q0.9"].to_numpy(float)[mask]
        out["picp_80"] = float(np.mean((y >= lo) & (y <= hi)))
    return out


def dev_metrics(runs: dict) -> pd.DataFrame:
    """Skill over CLIPER and PICP-80 per run on the daytime hours, by report.py's
    ``run_metrics`` (point forecast = the quantile mean; capacity from the ETL sidecars)."""
    from pvfc.evaluation import point_forecast_frame

    report = _report_module()
    frames = {}
    for key, (_, frame) in runs.items():
        frame = frame.sort_index().copy()
        frame["y_point"] = point_forecast_frame(frame)
        frames[key] = frame
    capacity = report.load_capacities(frames) if report is not None else {}
    rows = []
    for (model, setting, site), frame in frames.items():
        ref = frames.get(("cliper", setting, site))
        if ref is None:
            ref = next((f for (m, _, s), f in frames.items() if m == "cliper" and s == site), None)
        if report is not None:
            refs = {"cliper": ref["y_point"]} if ref is not None else {}
            m = report.run_metrics(frame, capacity[site], "daytime", refs)
        else:
            m = _direct_metrics(frame, ref)
        rows.append({"model": model, "setting": setting, "site": site,
                     "skill_cliper": m.get("skill_cliper", math.nan),
                     "picp_80": m.get("picp_80", math.nan)})
    return pd.DataFrame(rows)


def report_dev_skill(runs: dict) -> int:
    """Development days only: skill over CLIPER, PICP-80 and the P3 gates."""
    late = [p.name for p, f in runs.values() if f.index.max() >= DEV_END]
    if late:
        raise SystemExit(f"--dev-skill is for development days only (before {DEV_END.date()}); "
                         f"{len(late)} files reach later days, e.g. {late[0]}")
    table = dev_metrics(runs)
    stats = table.groupby(["model", "setting"]).agg(
        skill=("skill_cliper", "mean"), skill_min=("skill_cliper", "min"),
        picp=("picp_80", "mean"), n=("site", "count"))
    print("\nDevelopment days: skill over CLIPER (daytime RMSE; station mean [min]) and PICP-80")
    for (model, setting), r in stats.iterrows():
        print(f"  {model:22s} {setting:7s} {r.skill:7.3f} [{r.skill_min:7.3f}]  "
              f"PICP-80 {r.picp:6.3f}  ({int(r.n)} stations)")
    failures = 0
    print(f"P3 gates (NWP skill > {DEV_MIN_SKILL} and above No weather; PICP-80 in "
          f"{DEV_PICP[0]}-{DEV_PICP[1]}):")
    for model in sorted({m for m, _ in stats.index}):
        if family_name(model) in REFERENCES:
            continue
        bad = []
        nwp = stats.loc[(model, "nwp")] if (model, "nwp") in stats.index else None
        none = stats.loc[(model, "none")] if (model, "none") in stats.index else None
        if nwp is None:
            bad.append("no NWP run")
        else:
            if not nwp.skill > DEV_MIN_SKILL:
                bad.append(f"NWP skill {nwp.skill:.3f}")
            if none is not None and not nwp.skill > none.skill:
                bad.append(f"NWP skill not above No weather ({none.skill:.3f})")
        for setting, r in (("nwp", nwp), ("none", none)):
            if r is not None and np.isfinite(r.picp) and not DEV_PICP[0] <= r.picp <= DEV_PICP[1]:
                bad.append(f"PICP-80 {r.picp:.3f} ({setting})")
        failures += int(bool(bad))
        print(f"  {model:22s} {'FAILED: ' + '; '.join(bad) if bad else 'ok'}")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--predictions-dir", required=True)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="the batch's config: its models, data.panel and splits make the plan")
    ap.add_argument("--models", default=None, help="comma-separated (default: every config model)")
    ap.add_argument("--weather-modes", default="nwp,none", help="comma-separated")
    ap.add_argument("--sites", default=None, help="comma-separated (default: data.panel)")
    ap.add_argument("--logs-dir", action="extend", nargs="+", default=[],
                    help="one or more logs folders (e.g. results/logs results/logs_resume1)")
    ap.add_argument("--no-ledger", action="store_true",
                    help="the logs of one run.py process (P1): no ledger, every run log")
    ap.add_argument("--expect-rows", type=int, default=None,
                    help="default: splits.test_days x 24 x the test windows")
    ap.add_argument("--expect-folds", type=int, default=None, help="default: splits.max_folds")
    ap.add_argument("--expect-files", type=int, default=None,
                    help="the planned runs, each with one file (e.g. 156)")
    ap.add_argument("--reference-root", nargs="?", const=str(OLD_RESULTS), default=None,
                    help=f"earlier results root (alone: {OLD_RESULTS})")
    ap.add_argument("--reproduce", action="store_true",
                    help="compare forecasts with the earlier runs of identical params")
    ap.add_argument("--max-mps-gb", type=float, default=4.0)
    ap.add_argument("--dev-skill", action="store_true",
                    help="development days only: skill over CLIPER, PICP-80, the P3 gates")
    args = ap.parse_args()

    folder = Path(args.predictions_dir)
    if not folder.is_dir():
        raise SystemExit(f"no predictions folder {folder}")
    plan = load_plan(Path(args.config), args.models, args.weather_modes, args.sites)
    folds = args.expect_folds or int(plan["splits"]["max_folds"])
    rows = args.expect_rows or int(plan["splits"]["test_days"]) * 24 * folds
    found, problems = read_dir(folder)
    runs, missing, outside, synthetic = match_plan(found, plan)
    failures = report_plan(folder, plan, len(runs), missing, outside, synthetic,
                           args.expect_files) + len(problems)
    for line in problems:
        print(f"  FAILED {line}")
    if not runs:
        sys.exit(1)

    failures += report_structure(runs, rows, folds)
    failures += report_weather_gap(runs, plan)
    root = Path(args.reference_root or OLD_RESULTS).expanduser()
    if args.reference_root:
        failures += report_reference(runs, root)
    if args.reproduce:
        report_reproduction(runs, root)
    log_dirs = [Path(d) for d in args.logs_dir]
    per_run = rows // 24 // MEMORY_LOG_EVERY
    log_failures = report_run_logs(log_dirs, plan, per_run, args.max_mps_gb, args.no_ledger) \
        if log_dirs else 0
    gate_failures = report_dev_skill(runs) if args.dev_skill else 0

    print(f"\nStructure: {'ok' if not failures else f'FAILED ({failures})'}"
          + (f"; ledgers and logs: {'ok' if not log_failures else f'FAILED ({log_failures})'}"
             if log_dirs else "")
          + (f"; P3 gates: {'ok' if not gate_failures else f'FAILED ({gate_failures})'}"
             if args.dev_skill else ""))
    sys.exit(1 if failures or log_failures or gate_failures else 0)


if __name__ == "__main__":
    main()
