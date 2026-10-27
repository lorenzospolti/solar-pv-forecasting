"""Overnight batch: every panel station x weather setting x model, one process per run.

Each run is ``experiments/run.py --site S --weather-mode W --models M --skip-done``
in its own process: peak-memory numbers stay per model, and one crash cannot take
down the night. Every run logs to ``results/logs/`` and appends its rows to the
shared results store when it finishes, and ``--skip-done`` makes the batch
resumable: start the same command again and it continues where it stopped.

The order is model-major: references and the zero-shot foundation models first
(Chronos-B, Chronos-S, TiRex-2, TimesFM-3: a failure of the two new models on real
data shows in the first hours), then LightGBM (the key purpose-trained comparator),
SARIMA, LSTM and Chronos-2 fine-tuning. A night that ends early still leaves
complete model x setting blocks across ALL stations (comparable rows for the master
table) instead of a few stations with every model. The default config is the
definitive run (DEFINITIVE_RUN.md): 13 models x 2 settings x 6 stations = 156 runs.

Before the first run, ``check_data.py`` is run for every station (pre-flight, and
it builds the nwp/none caches); a failing station stops the batch.

Usage:
    python experiments/run_panel.py --dry-run                  # plan + rough hours
    python experiments/run_panel.py                            # everything, resumable
    python experiments/run_panel.py --logs-dir results/logs_resume1   # resume, keeping the logs
    python experiments/run_panel.py --models lightgbm,sarima --weather-modes nwp
    python experiments/run_panel.py --sites "SQ1,LSK North" --max-folds 1   # quick check

On macOS the batch keeps the machine awake with ``caffeinate`` while it runs; keep
the Mac on power with the lid open (closing the lid still sleeps it).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

#: model-major run order; models not listed run last, in config order
RUN_ORDER = ["persistence", "smart_persistence", "seasonal_naive", "climatology", "cliper",
             "chronos2", "chronos2_small", "tirex2", "timesfm3", "lightgbm", "sarima", "lstm",
             "chronos2_ft"]

#: rough wall-clock seconds per fold on the M4, from the 1 Oct smoke test (CYT (Station 2),
#: first fold, ~400 training days); later folds train on up to ~2.5 years, hence GROWTH.
#: chronos2 (Chronos-2-base, up to 8,192 h), tirex2 and timesfm3: estimates of 14 Oct
#: (DEFINITIVE_RUN.md section 2); timesfm and tirex: the TimesFM-2.5 and TiRex rows of
#: configs/extras.yaml and configs/covariates.yaml
SECONDS_PER_FOLD = {"persistence": 1, "smart_persistence": 1, "seasonal_naive": 1,
                    "climatology": 1, "cliper": 1, "chronos2": 65, "chronos2_small": 12,
                    "tirex2": 20, "timesfm3": 100, "timesfm": 18, "tirex": 16,
                    "lstm": 127, "sarima": 170, "chronos2_ft": 126, "lightgbm": 250}
GROWTH = 1.5


def build_plan(models: list[str], weather_modes: list[str], sites: list[str]) -> list[tuple]:
    """(model, weather_mode, site) runs, model-major in ``RUN_ORDER``."""
    rank = {m: i for i, m in enumerate(RUN_ORDER)}
    ordered = sorted(models, key=lambda m: (rank.get(m, len(rank)), models.index(m)))
    return [(m, w, s) for m in ordered for w in weather_modes for s in sites]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", text).strip("_")


def _keep_awake() -> subprocess.Popen | None:
    """caffeinate -i (no idle sleep) -s (no system sleep on AC) for as long as we live."""
    if sys.platform != "darwin" or not shutil.which("caffeinate"):
        return None
    return subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"))
    ap.add_argument("--sites", default=None, help="comma-separated (default: data.panel)")
    ap.add_argument("--weather-modes", default="nwp,none", help="comma-separated")
    ap.add_argument("--models", default=None, help="comma-separated (default: every config model)")
    ap.add_argument("--max-folds", type=int, default=None, help="passed to run.py")
    ap.add_argument("--results", default=str(ROOT / "results" / "results.parquet"))
    ap.add_argument("--predictions-dir", default=str(ROOT / "results" / "predictions"))
    ap.add_argument("--logs-dir", default=str(ROOT / "results" / "logs"))
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--no-preflight", action="store_true", help="skip check_data.py per station")
    ap.add_argument("--no-caffeinate", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    sites = [s.strip() for s in args.sites.split(",")] if args.sites else list(cfg["data"]["panel"])
    modes = [w.strip() for w in args.weather_modes.split(",") if w.strip()]
    names = [m["name"] for m in cfg["models"]]
    models = [m.strip() for m in args.models.split(",")] if args.models else names
    unknown = sorted(set(models) - set(names))
    if unknown:
        raise SystemExit(f"--models: not in {args.config}: {unknown}")
    plan = build_plan(models, modes, sites)

    folds = args.max_folds or cfg["splits"].get("max_folds") or 9
    est_h = sum(SECONDS_PER_FOLD.get(m, 60) * GROWTH * folds for m, _, _ in plan) / 3600
    print(f"{len(plan)} runs: {len(models)} models x {len(modes)} weather settings x "
          f"{len(sites)} stations, {folds} folds each; rough total {est_h:.1f} h "
          f"(1 Oct smoke-test timings, 14 Oct estimates for the new models; finished runs "
          "are skipped)")
    if args.dry_run:
        for i, (m, w, s) in enumerate(plan, 1):
            print(f"  {i:3d}. {m:18s} {w:5s} {s}")
        return

    awake = None if args.no_caffeinate else _keep_awake()
    logs = Path(args.logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    if not args.no_preflight:
        for s in sites:
            log = logs / f"check_data__{_slug(s)}.log"
            with open(log, "w") as fh:
                rc = subprocess.call([py, str(ROOT / "experiments" / "check_data.py"),
                                      "--config", args.config, "--site", s],
                                     stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
            print(f"pre-flight {s}: {'ok' if rc == 0 else 'FAILED, see ' + str(log)}")
            if rc != 0:
                raise SystemExit("pre-flight failed: fix the station or drop it from --sites")

    ledger = logs / "panel_ledger.csv"
    new_ledger = not ledger.exists()
    failed: list[str] = []
    t_batch = time.time()
    with open(ledger, "a", newline="") as lf:
        writer = csv.writer(lf)
        if new_ledger:
            writer.writerow(["started", "model", "weather_mode", "site", "status",
                             "exit_code", "wall_s", "log"])
        for i, (m, w, s) in enumerate(plan, 1):
            log = logs / f"{_slug(s)}__{w}__{m}.log"
            cmd = [py, str(ROOT / "experiments" / "run.py"), "--config", args.config,
                   "--site", s, "--weather-mode", w, "--models", m, "--skip-done",
                   "--results", args.results, "--predictions-dir", args.predictions_dir]
            if args.max_folds:
                cmd += ["--max-folds", str(args.max_folds)]
            started = datetime.now().isoformat(timespec="seconds")
            t0 = time.time()
            with open(log, "w") as fh:
                rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
            wall = time.time() - t0
            skipped = rc == 0 and "SKIP " in log.read_text(errors="replace")
            status = "skipped" if skipped else ("ok" if rc == 0 else "FAILED")
            if rc != 0:
                failed.append(f"{m} / {w} / {s} (log: {log})")
            writer.writerow([started, m, w, s, status, rc, round(wall, 1), str(log)])
            lf.flush()
            print(f"[{i:3d}/{len(plan)}] {status:7s} {m:18s} {w:5s} {s:18s} "
                  f"{wall:7.0f} s  (batch {(time.time() - t_batch) / 3600:.2f} h)", flush=True)

    print(f"\nDone in {(time.time() - t_batch) / 3600:.2f} h; ledger: {ledger}")
    if failed:
        print("FAILED runs (rerun the same command to retry only these):")
        for f in failed:
            print("  " + f)
    if awake is not None:
        awake.terminate()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
