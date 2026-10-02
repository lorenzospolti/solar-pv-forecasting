# Solar PV Day-Ahead Forecasting — Benchmark Harness

Bachelor's thesis (Università di Milano-Bicocca, Sept 2026). **Variant B:** do
zero-shot & fine-tuned time-series *foundation models* (Chronos-2, Moirai-2,
TimesFM-2.5, TinyTimeMixer) match or beat purpose-trained classical/ML/deep
models (smart persistence, SARIMA, LightGBM, LSTM) for **day-ahead hourly PV
power** forecasting **on consumer hardware (M4, 16 GB)** — and at what compute cost?

Evaluation follows the solar-forecasting verification canon: model the
**clear-sky index**, score by **skill over smart persistence**, and report
**probabilistic calibration** (CRPS / pinball / reliability).

## Quickstart

```bash
mamba env create -f environment.yml        # or: make env
conda activate pv
pip install -e ".[forecast,dev]"           # or: make install
make check                                 # toolchain smoke test (MPS, pvlib, lightgbm)
make test                                  # harness unit/smoke tests
make smoke                                 # end-to-end run on a SYNTHETIC station (no download)
```

The synthetic-station path lets the whole harness run before any data is
downloaded. For real runs, download the HKUST dataset (Dryad
`doi:10.5061/dryad.m37pvmd99`) into `data/raw/` and implement the ETL in
[`src/pvfc/data.py`](src/pvfc/data.py) (Week 3).

## Architecture (one shared harness)

Every model is a `Forecaster`; the `Runner` loops `(model × fold × site)` over a
single causal `Splitter` and `Evaluator`, so all models get identical splits,
masking, and metrics by construction.

| Module | Role |
|---|---|
| `models/base.py` | `Forecaster` ABC (`fit` / `predict(history, future)`) |
| `models/baselines.py` | persistence, **smart persistence**, seasonal-naive, climatology, CLIPER (Yang 2019) |
| `models/darts_wrapper.py` | adapters for SARIMA/LightGBM/LSTM/foundation models (darts) |
| `splits.py` | expanding-window blocked walk-forward (whole-day blocks, embargo) |
| `normalize.py` | clear-sky-index transform + train-only scaling |
| `clearsky.py` | pvlib clear-sky GHI → empirical clear-sky **power** template |
| `evaluation.py` | RMSE/MAE/nMAE, skill score, pinball/CRPS/PICP/reliability |
| `runner.py` | the loop; writes tidy rows to `results/results.parquet` (config-hash keyed) |

Full rationale, schedule, risks, and reading list: see the thesis plan at
`~/.claude/plans/analyze-thoroughly-this-project-kind-bear.md`.

## Status
**Protocol audit fixes applied (10 Jun 2026, `v0.4-protocol-fixes` — see
`CRITICAL_REVIEW.md`):** Open-Meteo backward-averaged radiation realigned (−1 h);
day blocks & issue times now on **local (HKT) days** (`splits.day_tz`); the clear-sky
template is fit on the **first 365 days only** (no test-period leakage in `k`);
daytime data gaps stay NaN instead of fake zeros; the runner strips `power`/`k` from
the future frame, fits **once per fold** and scores both masks, and reports
`fit_s`/`predict_s`/`peak_rss_mb` plus per-quantile coverage (`cov_q*`) for
reliability diagrams; nwp frames carry **only** forecast weather; test folds now
span May 2022–Nov 2023 (10 folds, 60-day stride). CYT processed caches regenerated.

Harness spine + baselines + **real HKUST ETL** (per-station nameplate capacity from
the Brick `.ttl` metadata) + **NWP weather feed** (Open-Meteo) + **SARIMA (AutoARIMA),
LightGBM, and LSTM** wrappers are implemented. **Foundation-model wrappers (Chronos-2,
TimesFM-2.5, TiRex) are now wired** in `darts_wrapper.py` — zero-shot and fine-tuned
(via an `arch`/label split; `make report` builds tables + figures from results).
The forecast extra pins **`darts[torch]>=0.44`** (fine-tuning needs >=0.42, TiRex
>=0.44) plus `tirex-ts`. The old PyPI name `u8darts` stopped at 0.41.0 (Feb 2026):
never pin it. `make checkpoints` pre-downloads the three foundation checkpoints.

**ETL v2 + protocol fixes (Oct 2026, see the 28 Sep review in `~/Thesis/thesis-review-2026-09-28`):**
- hourly station files label the *end* of each hour; they are relabelled to interval
  start (15-min files already are), and clear-sky GHI / sun position are evaluated per
  interval (hour mean, midpoint) instead of at the instant HH:00;
- the clear-sky template is binned by **sun position** (declination x hour angle), not
  calendar month: no holes when a month is missing, no month-boundary steps;
- raw power above 1.1 x nameplate is removed before resampling (`data/processed/qc_log.csv`);
- processed caches carry `ETL_VERSION` in their name; `CYT (Station 1)` (timestamps
  offset 30-45 min) and `Zone A2` (output regime change in May 2023) are excluded
  (`pvfc.data.EXCLUDED_STATIONS`); the development station is `CYT (Station 2)`;
- weather modes `nwp` | `none` (no weather covariates) | `oracle` (tower; its
  pyranometer drifts, so diagnostics only);
- folds are calendar-anchored (`first_test_day: 2022-07-05`, 9 x 30-day windows), so
  every station is scored on the same days;
- LightGBM sees the forecast covariates of the hour it predicts (future lags 0..23)
  and yesterday's whole profile (lags 1..24) — v0.5 only saw midnight values;
- persistence baselines fall back to the same hour of an earlier day after a gap;
- Chronos-2 fine-tuning: lr 1e-5 (AutoGluon's default fine-tuning rate, which AutoGluon
  applies with LoRA; we fine-tune fully; darts' 1e-3 destroys the pretrained weights),
  5 epochs on one training window per day (`fit_stride: 24`);
- TiRex is licence-gated (NXAI Community License): set `accept_license: true` in its
  config entry once you have reviewed the terms;
- `run.py` saves rows after each model, writes per-timestamp predictions to
  `results/predictions/`, and takes `--models` / `--max-folds`.

**Scoring and ETL v3/v4 (2 Oct 2026, panel v2):**
- point metrics of probabilistic models use the **mean of the nine quantiles** (scoring
  v2, `pvfc.evaluation.point_forecast`: squared error is minimised by the mean, not the
  median); every results row records `scoring_version`;
- `make rescore` (`experiments/rescore.py`) rebuilds every metric row of the store from
  the saved predictions under the current rules (timing, RSS and env carried over;
  `skill_cliper` added); `--point native` reproduces scoring v1 exactly;
- new reference baselines: **climatology** and **CLIPER** (Yang 2019; the standard of
  reference recommended by Yang et al. 2020), reported next to smart persistence;
- `make report` builds the thesis tables and figures from the predictions: pooled
  per-station skill, calibration, compute, and day-level Diebold-Mariano tests for 11
  named comparisons with Holm adjustment;
- every fit starts from a **fresh model** (panel v1's LSTM and fine-tuned Chronos-2
  kept training across folds); zero-shot checkpoints are loaded once;
- **ETL v3** (hashed into every run): negative meter errors (< -0.05 x nameplate) and
  impossible night readings are missing; only genuine gaps of <= 2 h are bridged;
  **outage hours** (runs of >= 3 h of zero output under a high sun, `pvfc.qc`) are
  missing data and the clear-sky template is refit without them; the template's hour
  angle is wrapped to [-180, 180); Open-Meteo cache years are always complete (SQ15
  lacked weather before 4 May 2021);
- **ETL v4** (`data.etl_version: 4`): mean power from the `power(W)` column, as in v2.
  v3 had moved to the energy column, but at the 15-min stations that column sits about
  3% above the inverters' AC records (the power column matches them to within 0.3%)
  and reads zero in 2-23% of the quarter hours with 20 W-1 kW of output: that is how
  three very dark storm spells (May, June, Sep 2023) passed for outages at LSK North,
  SQ567 and UG Hall6 (SQ1 and the hourly stations read about zero in those storms, so
  the outage rule still removes 18 of those hours there);
- missing hours in a darts model's input are filled like the references fill theirs,
  from the same hour of an earlier day (`pvfc.normalize.fill_index_gaps`), not with a
  forward fill that turned every outage day into a day without output;
- panel v1 (ETL v2, warm-started models) is archived in `results/archive/panel_v1/`,
  the stopped first attempt at panel v2 (ETL v3, forward fill) in
  `results/archive/panel_v2_etl3_stopped/`.

### Foundation-model roster

| Arch | Library | Zero-shot label | Fine-tuned label | Dep |
|---|---|---|---|---|
| chronos2 | darts | `chronos2` | `chronos2_ft` | `darts[torch]>=0.44` |
| timesfm | darts | `timesfm` | — (out of scope) | `darts[torch]>=0.44` |
| tirex | darts | `tirex` | — | `darts[torch]>=0.44`, `tirex-ts` |
| ttm | **granite-tsfm** (optional stretch) | `ttm` | — | `granite-tsfm[notebooks]>=0.2.16` |
| moirai2 | **uni2ts** (out of scope) | `moirai2` | — | `uni2ts>=1.2` |

The external wrappers (TTM, Moirai-2) have no fine-tuning loop: `finetune: true`
raises instead of producing a zero-shot forecast under an `_ft` label. To try TTM
zero-shot: `python -m pip install 'granite-tsfm[notebooks]>=0.2.16'`, then uncomment
its entry in `configs/experiment.yaml` (keep `quantile: false`: TTM is deterministic).

To run on real data: download `Dataset.zip` from
[Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.m37pvmd99) into `data/raw/`
(the page guards automated downloads, so use a browser), then `make data` to
extract + inspect, check a station's ETL, and run the model ladder on it, e.g.:

```bash
python experiments/check_data.py --site "CYT (Station 2)"
python experiments/run.py --site "CYT (Station 2)" --weather-mode nwp
python experiments/run.py --site "CYT (Station 2)" --weather-mode none --models lightgbm,chronos2
python experiments/run.py --max-folds 1 --results results/smoke.parquet   # quick one-fold check
make report   # -> thesis/tables/ + thesis/figures/
```

### Overnight panel runs

`experiments/run_panel.py` (`make panel`) runs every panel station x weather setting
x model as a separate `run.py` process, model-major (references and zero-shot models
first, then LightGBM, SARIMA, LSTM, Chronos-2 fine-tuning), after a `check_data.py`
pre-flight per station. Each run logs to `results/logs/`, a ledger records status and
wall time, and `--skip-done` makes the batch resumable: start the same command again
and finished runs are skipped. It keeps the Mac awake with `caffeinate` (stay on power,
lid open). The full 120-run panel took 19.3 h on the M4 (1-2 Oct 2026).

```bash
python experiments/run_panel.py --dry-run                         # plan + rough hours
nohup python experiments/run_panel.py > results/logs/panel.out 2>&1 &   # survives closing the terminal
python experiments/run_panel.py --models lightgbm --weather-modes nwp   # a subset
```

## Third-party models and licences

The benchmark downloads pre-trained checkpoints at run time; none are redistributed
in this repository.

- **Chronos-2** (`autogluon/chronos-2-small`, Amazon) and **TimesFM-2.5**
  (`google/timesfm-2.5-200m-pytorch`): Apache-2.0.
- **TiRex** (`NX-AI/TiRex`): NXAI Community License, accepted in
  `configs/experiment.yaml` (`accept_license: true`).
  Built with technology from NXAI. This product includes materials developed at NXAI
  that are licensed under the NXAI Community License, Copyright © NXAI GmbH, All Rights
  Reserved. Licence text: <https://github.com/NX-AI/tirex/blob/main/LICENSE>.
- **HKUST rooftop PV dataset** (Lin et al., *Scientific Data* 12:63, 2025;
  doi:10.5061/dryad.m37pvmd99): CC0.
