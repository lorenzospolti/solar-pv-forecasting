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
| `models/baselines.py` | persistence, **smart persistence**, seasonal-naive |
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
- Chronos-2 fine-tuning: lr 1e-5 (AutoGluon's default; darts' 1e-3 destroys the
  pretrained weights), 5 epochs on one training window per day (`fit_stride: 24`);
- TiRex is licence-gated (NXAI Community License): set `accept_license: true` in its
  config entry once you have reviewed the terms;
- `run.py` saves rows after each model, writes per-timestamp predictions to
  `results/predictions/`, and takes `--models` / `--max-folds`.

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
