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
Zero-shot Chronos-2/TimesFM-2.5 run on darts 0.41; **fine-tuning needs darts>=0.42**
(pin is `u8darts[torch]>=0.43`).

**v0.5 (`v0.5-extended-foundation-roster`):** Moirai-2 (Salesforce, via `uni2ts`)
and TTM / TinyTimeMixer (IBM, via `granite-tsfm`) are now wired through a separate
*external-foundation* wrapper (`src/pvfc/models/external_foundation.py`). The
arch/label split scores zero-shot and fine-tuned as distinct result rows
(`moirai2`/`moirai2_ft`, `ttm`/`ttm_ft`). Both libraries are **optional**: the
darts-native roster (Chronos-2, TimesFM-2.5, TiRex) keeps working without them.
Each wrapper lazy-imports its dep and raises a clear install message if missing.

### Foundation-model roster (v0.5)

| Arch | Library | Zero-shot label | Fine-tuned label | Dep |
|---|---|---|---|---|
| chronos2 | darts | `chronos2` | `chronos2_ft` | `u8darts[torch]>=0.43` |
| timesfm | darts | `timesfm` | `timesfm_ft` | `u8darts[torch]>=0.43` |
| tirex | darts | `tirex` | `tirex_ft` | `u8darts[torch]>=0.44` |
| moirai2 | **uni2ts** (optional) | `moirai2` | `moirai2_ft` | `uni2ts>=1.2` |
| ttm | **granite-tsfm** (optional) | `ttm` | `ttm_ft` | `granite-tsfm[notebooks]>=0.2.16` |

To opt into Moirai-2 / TTM on the Mac:

```bash
pip install 'uni2ts>=1.2'                      # for moirai2 / moirai2_ft
pip install 'granite-tsfm[notebooks]>=0.2.16'  # for ttm / ttm_ft
```

Then uncomment the corresponding entries in `configs/experiment.yaml` (search
for "external-library foundation models"). The first run downloads the HF
checkpoint(s); subsequent runs reuse the cache.

To run on real data: download `Dataset.zip` from
[Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.m37pvmd99) into `data/raw/`
(the page guards automated downloads, so use a browser), then `make data` to
extract + inspect, then run the model ladder on a station, e.g.:

```bash
python experiments/run.py --config configs/experiment.yaml --site "CYT (Station 1)" --weather-mode oracle
python experiments/run.py --config configs/experiment.yaml --site "CYT (Station 1)" --weather-mode nwp
make report   # -> thesis/tables/ + thesis/figures/
```
