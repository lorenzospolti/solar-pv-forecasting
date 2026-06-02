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
Harness spine + baselines + **real HKUST ETL** + **NWP weather feed** (Open-Meteo)
+ **SARIMA (AutoARIMA), LightGBM, and LSTM** wrappers are implemented and verified on
synthetic data. Foundation-model wrappers (Chronos-2, TimesFM-2.5, TTM) are next
(Week 8) — the darts 0.41 classes are present and constructible in the env.

To run on real data: download `Dataset.zip` from
[Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.m37pvmd99) into `data/raw/`
(the page guards automated downloads, so use a browser), then `make data` to
extract + inspect, set `data.site_id` in the config to a discovered station, and
`make check-data` / `python experiments/run.py --config configs/experiment.yaml`.
