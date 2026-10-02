```{=latex}
\appendix
```

# Reproducibility checklist {#sec:appendix-repro}

This appendix lists what is needed to rebuild every result of the thesis from the raw
data; @sec:method-repro summarises the measures that make the runs traceable. All
reported results come from panel v2, run with the code at tag `v0.7-panel-v2` (commit
`54d1394`) on frames of data-processing version 3 (ETL v3).

<!-- TODO(Lorenzo): add the repository URL (and an archived snapshot, e.g. a Zenodo DOI) once the repository is public. -->

## Software environment

The environment `pv` is built from conda-forge (`environment.yml`); darts and tirex-ts
come from PyPI through the extras of the package (`pyproject.toml`: `forecast` installs
`darts[torch]>=0.44` and `tirex-ts>=1.4`, `dev` installs pytest and ruff), and the exact
versions of every installed package are frozen in `requirements-lock.txt` (`make lock`).
@Tbl:appendix-software lists the main libraries; statsmodels 0.14.6, which provides only a
fallback for SARIMA, and Transformers 4.57.6 are installed but not used by the runs.

: Main software of the runs (Python 3.12.13, macOS 26.6 on arm64). {#tbl:appendix-software}

| Library | Version | Role in the runs |
|:---------------------|:--------|:--------------------------------------------------|
| NumPy | 2.4.6 | arrays and metrics |
| pandas | 2.3.3 | time series and results store |
| SciPy | 1.15.2 | clear-sky template; t distribution of the DM test |
| scikit-learn | 1.7.2 | input scaling of the LSTM; LightGBM interface |
| statsforecast | 2.0.3 | AutoARIMA (SARIMA) |
| LightGBM | 4.6.0 | gradient-boosted trees |
| PyTorch | 2.10.0 | LSTM and foundation models (MPS backend) |
| PyTorch Lightning | 2.5.2 | training loop of the darts neural models |
| darts | 0.47.0 | model wrappers and fine-tuning |
| tirex-ts | 1.4.2 | TiRex |
| huggingface_hub | 0.36.2 | checkpoint download |
| pvlib | 0.15.1 | solar position and clear-sky irradiance |

## Hardware

All runs used one Apple MacBook Air (13-inch, M4, 2025; model identifier Mac16,12) with
an M4 chip (10 CPU cores, 4 of them performance cores; 8 GPU cores), 16 GB of unified
memory and macOS 26.6. The laptop has no fan, so long runs may be slowed by thermal
throttling. The LSTM and the foundation models ran on the GPU through PyTorch's MPS
backend, with `PYTORCH_ENABLE_MPS_FALLBACK=1` sending unsupported operations to the CPU;
the other models ran on the CPU. The batch kept the machine awake with `caffeinate -i -s`
(on mains power, lid open) and ran one run at a time.

The peak resident set size (`pvfc.utils.peak_rss_mb`) does not count memory that PyTorch
allocates on the GPU through Metal, although CPU and GPU share one physical memory: in a
check, a 1 GiB tensor on the GPU raised the peak by about 40 MB, the same tensor on the
CPU by about 1 GiB. For the LSTM and the foundation models the reported peak is therefore
a lower bound (@sec:method-compute).

## Data and pretrained weights

- **Raw data.** The HKUST dataset [@lin2024hkustdata] must be downloaded by hand from
  Dryad (`Dataset.zip`, doi:10.5061/dryad.m37pvmd99) into `data/raw/`, because the page
  blocks automated downloads; `make data` extracts and inspects it.
- **Weather.** The Open-Meteo archive of historical forecasts [@openmeteo] is fetched
  for the campus coordinate and cached by `pvfc.weather` (`make nwp-cache`), in complete
  calendar years.
- **Processed frames.** `pvfc.data.load_station` writes one frame per station and
  weather setting, `data/processed/<station>__<setting>__etl3.parquet`, with the
  nameplate capacity in a `.cap.json` sidecar and the quality-control counts in
  `data/processed/qc_log.csv`. The ETL version is part of the file name, so frames built
  by an older ETL are never reused.
- **Checkpoints.** `make checkpoints` downloads the three foundation models from Hugging
  Face (@tbl:appendix-checkpoints). darts pins the revision of TimesFM-2.5; the other two
  are taken from the default branch of their repositories, so an exact rebuild must use
  the revisions listed.

: Pretrained checkpoints used by the runs. {#tbl:appendix-checkpoints}

| Model | Hugging Face repository | Revision | Licence |
|:----------------|:----------------------------------|:-----------|:------------------------|
| Chronos-2-small | `autogluon/chronos-2-small` | `ddec013` | Apache-2.0 |
| TimesFM-2.5 | `google/timesfm-2.5-200m-pytorch` | `1d95242` | Apache-2.0 |
| TiRex | `NX-AI/TiRex` | `63c7409` | NXAI Community License |

The NXAI Community License of TiRex must be accepted explicitly; the configuration does
so (`accept_license: true`, reviewed by the author on 1 October 2026).

## Configuration, hashes and results files

- **Configuration.** `configs/experiment.yaml` holds every setting: data (panel,
  weather setting, `etl_version`), folds, evaluation and models. Command-line overrides
  of `experiments/run.py` (station, weather setting, number of folds) are written into
  the configuration before it is hashed.
- **Configuration hash.** The run configuration (the `data` and `splits` blocks, the
  model entry and the seed) is serialised as JSON with sorted keys and hashed with
  SHA-1; the first 12 hexadecimal characters identify the run (`pvfc.utils.config_hash`).
  The evaluation settings and the code are not hashed; a change of scoring rule is
  recorded by the `scoring_version` of each row instead.
- **Version check.** `experiments/run.py` stops if `data.etl_version` differs from
  `pvfc.data.ETL_VERSION`, so results on rebuilt data never share a hash with older ones.
- **Results store.** `results/results.parquet` holds one row per run, fold and hour set
  (daytime or all hours), keyed by the hash; re-running a configuration replaces its
  rows. Each row also records the environment fingerprint (`pvfc.utils.env_fingerprint`):
  Python version, platform and the versions of NumPy, pandas, scikit-learn, LightGBM,
  statsmodels, statsforecast, PyTorch, pvlib, darts and tirex-ts.
- **Saved forecasts.** `results/predictions/<station>__<setting>__<model>__<hash>.parquet`
  holds, for every test hour, the observation, the model's raw output (native point
  forecast and quantiles), the Smart persistence forecast, the template, the daytime
  flag and the fold. `experiments/rescore.py` and `experiments/report.py` compute every
  metric from these files.
- **Logs.** `results/logs/<station>__<setting>__<model>.log` per run,
  `results/logs/check_data__<station>.log` for the pre-flight checks,
  `results/logs/panel_ledger.csv` with the start time, status, exit code and wall time of
  every run, and `results/logs/panel.out` with the output of the batch.
- **Panel v1.** The earlier run (ETL v2, with the LSTM and fine-tuned Chronos-2 trained on
  across folds) is archived in `results/archive/panel_v1/` and is not reported.

<!-- TODO(Lorenzo): once panel v2 has finished, add a table (or a CSV in the repository) with the 144 configuration hashes behind the Results tables, e.g. from results/results.parquet grouped by model, weather_mode and site. -->

## Commands

The `make` targets of @tbl:appendix-make assume that the `pv` environment is active.
From a fresh checkout, the results are rebuilt by downloading the raw data (above), then
running `make env`, `conda activate pv`, `make install`, `make test`, `make data` and
`make reproduce`; `make manuscript` builds this document.

: `make` targets of the repository. {#tbl:appendix-make}

| Target | Action |
|:-----------------|:-----------------------------------------------------------------------|
| `env` | create the conda environment `pv` from `environment.yml` |
| `install` | install the `pvfc` package with the `forecast` and `dev` extras |
| `check` | check the toolchain (versions, MPS, pvlib, LightGBM) |
| `test` | run the regression tests |
| `data` | extract and inspect the Dryad archive in `data/raw/` |
| `nwp-cache` | fetch and cache the Open-Meteo archive for 2021--2023 |
| `check-data` | run the pre-flight checks of the data pipeline |
| `checkpoints` | download the three foundation-model checkpoints |
| `panel` | run the 144 runs, one process each, skipping runs already stored |
| `rescore` | recompute every metric row from the saved forecasts |
| `report` | build the tables and figures of the Results chapter |
| `reproduce` | `checkpoints`, then `panel`, `rescore` and `report` |
| `lock` | freeze the installed packages into `requirements-lock.txt` |
| `manuscript` | build the thesis PDF with pandoc and tectonic |

`make panel` (`experiments/run_panel.py`) first runs the pre-flight checks of
`experiments/check_data.py` for every station and stops if one fails; it can be
restarted at any point and skips the runs whose configuration hash is already in the
results store.

<!-- TODO(Lorenzo): panel v2 wall time from results/logs/panel.out (panel v1, 120 runs, took 19.3 h). -->

## Seeds and determinism

The configured seed (0) initialises Python's `random`, NumPy and PyTorch, including the
MPS generator (`pvfc.utils.set_seed`). Every darts model receives the random state 0, so
the LSTM starts from the same weights in every fold, and the bootstrap of the
significance tests uses seed 0. Floating-point results on the GPU are not bit-identical
to those on the CPU, so the neural models are reproducible across devices only up to
small numerical differences; with a single seed, their sensitivity to initialisation was
not measured.

## Regression tests

`make test` runs 107 tests (`tests/`). Those that guard the protocol of Chapter
[-@sec:method] check:

- the data pipeline (`test_data_smoke.py`, `test_etl.py`, `test_qc.py`): UTC hourly index,
  interval-start labels, power from the energy column, meter-error rules, bridging of
  gaps of at most two hours only, the outage rule, the wrapped hour angle and complete
  weather years;
- the protocol (`test_protocol.py`): whole local-day folds, the calendar anchor, the
  training-only clear-sky template, the same-hour fallback of the references, the removal
  of the target from the future frame, one fit per fold scored under both hour sets, and
  the coverage guard;
- the models (`test_models_darts.py`, `test_baselines.py`): the LightGBM lag wiring, a
  fresh model in every fold, zero-shot checkpoints loaded once, and Climatology and
  CLIPER;
- scoring and reporting (`test_scoring.py`, `test_report.py`): the quantile-mean point
  forecast, rescoring that reproduces the runner's rows exactly, the DM statistic,
  Newey--West variance, Holm adjustment and block bootstrap, and the exclusion of runs
  that miss scored hours;
- the batch (`test_run_scripts.py`): the run order and the resumable `--skip-done`.
