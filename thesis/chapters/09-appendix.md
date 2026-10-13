```{=latex}
\appendix
```

# Reproducibility checklist {#sec:appendix-repro}

This appendix lists what is needed to rebuild every result of the thesis from the raw
data; @sec:method-repro summarises the measures that make the runs traceable. All
reported results come from the main run, called panel v2 in the repository, run with the
code at tag `v0.7-panel-v2` (commit `55bd778`) on frames of data-processing version 4
(ETL v4, for the fourth version of the extract–transform–load pipeline), and from the extra runs, made with the same model and data code (`src/`), extended for the XReg and maximum-context configurations (commit `ef5833e`, unchanged for the grid and inner-window runs at commit `5a1fecd`).

## What was fixed when

@Tbl:decisions dates every protocol choice and states what had been seen of the test
windows when it was made; `DECISIONS.md` in the repository is its source. All runs used
the same 270 test days, so no result comes from an untouched test set: the choices made
after a run are the ones a reader should weigh.

: Protocol choices in date order (2026), with what had been seen when each was made. The
first run is called panel v1 in the repository and the main run panel v2. Skill in the
second row is over Smart persistence, scored on the median and on the data processing of
the time. {#tbl:decisions}

| When | Choice | What had been seen |
|:-------------------|:--------------------------------------------------|:--------------------------------------|
| Before the first run | Panel, folds, metrics, model settings and contexts | No test result, apart from the next two rows |
| 1 Oct, smoke test | Fine-tuning recipe: learning rate $10^{-5}$, five epochs over daily windows | One test fold at CYT (Station 2): +0.35, against +0.24 zero-shot and +0.08 and +0.25 at the default rate $10^{-3}$ |
| 1 Oct, smoke test | LightGBM lag inputs corrected (a wiring bug) | A test fold |
| 2 Oct | Climatology and CLIPER added; point forecast = mean of the nine quantiles; the eleven comparisons fixed | The first run, on the same test days |
| 2 Oct | Main run restarted on corrected data processing | A stopped attempt with two data defects; no result used |
| 3–4 Oct | Longer contexts, base checkpoint, TiRex-long, two more fine-tuning seeds (‡) | The main run |
| 4 Oct | CLIPER as headline reference; intervals as uncertainty; E1–E5 without tests; RQ1 answered by the best zero-shot configuration; one last run declared | The main run and the extra runs |
| 4 Oct | Chronos-B-long shown in the main table and figures | All runs, including the last |
| 12–13 Oct | TiRex-XReg, TimesFM-XReg, Chronos-B-max, TimesFM-max (‡) declared before they ran; E6–E15 without tests; TimesFM 3.0 and TiRex-2 declared, not run (licence steps not done) | All earlier runs; regression settings fixed on days before the first test day |
| 13 Oct | Real-time selection of the zero-shot configuration (Chronos-sel ‡): rule, eight candidates, inner windows, V1–V4 without tests, reporting; Chronos-S-max and Chronos-B-mid (‡) run to complete the grid; no learning-rate search | All earlier runs; no inner-window forecast |

## Where each step is implemented

: The code behind each step of Chapters [-@sec:data] and [-@sec:method] (package `pvfc` in
`src/`, scripts in `experiments/`). {#tbl:code-map}

| Step | Section | Code |
|:-----------------------------------------|:------|:-------------------------------------------------|
| Reading the files; power column; relabelling of the hourly files | [-@sec:data-time] | `pvfc.data._read_power`, `pvfc.data._read_ts_csv` |
| Nameplate capacity from the Brick model | [-@sec:data-dataset] | `pvfc.data.load_nameplate_capacities` |
| Excluded stations | [-@sec:data-panel] | `pvfc.data.EXCLUDED_STATIONS` |
| Meter-error rules; bridging of short gaps | [-@sec:data-qc] | `pvfc.data._qc_power`, `pvfc.data._bridge_short_gaps` |
| Outage rule | [-@sec:data-qc] | `pvfc.qc` |
| Pre-flight checks | [-@sec:data-qc] | `experiments/check_data.py` |
| Template $P^{\mathrm{cs}}_t$ | [-@sec:data-clearsky] | `pvfc.clearsky.empirical_clearsky_power` |
| Weather archive and alignment | [-@sec:data-weather] | `pvfc.weather` |
| Issue times; target removed from the future frame | [-@sec:method-task] | `pvfc.runner` |
| Gap filling of the model inputs | [-@sec:method-target] | `pvfc.normalize.fill_index_gaps` |
| Weather regression of TiRex and TimesFM | [-@sec:method-covariates] | `pvfc.models.xreg` |
| Model settings | [-@sec:method-models] | `configs/experiment.yaml`, `configs/extras.yaml`, `configs/covariates.yaml`, `configs/grid.yaml` |
| Inner windows; selection of the zero-shot configuration; optimism estimates | [-@sec:method-selection] | `configs/validation.yaml`, `experiments/select_zero_shot.py` |
| Metrics, intervals, tests, tables and figures | [-@sec:method-metrics], [-@sec:method-tests] | `experiments/rescore.py`, `experiments/report.py` |
| Peak memory | [-@sec:method-compute] | `pvfc.utils.peak_rss_mb` |
| Batch of runs | [-@sec:method-compute] | `experiments/run_panel.py` |

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
| SciPy | 1.15.2 | clear-sky power $P^{\mathrm{cs}}_t$; t distribution of the DM test |
| scikit-learn | 1.7.2 | input scaling of the LSTM; LightGBM interface |
| statsforecast | 2.0.3 | AutoARIMA (SARIMA) |
| LightGBM | 4.6.0 | gradient-boosted trees |
| PyTorch | 2.10.0 | LSTM and foundation models (on the Apple GPU) |
| PyTorch Lightning | 2.5.2 | training loop of the darts neural models |
| darts | 0.47.0 | model wrappers and fine-tuning |
| tirex-ts | 1.4.2 | TiRex |
| huggingface_hub | 0.36.2 | checkpoint download |
| pvlib | 0.15.1 | solar position and clear-sky irradiance |

## Hardware

All runs used one Apple MacBook Air (13-inch, M4, 2025; model identifier Mac16,12) with
an M4 chip (10 CPU cores, 4 of them performance cores; 8 GPU cores), 16 GB of unified
memory and macOS 26.6. The laptop has no fan, so long runs may be slowed by thermal
throttling. The LSTM and the foundation models ran on the GPU through PyTorch's Metal Performance Shaders
(MPS) backend, with `PYTORCH_ENABLE_MPS_FALLBACK=1` sending unsupported operations to the CPU;
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
  weather setting, `data/processed/<station>__<setting>__etl4.parquet`, with the
  nameplate capacity in a `.cap.json` sidecar and the quality-control counts in
  `data/processed/qc_log.csv`. The ETL version is part of the file name, so frames built
  by an older ETL are never reused.
- **Checkpoints.** `make checkpoints` downloads the three foundation models of panel v2
  from Hugging Face; the base checkpoint of Chronos-2, used only in the extra runs, is
  downloaded on first use (@tbl:appendix-checkpoints). darts pins the revision of
  TimesFM-2.5; the others are taken from the default branch of their repositories, so an
  exact rebuild must use the revisions listed.

: Pretrained checkpoints used by the runs. {#tbl:appendix-checkpoints}

| Model | Hugging Face repository | Revision | Licence |
|:----------------|:----------------------------------|:-----------|:------------------------|
| Chronos-2-small | `autogluon/chronos-2-small` | `ddec013` | Apache-2.0 |
| Chronos-2-base | `amazon/chronos-2` | `29ec376` | Apache-2.0 |
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
- **Extra runs.** Thirteen configurations added after panel v2 had been seen
  (@sec:method-models), on the same data and folds. Seven are in `configs/extras.yaml`:
  zero-shot Chronos-2 at 1,024 and 2,048 hours of context (Chronos-S-mid, Chronos-S-long),
  the base checkpoint zero-shot at 336 and 2,048 hours (Chronos-B, Chronos-B-long), TiRex at
  2,048 hours (TiRex-long), and Chronos-S-FT with seeds 1 and 2 (NWP setting only). Four
  are in `configs/covariates.yaml`: Chronos-B-max, TimesFM-max (no-weather setting only),
  TiRex-XReg and TimesFM-XReg (NWP setting only; @sec:method-covariates). Two are in
  `configs/grid.yaml`: Chronos-S-max and Chronos-B-mid. Their results are in
  `results/extra/`, `results/covariates/` and `results/grid/` (results store, saved forecasts
  and logs), which `experiments/report.py` reads together with the panel.
- **Inner windows and the selected row.** `configs/validation.yaml` runs CLIPER and the
  eight candidates of @sec:method-selection on the nine 29-day inner windows, with the
  test procedure; the results are in `results/validation/`, which `experiments/report.py`
  refuses to read as test forecasts. `experiments/select_zero_shot.py` applies the rule,
  writes the selection path (@tbl:selection-path), assembles Chronos-sel from the
  candidates' test forecasts into `results/validation/selected/`, which the report reads,
  and computes the three optimism estimates. `results/validation/PREREGISTRATION.txt`
  holds the SHA-256 hashes of the rule's files, written before the first of these runs.
  `DECISIONS.md` in the repository dates every protocol choice and states what had been
  seen when it was made (@tbl:decisions).
- **Data-processing versions.** ETL v3 fixed four defects found in an audit of ETL v2:
  negative meter errors kept as zeros, zeros interpolated into long gaps, an unwrapped hour
  angle and a weather cache that began after SQ15's first 993 hours. ETL v4 returned from
  the energy column, adopted in v3, to the power column (@sec:data-time).
- **Panel v1.** The earlier run (ETL v2, with the LSTM and fine-tuned Chronos-2 trained on
  across folds) is archived in `results/archive/panel_v1/` and is not reported. It
  prompted the Climatology and CLIPER references, the quantile-mean point forecast
  (scoring version 2; version 1 scored the probabilistic models on their median) and the
  choice of the eleven comparisons (@sec:method-tests).
- **Stopped first attempt at panel v2.** A first panel v2 run, on ETL v3 (which read the
  15-minute stations from the energy column) and with a forward fill of the gaps in the
  model inputs, was stopped after 96 of the 144 runs when these two defects were found; it
  was repeated on ETL v4, with the gaps filled as in @sec:method-target. It is archived in
  `results/archive/panel_v2_etl3_stopped/` (tag `v0.7-panel-v2-etl3-stopped`, commit
  `54d1394`) and is not reported.

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
results store. Panel v2 began at 21:37 on 2 October 2026 and its last run ended at 20:00
on 3 October: 22.4 hours elapsed, of which the 144 runs took 18.5 hours
(`results/logs/panel_ledger.csv`); panel v1, 120 runs, had taken 19.3 hours. The restart
of the batch overwrote 132 of the 144 run logs of panel v2, because `run_panel.py` opens a
run's log afresh; the ledger, the results store and the saved forecasts are complete.

## Seeds and determinism

The configured seed (0) initialises Python's `random`, NumPy and PyTorch, including the
MPS generator (`pvfc.utils.set_seed`). Each darts model takes its random state from the
`seed` key of its own parameters, 0 unless set, so the LSTM starts from the same weights
in every fold; the top-level `seed` of the configuration reaches only `set_seed` and the
hash. The two extra fine-tuning seeds set that key (`seed: 1` and `seed: 2` in
`configs/extras.yaml`), and their forecasts differ from those of seed 0. The bootstrap of
the tests and intervals uses seed 0. Floating-point results on the GPU are not
bit-identical to those on the CPU, so the neural models are reproducible across devices
only up to small numerical differences. Chronos-2 (fine-tuned) was run with three seeds;
the LSTM with one, so its sensitivity to initialisation was not measured.

## Regression tests

`make test` runs 142 tests (`tests/`). Those that guard the protocol of Chapter
[-@sec:method] check:

- the data pipeline (`test_data_smoke.py`, `test_etl.py`, `test_qc.py`): UTC hourly index,
  interval-start labels and the relabelling of the hourly files, power from the power
  column, meter-error rules, bridging of gaps of at most two hours only, the outage rule,
  the wrapped hour angle and complete weather years;
- the protocol (`test_protocol.py`): whole local-day folds, the calendar anchor, the
  training-only template, the same-hour fallback of the references, the removal
  of the target from the future frame, one fit per fold scored under both hour sets, and
  the coverage guard;
- the models (`test_models_darts.py`, `test_gap_fill.py`, `test_baselines.py`,
  `test_covariate_models.py`): the LightGBM lag wiring, a fresh model in every fold,
  zero-shot checkpoints loaded once, the filling of gaps in the model inputs from the same
  hour of an earlier day (no dark day after an outage), Climatology and CLIPER, and the
  weather regression;
- scoring and reporting (`test_scoring.py`, `test_report.py`): the quantile-mean point
  forecast, rescoring that reproduces the runner's rows exactly, the DM statistic,
  Newey--West variance, Holm adjustment and block bootstrap, the day-block bootstrap of
  the intervals (all-ones weights reproduce the point estimates; a model equal to its
  reference gives [0, 0]), and the exclusion of runs that miss scored hours;
- the batch (`test_run_scripts.py`): the run order and the resumable `--skip-done`.
