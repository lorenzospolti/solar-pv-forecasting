# Decision log

Protocol decisions in date order, with what was known when each was made. The thesis
states every choice made after test-window results were seen; this file is its source.
Times are local (Europe/Rome).

## Before panel v2

- **1 Oct 2026. Fine-tuning recipe for Chronos-2.** Learning rate 1e-5, five epochs,
  batches of 32, one training window per day ending at local midnight. Chosen in a smoke
  test on CYT (Station 2), fold 0, which is a test fold. Excluding that fold or that
  station leaves the comparison with LightGBM unchanged (thesis, Method).
- **1-2 Oct 2026. Panel v1** (superseded, not reported) ran on the same 270 test days.
  What it informed:
  - the Climatology and CLIPER references were added;
  - the point forecast of a probabilistic model became the mean of its nine quantiles
    (scoring version 2, 2 Oct);
  - the family of 11 pairwise comparisons was chosen after day-level tests on panel v1
    and fixed in `experiments/report.py` on 2 Oct (commits 7646cd5, 54d1394, 55bd778),
    before panel v2 started at 21:37.
- **2-3 Oct 2026. Panel v2:** 144 runs (6 stations x 2 weather settings x 12 models x
  9 folds), complete on 3 Oct at about 20:00; 18.5 h of run time.

## After panel v2 results were seen

- **3 Oct 2026, 20:22 - 4 Oct, 01:37. Extra runs** (`configs/extras.yaml`, results in
  `results/extra/`), same folds and code as panel v2:
  - Chronos-2-small zero-shot at 1,024 h and 2,048 h of context;
  - Chronos-2-base zero-shot at 336 h;
  - TiRex at 2,048 h;
  - Chronos-2 fine-tuned with seeds 1 and 2 (NWP setting).
- **4 Oct 2026. Fact-check and triage** of the 3 Oct design reassessment. No 60-day
  re-run: it would not decide LightGBM vs fine-tuned Chronos-2 (power for the observed
  effect 0.15 -> 0.26). The month gaps of the test calendar are stated as a limitation.

## 4 Oct 2026, written before the run it declares

- **One last model run:** Chronos-2-base zero-shot at a 2,048-h context
  (`chronos2_base_ctx2048`), both weather settings, six stations, nine folds, same code
  as panel v2 (`src/` unchanged since d07d0c4). Its result is reported whatever it shows.
- **No further model runs** unless the tutor asks for one; any such run gets its own
  entry here before it starts.
- **Reporting rules, fixed before that run's result is seen:**
  - Skill is reported over CLIPER, with smart persistence as a row and its skill
    scores in the appendix.
  - Uncertainty: 95% intervals from a 7-day moving-block bootstrap over test days
    (10,000 resamples, seed 0), the same days for every station and model. They
    replace the [min, max] across stations in the main text.
  - The 11 comparisons fixed on 2 Oct stay as they are (Holm-adjusted).
  - Comparisons that involve the extra runs are post hoc. They are reported as paired
    skill differences with 95% intervals and no tests:
    - E1: Chronos-2 zero-shot 2,048 h vs 336 h (NWP)
    - E2: LightGBM vs Chronos-2 zero-shot 2,048 h (NWP)
    - E3: Chronos-2 fine-tuned 336 h vs Chronos-2 zero-shot 2,048 h (NWP)
    - E4: LightGBM vs Chronos-2-base zero-shot 2,048 h (NWP)
    - E5: Chronos-2 fine-tuned 336 h vs Chronos-2-base zero-shot 2,048 h (NWP)
  - RQ1 is answered with the best zero-shot configuration tested. That selection is
    post hoc and favours the zero-shot model.
  - The probabilistic score is reported as the quantile score: the mean pinball loss
    over the nine quantile levels, divided by capacity.
- **Not done, by decision:** 60-day test windows; a "published-configuration" LightGBM;
  an NWP-only linear reference; an NWP-uptake regression; conformal recalibration;
  model averaging; new inputs; a raw-power target; a learning-rate ablation;
  fine-tuning at a long context (it does not fit in 16 GB at the recipe's batch of 32).

## 4 Oct 2026, after the declared run

- **Outcome of `chronos2_base_ctx2048`:** started 11:20, finished 11:41 (0.34 h of run
  time); 12 of 12 runs ok in `results/extra/logs_base2048/panel_ledger.csv`. Checks
  passed: 12 prediction files, each with 6,480 rows, 9 folds, no missing forecast and the
  same time index and folds as the panel's Chronos-2 file for that station; no line with
  WARNING, Traceback or "without covariates" in the run logs; the NWP and no-weather
  forecasts differ at every station.
- **Result:** skill over CLIPER 0.274 [0.227, 0.316] with NWP and 0.008 [−0.037, 0.059]
  without weather; 11.9 s per test window, no training. It is the best zero-shot
  configuration tested, so by the rule above it answers RQ1. With NWP it cannot be told
  apart from LightGBM (E4: 0.007 [−0.029, 0.041]) or from Chronos-2 fine-tuned at 336 h
  (E5: −0.016 [−0.051, 0.018]).
- **Presentation, decided after seeing this result:** the main table and figures show
  the configuration fixed before panel v2 (Chronos-2-small, zero-shot, 336 h) and the one
  that answers RQ1 (Chronos-2-base, zero-shot, 2,048 h). Chronos-2-small at 2,048 h and
  the other configurations stay in the Chronos-2 configuration table.

## 12 Oct 2026, written before the runs it declares

- **Covariate runs** (`configs/covariates.yaml`, design in `COVARIATES.md`). Results go
  to `results/covariates/`, never to `results/results.parquet` or `results/extra/`.
  - Post hoc: decided by Lorenzo on 12 Oct, after panel v2, the extras and
    `chronos2_base_ctx2048` had been seen. A further run in the sense of the 4 Oct
    entry, so it gets this entry before it starts.
  - Same as panel v2: data (ETL v4), the nine 30-day test windows from 5 Jul 2022,
    six stations, the runner's day-ahead protocol, scoring version 2.
- **The runs** (each x 6 stations x 9 folds):

  | Batch | Configuration | What it is | Settings | Runs | Env |
  |---|---|---|---|---:|---|
  | 1 | `tirex_xreg` | TiRex, 2,048 h; weather through an in-context ridge regression | NWP | 6 | pv |
  | 1 | `timesfm_xreg` | TimesFM-2.5, all history up to 16,360 h; the same regression | NWP | 6 | pv |
  | 2 | `chronos2_base_ctx8192` | Chronos-2-base at its maximum context, 8,192 h | NWP, No weather | 12 | pv |
  | 2 | `timesfm_ctx16k` | TimesFM-2.5 univariate, all history up to 16,360 h | No weather † | 6 | pv |
  | 3 | `timesfm3`, `tirex2` | zero-shot, covariates as known-future inputs | NWP, No weather | 24 | pv-cov |

  Batch 3 runs only once Lorenzo has accepted the TimesFM-3 non-commercial licence
  v1.0 and the TiRex-2 terms on Hugging Face. Otherwise it does not run, and the
  outcome entry says so. Before it runs:
  - its two checkpoints are downloaded online and their revisions recorded here;
  - `accept_license: true` for `timesfm3` is committed, the only change to the config.
- **Why these settings:**
  - `timesfm_ctx16k` reads no covariates, so its two settings cannot differ: TimesFM,
    TiRex and TiRex-long gave bit-identical NWP and No-weather forecasts at all six
    stations. It runs once, under No weather, marked †. `report.py` infers † from
    paired runs, so for this row the mark rests on this entry.
  - `tirex_xreg` and `timesfm_xreg` run with NWP only. Without weather their
    regression would read only p_clear and the calendar terms: a configuration that
    was not in the development check and answers no question here. Their no-weather
    counterparts are the univariate rows at the same context (TiRex-long,
    `timesfm_ctx16k`), so E8 and E11 measure the regression and the weather together.
  - `chronos2_base_ctx8192`, `timesfm3` and `tirex2` read covariates natively: both
    settings, like Chronos-B-long. `timesfm3` and `tirex2` receive Chronos-2's
    covariates; the xreg models receive the same plus `kc_nwp`.
- **How the regression's settings were fixed:**
  - On development days before the first test day: 4 Jan-2 Jun 2022, three 30-day
    windows, six stations, NWP (`COVARIATES.md`, section 4). No test day was used.
  - Settings: Chronos-2's covariates plus the forecast clear-sky index `kc_nwp`, an
    8,192-h fit window, ridge 1.
  - `tirex_xreg` is the configuration checked. `timesfm_xreg` uses the checked
    regression, but TimesFM's context was set to its maximum (2,048-16,360 h, 683
    days of history) to match `timesfm_ctx16k`; the check ran it at 512-8,192 h.
- **Code, data and environment of each batch:**
  - Code: `src/`, the run scripts (`experiments/run_panel.py`, `run.py`,
    `check_data.py`) and `configs/covariates.yaml` at ef5833e, run from the main
    checkout (`~/Thesis/solar-pv-forecasting`, clean apart from `thesis/chapters/`).
    `src/` is not edited while a batch runs. The summary script is at ed386cb, written
    and checked against E1-E5 before any covariate run.
  - Since panel v2's code (d07d0c4), `src/` has only gained code: `xreg.py`,
    `covariate_foundation.py`, and in `darts_wrapper.py` an xreg branch and
    [min, max] contexts. None of it is active for a reported configuration.
  - Batches 1-2: env `pv`, unchanged (`environment-lock-pv-2026-10-12.yml`; the `env`
    fingerprint of every reported row).
  - Batch 3: env `pv-cov` (`environment-lock-pv-cov-2026-10-12.yml`), a clone of pv
    plus timesfm 3.0.2, tirex-2 0.3.1, xlstm 2.0.6, flashrnn 1.0.8, mlstm-kernels
    2.0.6 and ninja 1.13.2. No other package changed version.
  - Checkpoints: the revisions of the reported runs (TiRex 63c7409, TimesFM-2.5
    1d95242, Chronos-2-base 29ec376), held fixed with `HF_HUB_OFFLINE=1`. Batch 3's
    revisions are recorded when they are downloaded.
  - Pre-flight on the synthetic station only, no test day: the four configurations
    of batches 1-2 load offline and run one fold without a WARNING (12 Oct, 21:39).
    The synthetic station has no forecast GHI, so `kc_nwp` was not exercised.
  - Each logs directory gets a manifest written when the batch starts: python and
    env, the main checkout's HEAD and status, the checkpoint revisions, and the
    sha256 of the panel and extras stores.
- **Commands**, from `~/Thesis/solar-pv-forecasting` after `conda activate pv` (batch 3:
  `pv-cov`) and `export HF_HUB_OFFLINE=1`; every command also takes
  `--config configs/covariates.yaml --results results/covariates/results.parquet
  --predictions-dir results/covariates/predictions`:
  - batch 1, about 0.5 h: `python experiments/run_panel.py --models tirex_xreg,timesfm_xreg
    --weather-modes nwp --logs-dir results/covariates/logs_b1_xreg`
  - batch 2, about 1.5 h, two commands in turn:
    `--models chronos2_base_ctx8192 --weather-modes nwp,none --logs-dir
    results/covariates/logs_b2_chronos8192`, then `--models timesfm_ctx16k
    --weather-modes none --logs-dir results/covariates/logs_b2_timesfm16k`
  - batch 3, after the licence steps: `--models timesfm3,tirex2 --weather-modes nwp,none
    --logs-dir results/covariates/logs_b3_native`
  - One batch at a time.
  - A failed or interrupted run is resumed with the same command and a new logs
    directory.
  - A fix to code or config gets its own entry here before the rerun.
- **Checks after each batch** (as for `chronos2_base_ctx2048` on 4 Oct):
  - every run ok in its ledger;
  - one prediction file per station, configuration and setting, each with 6,480 rows,
    9 folds, no missing forecast and the same time index as the panel's files;
  - no WARNING, Traceback or "without covariates" in the logs, and no "kc_nwp dropped"
    in an NWP log. Expected and harmless: one pandas FutureWarning per xreg process,
    from the clear-sky cache;
  - NWP and No-weather forecasts differ where the model reads covariates;
  - `results/results.parquet` and `results/extra/` unchanged (sha256).
- **Reporting rules, fixed before any result is seen:**
  - Every result is reported whatever it shows, marked ‡.
  - Skill over CLIPER with its 95 % interval: 7-day moving-block bootstrap over the
    270 test days, 10,000 resamples, seed 0, the weights of the thesis tables.
  - Also reported: PICP-80 with its interval, QS / C, median time per test window and
    maximum peak RSS, which does not count GPU memory, as in the compute table
    (`experiments/covariate_summary.py`, built on report.py's functions).
  - Paired skill differences A − B with 95 % intervals and no tests. They join
    neither P1-P11 nor the Holm family:
    - E6 `tirex_xreg` vs Chronos-B-long (NWP)
    - E7 `tirex_xreg` vs LightGBM (NWP)
    - E8 `tirex_xreg` (NWP) vs TiRex-long †
    - E9 `timesfm_xreg` vs Chronos-B-long (NWP)
    - E10 `timesfm_xreg` vs LightGBM (NWP)
    - E11 `timesfm_xreg` (NWP) vs `timesfm_ctx16k` †
    - E12 `timesfm_ctx16k` † vs Chronos-B-long (NWP)
    - E13 `timesfm_ctx16k` † vs LightGBM (NWP)
    - E14 `chronos2_base_ctx8192` vs Chronos-B-long (NWP)
    - E15 `chronos2_base_ctx8192` vs LightGBM (NWP)
    - if batch 3 runs: E16-E17 `timesfm3` and E18-E19 `tirex2`, each vs
      Chronos-B-long and vs LightGBM (NWP).
  - RQ1 keeps the answer fixed on 4 Oct (Chronos-B-long). These rows are reported
    beside it and do not reopen the choice of the best zero-shot configuration:
    - the xreg rows add a regression fitted for every forecast;
    - choosing the best of a growing post hoc set would favour the zero-shot side
      further.
  - No further model runs after these unless the tutor asks for one; any such run
    gets its own entry here first.
