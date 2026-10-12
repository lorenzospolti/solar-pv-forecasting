# Weather covariates for TiRex and TimesFM, longer contexts, newer checkpoints

Written 10 Oct 2026. **Nothing in this file has run on the test days** (5 Jul 2022 –
26 Nov 2023). The design choices below were made on development days before the first
test day (section 4). The proposed runs are in `configs/covariates.yaml`; declare them in
`DECISIONS.md` before starting them (draft entry in section 6).

## 1. How long a context each model accepts

| Model (checkpoint) | Hard maximum (hours) | Where the limit comes from | Used so far |
|---|---:|---|---|
| TiRex (`NX-AI/TiRex`, 35.3 M) | **2,048** | `train_ctx_len = 2048` in the checkpoint. tirex-ts cuts longer inputs to 2,048 and NaN-pads shorter ones up to it. | 512 (panel); 2,048 (`tirex_ctx2048`), already the maximum |
| TimesFM-2.5 (`google/timesfm-2.5-200m-pytorch`) | **16,360** for a 24-h horizon | `context_length: 16384` in config.json. darts requires context + horizon ≤ 16,384. | 512 |
| Chronos-2 small / base | **8,192** | `context_length` in the checkpoint config. darts refuses longer inputs. | 336–2,048 |
| TimesFM-3 (`google/timesfm-3.0-pytorch`) | **15,360** | `_MAX_CONTEXT_LENGTH` in timesfm 3.0.2 | — |
| TiRex-2 (`NX-AI/TiRex-2`) | **2,048** | `context_len` of its config (model card). Could not be checked here: the repo is gated. | — |

How much history each fold has:
- before the first test day: 369–468 days (8,856–11,233 h);
- by the last fold: about 880–950 days.

What that means for each model:
- **Chronos-2:** can read its full 8,192 h on every fold.
- **TimesFM-2.5:**
  - reaches 16,360 h (682 days) only from the sixth of nine folds at most stations (fifth at SQ15, seventh at SQ567);
  - earlier folds get all the history there is.
- **TiRex:** stops at 85 days, whatever the history.

Two harness details matter for long contexts:

- **`context_days`** cuts the history before the model sees it. The default is 60 days, so it must be at least context / 24 + 1.
- **`input_chunk_length`**:
  - A fixed value makes darts refuse a fold whose training window is shorter than context + horizon.
  - A `[min, max]` pair is accepted by `darts_wrapper` from now on (darts ≥ 0.47). The model then reads every available hour up to `max`, and the missing hours are NaN-padded and masked.

## 2. Larger or newer models that run in 16 GB

Apart from Toto-2.0's 1 B and 2.5 B versions (about 4 and 9 GB of fp32 weights), none of these needs more than ~2 GB for inference: memory is not the limit on the M4.

**What exists (as of Oct 2026):**

| Model | Params | Max context | Weather covariates | Licence | Notes |
|---|---:|---:|---|---|---|
| **TimesFM-3** (Aug 2026) | ~330 M | 15,360 | **native** (past-only and known-future) | **non-commercial v1.0** (academic research allowed) | #1 foundation model on GIFT-Eval; #1 downloadable model on fev-bench. Torch (MPS) or Apple MLX backend. Wired: `timesfm3` |
| **TiRex-2** (Jul 2026) | 38.4 M + 44.1 M multivariate | 2,048 | **native** (past and known-future) | Apache-2.0 (gated repo on Hugging Face) | Successor of TiRex. CPU/MPS supported since 0.2.1. Wired: `tirex2` |
| TimesFM-2.5 | 200 M (+30 M quantile head) | 16,384 | XReg linear regression only | Apache-2.0 | What you run now |
| Chronos-2 base | 120 M | 8,192 | native | Apache-2.0 | No larger Chronos-2; no Chronos-3 |
| t0-beta (The Forecasting Co.) | ~256 M | ? | known-future | Apache-2.0 | Not checked here; needs `tfc-t0`; not in darts 0.47 |
| Toto-2.0 | 313 M / 1 B / 2.5 B | 4,096 | none yet | Apache-2.0 | Bigger, but cannot read the weather |
| TimesFM-2.0 | 500 M | 2,048 | XReg | Apache-2.0 | Bigger, older and weaker than 2.5 |

On fev-bench's 30 tasks with known covariates, the win rates are:

| Model | fev-bench win rate (known-covariate tasks) |
|---|---:|
| TimesFM-3 | 88.2 |
| Chronos-2 | 85.9 |
| TiRex-2 | 76.6 |
| TimesFM-2.5 | 61.8 |

These were recomputed from the fev repository on 9 Oct 2026.

Parameter count alone does not buy accuracy here: TimesFM-2.5 (200 M) beats TimesFM-2.0 (500 M), and TiRex (35 M) beats much larger models. For day-ahead PV, what counts is whether the model can **read the weather forecast**.

## 3. How the weather reaches TiRex and TimesFM

### 3a. With the checkpoints you already have: in-context regression (`xreg`)

darts 0.47 gives neither TiRex nor TimesFM-2.5 covariate support. Google's own answer for TimesFM-2.5 is XReg (`forecast_with_covariates`, Oct 2025):

- a linear model of the target on the covariates, fit on the context;
- the foundation model forecasts the residual.

`pvfc/models/xreg.py` re-implements it in numpy, so the same code serves TiRex and TimesFM and needs no JAX:

1. A ridge regression of k on the covariates is fit on the daytime hours of the last `fit_hours` of history.
2. The darts model, still univariate, forecasts the residual series. The residual is 0 at night; gaps are filled like k.
3. Every residual quantile is shifted by the regression's forecast, which is 0 at night. Power is then k × p_clear, as for every model.

Inputs: the covariates Chronos-2 receives, namely `p_clear`, the four weather columns and the four calendar terms.

Option `covariates+kc` adds `kc_nwp`, the forecast's own clear-sky index:

- it is forecast GHI over the interval-mean pvlib clear-sky GHI, the geometry behind `p_clear`;
- k is close to proportional to it, while k is far from linear in raw GHI (400 W/m² is an overcast noon but a clear early morning);
- it transforms inputs every model already gets, not new data. Say so in the Method chapter.

Differences from Google's xreg_lib:

- the fit uses daytime hours only;
- the intercept is not shrunk.

Use it with `xreg: {features: covariates+kc, ridge: 1.0, fit_hours: 8192}` on an `arch: tirex` or `arch: timesfm` entry.

### 3b. Native covariates: TimesFM-3 and TiRex-2 (`pvfc/models/covariate_foundation.py`)

Both receive the same covariates as known-future inputs over context + horizon (option `covariate_features: covariates+kc` adds `kc_nwp`). Zero-shot only.

How they were tested:

- **Done here:**
  - against the real package code (timesfm 3.0.2, tirex-2 0.3.1) with random weights: shapes, call signatures, quantile mapping;
  - with stand-ins: the covariates reach the model aligned with the hours it predicts.
- **Not done here:** a run with the real weights. Hugging Face is unreachable from the sandbox I worked in, and TiRex-2 is gated. The first real run happens on your Mac.

Licences:

- **TimesFM-3:** the config ships with `accept_license: false`. The weights are under the TimesFM non-commercial licence v1.0:
  - academic research is allowed;
  - no commercial or production use;
  - no redistribution;
  - the licence is revocable.
  Review it, then set the flag to `true`, as you did for TiRex.
- **TiRex-2:** accept the terms on <https://huggingface.co/NX-AI/TiRex-2> and run `hf auth login` once.

## 4. Pre-test check (development days only)

Set-up:

- **Days:** three 30-day windows starting 4 Jan, 5 Mar and 4 May 2022, all before the first test day; 90 days per station.
- **Stations and setting:** six stations, NWP setting.
- **Protocol:** the runner's day-ahead protocol.
- **Score:** skill over CLIPER on daytime hours, each station pooled, mean over stations.
- **Hardware:** a Linux CPU sandbox, so timings are not comparable with the M4.
- **Files:** `results/dev_covariates/`. Reproduce with `python experiments/precheck_covariates.py`.

| Configuration (dev days) | Skill over CLIPER | nRMSE | QS / C | PICP-80 |
|---|---:|---:|---:|---:|
| TimesFM-2.5, 512 h, univariate (the panel setting) | −0.051 | 0.171 | 0.0461 | 0.740 |
| TimesFM-2.5, up to 8,192 h, univariate | −0.039 | 0.169 | 0.0451 | 0.788 |
| TiRex, 2,048 h, univariate | 0.014 | 0.161 | 0.0426 | 0.775 |
| Regression alone (covariates + kc, 8,192 h), no foundation model | 0.277 | 0.118 | — | — |
| TiRex + xreg, covariates, 2,048-h fit | 0.251 | 0.122 | 0.0336 | 0.666 |
| TiRex + xreg, covariates + kc, 2,048-h fit | 0.266 | 0.120 | 0.0329 | 0.661 |
| TiRex + xreg, covariates, 8,192-h fit | 0.271 | 0.119 | 0.0328 | 0.684 |
| **TiRex + xreg, covariates + kc, 8,192-h fit** (`tirex_xreg`) | **0.283** | 0.117 | 0.0321 | 0.683 |
| TimesFM-2.5 + xreg, covariates, 8,192-h fit | 0.269 | 0.119 | 0.0330 | 0.629 |
| **TimesFM-2.5 + xreg, covariates + kc, 8,192-h fit** (`timesfm_xreg`) | **0.281** | 0.117 | 0.0322 | 0.634 |
| Chronos-2-base, 2,048 h, native covariates (= Chronos-B-long) | 0.325 | 0.110 | 0.0283 | 0.801 |

Reading it (every paired difference below holds at 6 of 6 stations):

- **What the weather adds:**
  - TiRex gains +0.27 skill and TimesFM-2.5 +0.33.
  - The regression does most of the work: on its own it reaches 0.277. The foundation model adds +0.005 at an 8,192-h fit and +0.02 at a 2,048-h fit.
  - `kc_nwp` adds +0.012. The 8,192-h fit window adds +0.016 over 2,048 h.
- **Where it still falls short:** Chronos-2-base with native covariates stays ahead by about +0.04. It is also far better calibrated: PICP-80 is 0.80, against 0.63–0.68 for the xreg models, whose residual quantiles are too narrow (no recalibration, by your 4 Oct decision).
- **The development days are easier than the test days for every model:** Chronos-B-long scores 0.325 here and 0.274 on the test days. Compare rows within this table, not with the thesis tables.

What to expect on the test days:

- The xreg rows would likely sit near or below LightGBM and Chronos-B-long, but far above today's univariate TiRex/TimesFM rows.
- That turns the † footnote ("cannot use the weather") into a measured result: given the forecast through a linear regression, both models gain a lot, yet stay behind a model that reads the covariates natively.
- The models with the best chance to beat Chronos-2 are the native ones: TimesFM-3 and TiRex-2.

## 5. Running it on the M4

```bash
conda activate pv
python -m pip install -e ".[forecast,covariates,dev]"   # [covariates]: only for timesfm3 / tirex2
hf auth login                                          # TiRex-2 is gated: accept its terms first
python experiments/fetch_checkpoints.py --all          # + amazon/chronos-2, TimesFM-3, TiRex-2
make test                                              # incl. tests/test_covariate_models.py
# declare the runs in DECISIONS.md (section 6), then:
python experiments/run_panel.py --config configs/covariates.yaml \
    --results results/covariates/results.parquet \
    --predictions-dir results/covariates/predictions --logs-dir results/covariates/logs
python experiments/covariate_summary.py                # skill over CLIPER next to the reported rows
```

Start with `--models tirex_xreg,timesfm_xreg`: they need no new install. The whole file is roughly one night on the M4. TimesFM-3 at 15,360 h is the slowest entry.

If MPS misbehaves for TimesFM-3 or TiRex-2, you have two options:

- add `device: cpu`;
- for TimesFM-3, add `backend: mlx` after `python -m pip install 'timesfm[mlx]>=3.0.2'`.

`experiments/covariate_summary.py` reproduces the master-table skill exactly on the existing predictions. For example, LightGBM NWP is 0.2803 and Chronos-B-long NWP 0.2736. The thesis tables still come from `experiments/report.py`, which needs the new names in `POST_HOC_MODELS` / `MAIN_MODELS` if you report them.

## 6. Draft entry for DECISIONS.md

> ## 10 Oct 2026, written before the runs it declares
>
> **Covariate runs** (`configs/covariates.yaml`, results in `results/covariates/`):
>
> - Same folds, data and code path as panel v2, both weather settings, six stations, nine folds.
> - Post hoc: decided after panel v2 and the extras had been seen.
>
> **The runs:**
>
> - TiRex and TimesFM-2.5 read the weather forecast through an in-context ridge regression (`tirex_xreg`, `timesfm_xreg`).
> - The longest contexts each model accepts:
>   - TimesFM-2.5 up to 16,360 h (`timesfm_ctx16k`);
>   - Chronos-2-base at 8,192 h (`chronos2_base_ctx8192`).
> - [if run] TimesFM-3 and TiRex-2 zero-shot, with the covariates as known-future inputs (`timesfm3`, `tirex2`).
>
> **How the regression's settings were fixed:**
>
> - On development days before the first test day: 4 Jan–2 Jun 2022, three 30-day windows, six stations (COVARIATES.md, section 4).
> - Settings: Chronos-2's covariates plus the forecast clear-sky index, an 8,192-h fit window, ridge 1.
> - No test day was used.
>
> **Reporting:**
>
> - Every result is reported whatever it shows, marked ‡.
> - Reported as paired skill differences with 95% intervals and no tests:
>   - each new row against Chronos-B-long (NWP);
>   - each new row against LightGBM (NWP);
>   - `tirex_xreg` and `timesfm_xreg` against their univariate rows.
