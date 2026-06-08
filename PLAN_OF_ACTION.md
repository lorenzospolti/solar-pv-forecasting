# Plan of Action — Solar PV Day-Ahead Forecasting Thesis

_Prepared 8 June 2026 · Bachelor's thesis, Università di Milano-Bicocca, target submission Sept 2026._

**Thesis question (Variant B):** do time-series *foundation models* (Chronos-2, TimesFM-2.5, TinyTimeMixer, Moirai-2) — zero-shot and fine-tuned — match or beat purpose-trained models (smart persistence, SARIMA, LightGBM, LSTM) for day-ahead hourly PV power on consumer hardware (M4, 16 GB), and at what compute cost? Scored in clear-sky-index space, by skill over smart persistence, with probabilistic calibration.

> **✅ Implemented in the 8 Jun 2026 session** (committed `fddfba0`, merged to `main`, tag `v0.3-foundation-wired`):
> Phase 0 (idempotent results store, stray file removed) · Phase 1 (SARIMA/LightGBM/LSTM enabled; `--site`/`--weather-mode`/`--all-models` flags) · Phase 2 (per-station nameplate capacity from the Brick `.ttl`; CYT now 25 kW, caches regenerated) · Phase 3 (Chronos-2/TimesFM-2.5/TiRex wired, zero-shot + fine-tune via an `arch`/label split; `darts>=0.43` pin; Moirai/TTM guarded) · Phase 4 (`experiments/report.py` → tables + figures) · Phase 5 (manuscript skeleton + `references.bib`). Verified: ruff clean, 8/8 tests pass, synthetic E2E idempotent, real CYT baselines over 6 folds.
>
> **⛔ Still requires your Mac (can't run in the sandbox):** rebuild the env at `darts>=0.43`; run the full ladder (SARIMA/LightGBM/LSTM, then foundation models) on real stations × {oracle, nwp}; download the HF foundation checkpoints; decide the station panel and the Moirai/TTM roster (§5); write the manuscript. The phases below are the durable roadmap; checkboxes left unticked are the parts that still need a real run or your decision.

---

## 1. Where the project actually stands

**Solid and working:**

- **Harness spine is complete** (~1,900 LOC, all modules substantive): `Forecaster` ABC, `Runner` looping `model × fold × site`, expanding-window blocked walk-forward `Splitter`, clear-sky-index `normalize`, pvlib `clearsky`, and a full `Evaluator` (RMSE/MAE/MBE/R²/nRMSE/nMAE, skill score, pinball, CRPS, PICP-80, interval width, reliability, monotone-quantile enforcement).
- **Baselines run end-to-end** — persistence, smart persistence, seasonal-naive produced 15 result rows (5 folds each) on the synthetic station.
- **Real ETL works.** One station, **CYT (Station 1)**, is fully processed into both `oracle` and `nwp` frames: 23,303 hourly rows, 2021-05-04 → 2023-12-31, capacity ≈ 22.7 kW.
- **NWP weather feed works.** Open-Meteo archived day-ahead forecast cached for 2021–2023; `nwp` frame carries `cloud_cover`, `ghi`, `temp_air`, etc.
- **Environment builds.** conda `pv` env OK; forecast extras installed (darts 0.41.0, chronos-forecasting 2.2.2, torch 2.10, lightgbm 4.6, pvlib 0.15.1, statsmodels 0.14.6; Python 3.12, macOS arm64).
- **SARIMA / LightGBM / LSTM are wired** in `darts_wrapper.py` (`_WIRED = {sarima, lightgbm, lstm}`) — clear-sky-index target, future covariates, quantile/sample paths, MPS-aware LSTM.

**Partial / not yet done:**

- **No real-data results exist.** `results.parquet` is 100% synthetic baselines. SARIMA/LightGBM/LSTM have never been run on real data and are commented out in `configs/experiment.yaml`.
- **Only 1 of ~60 stations processed.** `list_stations()` discovers all sites, but ETL has been run for CYT (Station 1) only.
- **Foundation models are not wired.** `DartsForecaster._build()` has no branch for them; calling one raises `NotImplementedError`. `chronos2`/`timesfm` are mapped in `DARTS_CLASSES`; `ttm`/`moirai` are not mapped at all.
- **No reporting layer.** `thesis/figures/` and `thesis/tables/` are empty; `notebooks/` is empty. No script turns `results.parquet` into the thesis's tables/figures.
- **No manuscript.** There is no thesis document anywhere in the repo yet.

---

## 2. Critical gaps, risks & decisions

1. **Fine-tuning needs a darts upgrade.** Verified against the darts changelog: your installed **0.41.0 has `Chronos2Model` (zero-shot, since 0.40) and `TimesFM2p5Model` (new in 0.41)**, but **`enable_finetuning` only landed in 0.42.0**. The "fine-tuned foundation models" half of the thesis therefore requires **darts ≥ 0.42** (0.43 adds refinements; 0.44 adds `TiRexModel`; master adds IBM `PatchTSTFMModel`). Upgrading means re-validating the env.
2. **Moirai-2 has no native darts class** (none found through 0.44). Options: integrate Salesforce `uni2ts` directly, substitute another model (e.g. `TiRexModel` in darts 0.44, or IBM `PatchTSTFMModel`), or drop Moirai and state the scope cut. **Decision needed.**
3. **TinyTimeMixer (TTM):** not in `DARTS_CLASSES`. IBM's native darts option (`PatchTSTFMModel`) is master-only; "true" TTM needs IBM `granite-tsfm`. **Decision needed** (which TTM, which source).
4. **Known env conflict:** `gluonts 0.16.2` requires `numpy<2.2` but `numpy 2.4.6` is installed. Darts foundation models use HF `transformers`/`safetensors` (not gluonts), so this likely won't block them — but pin it down before it bites, ideally during the darts upgrade.
5. **Capacity is a 0.999-quantile proxy, not nameplate.** Per-station nameplate lives in the Brick `.ttl` metadata but isn't wired. Affects every normalized metric (nRMSE, nMAE, CRPS). Low effort, improves rigor.
6. **Git is mid-stream.** The repo has two commits ("Week 1" scaffold; "Weeks 3–5" ETL + model wrappers) and you're on an **unmerged feature branch `week3-5-data-etl-and-models`**; `.gitignore` is already in place. The branch isn't merged to `main` and the first real results aren't captured yet. Keep committing per phase, merge to `main` at each milestone, and tag.
7. **`append_results` doesn't deduplicate** by `config_hash` (README implies it does) — re-running a config appends duplicate rows. Add idempotent upsert before bulk runs accumulate.
8. **Housekeeping:** stray empty `Untitled-1.py` at the Thesis root; empty `configs/models/` (models are inline in `experiment.yaml`).

---

## 3. Phased plan

Effort tags: **S** ≤1 day · **M** a few days · **L** ~1 week+. Sequencing assumes ~12 weeks to submission.

### Phase 0 — Hygiene & guardrails (this week, S)

- [ ] Commit current work and **merge `week3-5-data-etl-and-models` → `main`** (`.gitignore` is already in place); tag the milestone so the harness + ETL state is a known-good baseline.
- [ ] Delete `Untitled-1.py`; remove or populate the empty `configs/models/`.
- [ ] Make `append_results` idempotent (drop existing rows with the same `config_hash` + `daytime_only` before appending), so re-runs overwrite rather than duplicate.
- [ ] Run `make test` and `make check` on the Mac to confirm the harness + toolchain are green before generating real results.

### Phase 1 — First real-data results table (week 1, M)

- [ ] In `configs/experiment.yaml`, uncomment **sarima, lightgbm, lstm** (baselines stay on).
- [ ] Run CYT (Station 1), **oracle** weather: `python experiments/run.py --config configs/experiment.yaml`.
- [ ] Duplicate the config with `weather_mode: nwp` and run again (oracle = perfect-foresight upper bound; nwp = realistic). This is the headline oracle-vs-nwp contrast.
- [ ] Sanity-check: smart persistence skill = 0 by construction; classical/ML models should land skill > 0 on daytime mask; LSTM trains on MPS without error. Eyeball one fold's predictions vs truth.
- [ ] Capture wall-clock from the `fit_predict_s` column — this is the compute-cost baseline the thesis needs.

### Phase 2 — Scale to a station panel (week 1–2, M)

- [ ] Choose a **representative subset of stations** (suggest 6–10 spanning capacity, orientation, and the with/without panel-level-optimizer split) rather than all 60 — keeps runtime tractable on an M4 and is defensible. _(See open decisions.)_
- [ ] Batch-process ETL for the subset (loop `load_station(site, weather_mode=…)` for oracle + nwp; frames cache to `data/processed/`).
- [ ] Wire `capacity_kw` from the Brick `.ttl` metadata in `data.py` (replace the 0.999-quantile proxy) for the chosen stations.
- [ ] Run the full classical/ML/deep ladder across the subset × {oracle, nwp}. Commit `results.parquet`.

### Phase 3 — Foundation models: the core contribution (weeks 2–4, L)

- [ ] **Decide the darts version** and upgrade (≥0.42 for fine-tuning; 0.43/0.44 for stability + TiRex). Rebuild env, re-run `make test`/`make check`, resolve the numpy/gluonts pin. Record the new env fingerprint.
- [ ] Wire **Chronos-2** and **TimesFM-2.5** in `DartsForecaster._build()` (add `elif` branches; map config params; handle the `FoundationModel` predict path). Start **zero-shot** — no training, just `predict` over each test day's history.
- [ ] Add **fine-tuned** variants via `enable_finetuning` (full + selectively frozen layers); keep zero-shot and fine-tuned as distinct `model` names so both land in `results.parquet`.
- [ ] Resolve **TTM** and **Moirai-2** per the decisions in §2 (wire, substitute, or document as scope cut).
- [ ] **Instrument compute cost** explicitly: peak RAM, wall-clock, and (zero-shot vs fine-tune) GPU/MPS time per model — this is half the thesis question, not an afterthought.

### Phase 4 — Analysis & reporting layer (week 4–5, M)

- [ ] Add `experiments/report.py` (or a notebook) that reads `results.parquet` and emits the thesis deliverables into `thesis/tables/` and `thesis/figures/`:
  - [ ] Master results table: nRMSE / nMAE / skill / CRPS per model × weather-mode × {daytime, all-hours}, aggregated over folds & stations with dispersion.
  - [ ] Skill-score bar/box plot vs smart persistence.
  - [ ] Reliability diagrams + PICP/sharpness for the probabilistic models.
  - [ ] **Skill-vs-compute scatter** (the headline "at what cost" figure).
  - [ ] Per-station and per-fold breakdowns for robustness discussion.

### Phase 5 — Manuscript (parallel from week 4, L)

- [ ] Set up the thesis document (LaTeX or the department template) with the standard skeleton: intro, related work, data, method (clear-sky-index harness, splits, metrics), experiments, results, discussion, conclusion, reproducibility appendix.
- [ ] Draft method + data now (stable); write results against generated tables/figures as they land.
- [ ] Maintain the reading list / citations alongside (Yang et al. verification framework, the four foundation-model papers, the HKUST Dryad dataset DOI).

### Phase 6 — Buffer, reproduce, defend (final ~2 weeks, M)

- [ ] `make reproduce` end-to-end from clean checkout to confirm every number in the manuscript regenerates.
- [ ] Freeze env (`requirements-lock.txt`), tag a release, archive `results.parquet`.
- [ ] Proof, prepare defense slides, leave slack for advisor feedback.

---

## 4. Do these three things first (today/tomorrow)

1. **Merge the feature branch → `main` + tag** — capture the harness + ETL milestone (you're on an unmerged `week3-5-data-etl-and-models`).
2. **Enable sarima/lightgbm/lstm and run CYT (Station 1) oracle + nwp** — turns "harness verified on synthetic" into your first real results table, the single biggest unblock.
3. **Lock the foundation-model scope** (§5) so the darts-version decision and Phase 3 can start without rework.

## 5. Open decisions (need your input)

- **Station panel:** which 6–10 stations? (Balance capacity range and the optimizer / no-optimizer split.) Or argue for all 60 if runtime allows.
- **Foundation-model roster:** keep all four (Chronos-2, TimesFM-2.5, TTM, Moirai-2), or drop the ones without native darts support (Moirai-2; TTM unless using `PatchTSTFMModel`/`granite-tsfm`)? This is the biggest scope lever.
- **darts upgrade timing:** upgrade to ≥0.42 now (before Phase 1) so the whole results set uses one version, or after Phase 1–2 to avoid re-running classical models? Recommendation: upgrade **before** Phase 3 but **after** Phase 1's quick win, and re-run classical models once on the final version for consistency.
- **Compute budget:** how much M4 wall-clock per run are you willing to spend? Drives station count × fold count × fine-tune epochs.

---

_State verified directly against the repo on 8 June 2026: source modules, `results.parquet`, processed `data/`, env logs, and the darts changelog (versions 0.39–0.44 + master)._
