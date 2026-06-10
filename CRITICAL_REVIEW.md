# Critical Review — 10 June 2026

Independent audit of the repo: all source modules read, data and results inspected programmatically, claims in README/PLAN_OF_ACTION cross-checked against the actual code and files.

## Verdict

The strategy is sound and you are on the right path. The thesis question (foundation models vs purpose-trained, zero-shot + fine-tuned, on consumer hardware, with compute cost) is well-posed and feasible by September. The harness design — one `Forecaster` contract, one `Runner`, identical causal folds, clear-sky-index target, skill over smart persistence, probabilistic scoring — follows the solar-verification canon and is above bachelor's level. The plan's sequencing (real CYT results → station panel → foundation models → report → manuscript) is right.

However, I found **two confirmed bugs and one methodological leak** that affect every result you would have produced, plus several protocol weaknesses. The good news: no real-data results exist yet, so nothing has to be re-run. Fix these *before* the first real run, not after.

---

## A. Confirmed errors (fix before any real run)

### A1. NWP irradiance is misaligned by one hour — verified empirically
Open-Meteo radiation variables are **backward-averaged** (value at HH = mean over [HH−1, HH]); your frames are labeled hour-beginning (resampled mean over [HH, HH+1]). Measured-vs-forecast daytime GHI correlation on CYT: **0.656 at 0 shift → 0.768 at −1 h**; the mean diurnal nwp-GHI peak sits at 13:00 local vs 12:00 for measured GHI / p_clear / power. Every nwp-mode model would be handicapped by a systematically lagged irradiance covariate.

**Fix:** in `weather.py`, shift the radiation columns (`ghi`, `ghi_direct`, `ghi_diffuse`) by −1 h after fetch (instantaneous vars temp/wind/rh/cloud_cover are valid at HH — either accept or average consecutive instants). Delete and regenerate `data/external/openmeteo_*.parquet` consumers, i.e. the `*__nwp.parquet` caches.

### A2. "Day-ahead" blocks are UTC days, not Hong Kong days
The index is UTC; `Splitter` and `Runner` normalize on UTC days, so a "test day" runs 08:00→07:59 HKT and forecasts are issued at 08:00 local. Models with recent context (LSTM 168 h window, SARIMA 60-day refit, foundation in-context) see *the same morning's* conditions for most of the daylight they're predicting. That is intraday information advantage mislabeled as day-ahead — it inflates skill vs the literature's definition and a referee would flag it.

**Fix:** define day blocks and issue boundaries in local time: use `index.tz_convert("Asia/Hong_Kong").normalize()` in `splits.py` and `_day_ahead_predictions` (HK has no DST, so it's unambiguous). Issue at 00:00 HKT predicting the next local day.

### A3. Clear-sky template leaks test data into the target transform
`attach_clearsky` is applied once to the full 2021–2023 frame, so `empirical_clearsky_power`'s per-(month,hour) 0.95-quantile performance ratio — and therefore `p_clear` and the target `k` — is fit on data from every future test period. `normalize.py`'s "fit on TRAIN ONLY" promise does not hold for the most important transform. Mitigation: the same template feeds all models *and* the smart-persistence reference, so skill scores are only mildly distorted — but absolute nRMSE/CRPS and the leakage-free claim are compromised.

**Pragmatic fix (recommended):** fit the performance-ratio table on the **first 365 days only** (the initial train window, which is inside every fold's train set by construction) and apply it to the whole index. One localized change in `data.py`, zero leakage, and you can note residual drift (degradation/soiling) as a documented simplification. The fully per-fold transform is cleaner but a bigger refactor; not worth it now.

### A4. Reporting layer would blend synthetic rows into thesis tables
`report.py` aggregates over everything grouped by `(model, weather_mode, daytime_only)` — `site` is not a group key and synthetic rows aren't filtered. Worse, the current `results.parquet` contains **only 15 stale synthetic rows from an older schema** (no `daytime_only`, no `weather_mode` columns), despite the plan's claim that real CYT baselines were run — those rows were never persisted. Stale + real rows would silently average together.

**Fix:** delete `results/results.parquet` (it holds nothing of value); add `site` to `GROUP_CANDIDATES`; exclude `site == "synthetic"` from thesis tables/figures.

### A5. Reliability diagram has no data path
README/plan promise reliability diagrams, `evaluation.reliability()` exists — but the runner never calls it and no per-quantile coverage is stored, so the promised figure can never be built. **Fix:** store per-quantile empirical coverage (e.g. `cov_q0.1 … cov_q0.9`) in each results row, and add the reliability figure to `report.py`.

---

## B. Protocol weaknesses (decide / document)

1. **Test folds cover only May–Oct 2022.** With `initial_train_days=365, test_days=30, step=30, max_folds=6` on data starting 2021-05-04, you never test on winter and never touch 2023 — 14 months of data unused and a seasonally biased evaluation. Recommend `step_days≈45–60` and `max_folds` ~10 (or cap-free) so test windows span a full year through 2023; trade station count, not seasonal coverage.
2. **The "nwp" feed is not a frozen day-ahead forecast.** Your docstring caveat is honest and correct (Historical Forecast API ≈ stitched near-analysis; Previous-Runs API only starts 2024). Two sharpening points: (a) the nwp frame still carries **measured** tower `rainfall`/`slp`/`visibility` — unused by default `WEATHER_COLS`, but a latent leak if anyone widens the covariate list; drop them in nwp mode. (b) In the thesis, label the contrast as *oracle vs best-available archived forecast* and never claim operational day-ahead realism. The conclusion direction is safe: if foundation models lose under near-perfect weather, they lose under worse.
3. **Point-forecast inconsistency.** Quantile models (LightGBM, SARIMA, foundation) emit the **median** as `y_hat`; LSTM/baselines emit a direct point. RMSE-optimal would be the mean — the median slightly handicaps quantile models on RMSE/skill. Either also emit the predictive mean, or state the choice explicitly.
4. **Compute-cost metric is contaminated.** `fit_predict_s` includes the reference's predict loop, conflates fit and predict, and `run.py` **fits every model twice** (once per daytime/all-hours mask) even though predictions are mask-independent. For the "at what cost" headline: time fit and predict separately, compute both masks from one prediction pass (halves total runtime), and add peak-RSS (`resource.getrusage`) for the memory story.
5. **Future frame hands models the realized target.** `predict(history, future)` receives `full.loc[day_idx]` including `power` and `k`; only a docstring forbids reading them. Drop target-derived columns in the runner (defense in depth) and assert `len(common) == len(test_idx)` so a model can't silently get scored on a subset.
6. **Gaps become fake zeros in training.** Missing power (3.2% of CYT) → `k=0` via `fillna(0.0)`, so models train on false daytime zeros (eval masks them, training doesn't). Keep `k` NaN where power is NaN; `_target_series` already ffills for darts.
7. **Embargo is cosmetic at predict time (fine, but say it).** Rolling prediction conditions on all actuals before each test day (correct day-ahead practice), so the embargo only protects global-model *fitting*. SARIMA's 60-day trailing refit effectively trains into the test period's neighborhood while LightGBM/LSTM weights are frozen at fold-train end — both defensible ("information available at issue time"), but the asymmetry needs one explicit paragraph in the method chapter.
8. **`persistence` duplicates `seasonal_naive_24h`** (identical lag — your own results rows are bit-identical). Use 168 h for seasonal-naive or drop it.
9. **SARIMA quantile path is unverified.** `AutoARIMA` + `predict_likelihood_parameters=True` may not be supported by the statsforecast wrapper in your darts version; the except-clause only rescues foundation models. Test this first on the Mac; fallback is the `num_samples` sampling path you already wrote for fixed-order ARIMA.
10. **Station-name → TTL capacity matching is regex+alias based.** Fine for CYT; before the panel run, add a one-shot check that prints unmatched stations so silent 0.999-quantile fallbacks don't sneak into nRMSE denominators.

## C. Strategy calls (my recommendations on your open decisions)

- **Roster:** Chronos-2 + TimesFM-2.5 + TiRex via darts ≥ 0.44 (zero-shot + fine-tune where supported) as the core. **Drop Moirai-2** (uni2ts integration is heavy; weak expected payoff). **TTM is worth a stretch goal** via `granite-tsfm` with a thin custom `Forecaster` — your harness is model-agnostic, darts purity is not a thesis requirement, and a 1–5 M-parameter model is the *best possible fit* for the consumer-hardware narrative. If time runs out, the scope cut is defensible.
- **darts upgrade:** go straight to ≥ 0.44 **now**, before any real results, so everything runs on one version (re-running classical models later "for consistency" is wasted compute). Remove `gluonts` from the env to kill the numpy conflict — nothing in your stack uses it.
- **Station panel:** 6–8 stations stratified by capacity × optimizer/no-optimizer is right. Pick them by data completeness first (run a gap report), then stratify.
- **Compute budget:** fine-tune on the small checkpoints (`chronos-2-small`), few epochs, and if the panel × folds × FT grid is too slow, cut FT to 2–3 stations (keeping all models on identical folds within each station) rather than cutting folds.
- **Cheap, high-value ablation:** one foundation model run on raw power vs clear-sky index `k`. It directly supports your "score in k-space" methodology chapter and costs one config entry.

## D. Ordered plan of action

**Step 0 — this week: fixes (1–2 days, all in-sandbox testable except darts).**
1. A1 radiation shift + regenerate nwp caches; A2 local-day splitting/issuing; A3 train-window clear-sky fit; A4 report grouping + purge stale parquet; A5 reliability columns.
2. B4 timing split + single-fit-two-masks; B5 runner drops `power`/`k` from `future` + length assert; B6 NaN-k for gaps; B2a drop measured leftovers from nwp frames; B8 seasonal-naive→168 h.
3. Extend folds for seasonal coverage (B1). `make test && make check`, commit, tag `v0.4-protocol-fixes`. Switch back to `main` for daily work (you're still sitting on the merged feature branch).

**Step 1 — env + first real results (2–4 days).**
Rebuild env at darts ≥ 0.44 (drop gluonts), re-run smoke/tests. Run CYT full ladder (baselines + SARIMA/LightGBM/LSTM) × {oracle, nwp} on the fixed protocol; verify SARIMA quantile path (B9); eyeball one fold's predictions vs truth; sanity-check skill(smart persistence)=0 and nwp ≤ oracle skill. This is your first thesis table.

**Step 2 — station panel (week 2).**
Gap/quality report over all 60 stations → pick 6–8 + capacity-match audit (B10) → batch ETL → ladder runs. Commit `results.parquet`.

**Step 3 — foundation models (weeks 2–4, the long pole).**
Zero-shot Chronos-2/TimesFM-2.5/TiRex on panel × both weather modes; then fine-tuned variants (start CYT-only, expand if budget allows); instrument wall-clock + peak RAM per model. Raw-vs-k ablation on one model. TTM stretch goal last.

**Step 4 — report + manuscript (weeks 4–9).**
`make report` with per-site dispersion, reliability diagrams, skill-vs-compute scatter. Write data + method chapters now (they're stable); results as tables land. Include the honest-NWP paragraph (B2) and the embargo/refit paragraph (B7).

**Step 5 — freeze (weeks 10–12).**
Clean-checkout reproduction, env lock, tag, defense slides, buffer for advisor feedback.

---

*Everything in section A was verified against the actual files on 10 June 2026 (code read in full; CYT parquets, results store, git history inspected). Nothing here requires re-running existing experiments — there are none yet, which is exactly why now is the moment to fix the protocol.*
