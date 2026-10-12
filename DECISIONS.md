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
