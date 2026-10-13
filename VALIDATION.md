# Real-time selection of the zero-shot Chronos-2 configuration

Written 13 Oct 2026, before any forecast on an inner window existed and before any
validation score was computed. The declaration, with the date and time, is in
`DECISIONS.md` (13 Oct 2026); this file is the design and the plan of the runs.

## 1. The problem

RQ1 was answered with the best zero-shot configuration tested (DECISIONS.md, 4 Oct):
Chronos-B-long, the best of six zero-shot Chronos-2 configurations *on the test days*.
Picking the best of several configurations by their test skill overstates the winner's
skill (Cawley & Talbot 2010; Tibshirani & Tibshirani 2009), and the hyperparameters of a
forecasting model should be chosen ex ante, on data the test set never touches (Lago et
al. 2021, §5.7). Two things are wrong with the pick, and only the first can be fixed:

1. **Selection on the test days.** Fixed here: the configuration is chosen in real time,
   for each test window, on days before that window only.
2. **The candidate list was drawn up after the main run had been seen.** No later step
   undoes that. Every candidate and the selected row stay marked ‡ and get no test.

The bias that (1) removes is expected to be small: only Chronos-B-max is within noise of
Chronos-B-long on the test days (E14: −0.005 [−0.014, 0.005]), so the expected optimism
of picking the better of the two is about sd(D) / sqrt(2 pi) = 0.002 skill. The point of
the exercise is a defensible RQ1 row, not a different conclusion.

## 2. Candidates: the full grid

Chronos-2 {small, base} x context {336, 1,024, 2,048, 8,192} h, zero-shot: a factorial
grid over the documented context range of the model, rather than "the six that happened
to run". Six configurations exist; two are new and run on the test windows first
(`configs/grid.yaml`):

| Configuration | Short name | Checkpoint | Context | Test-window run |
|---|---|---|---:|---|
| `chronos2` | Chronos-S | small | 336 | panel v2 |
| `chronos2_ctx1024` | Chronos-S-mid ‡ | small | 1,024 | extras |
| `chronos2_ctx2048` | Chronos-S-long ‡ | small | 2,048 | extras |
| `chronos2_ctx8192` | Chronos-S-max ‡ | small | 8,192 | **new** (`grid.yaml`) |
| `chronos2_base` | Chronos-B ‡ | base | 336 | extras |
| `chronos2_base_ctx1024` | Chronos-B-mid ‡ | base | 1,024 | **new** (`grid.yaml`) |
| `chronos2_base_ctx2048` | Chronos-B-long ‡ | base | 2,048 | extras |
| `chronos2_base_ctx8192` | Chronos-B-max ‡ | base | 8,192 | covariates |

The new configurations copy their grid neighbours' params, changing only the
checkpoint: Chronos-S-max those of Chronos-B-max (`[2048, 8192]` h, 345 days of history),
Chronos-B-mid those of Chronos-S-mid (1,024 h, 90 days).

## 3. Windows

The test calendar has a 60-day stride and 30-day test windows, so between two test
windows lie 30 days: the last is fold k's embargo day, the 29 before it are fold k's
**inner window**. No inner day is a test day, and no inner day is the embargo day.

| Fold | Inner window (29 days) | Embargo day | Test window (30 days) |
|---:|---|---|---|
| 0 | 5 Jun – 3 Jul 2022 | 4 Jul 2022 | 5 Jul – 3 Aug 2022 |
| 1 | 4 Aug – 1 Sep 2022 | 2 Sep 2022 | 3 Sep – 2 Oct 2022 |
| … | + 60 days per fold | | |
| 8 | 28 Sep – 26 Oct 2023 | 27 Oct 2023 | 28 Oct – 26 Nov 2023 |

`configs/validation.yaml` runs the inner windows with the test procedure unchanged
(the runner's day-ahead protocol, issue at local midnight, the history strictly before
it, the daytime mask, scoring version 2); only the splitter's first day (5 Jun 2022)
and window length (29 days) differ. The splitter's guard on the first fold's training
days drops from 365 to 330, because 30 days earlier SQ567 (from 1 Jul 2021) has 338; the
guard is not the clear-sky template window (the template is fitted once per station in
the ETL), and no reported forecast depends on it. Known overlap: the inner window of
fold 0 shares 26 days (5–30 Jun 2022) with SQ567's template year, so the clear-sky index
is in-sample there for every candidate alike.

Each inner window is run for the eight candidates and for CLIPER, the criterion's
reference, in both weather settings at the six stations.

## 4. The rule (`experiments/select_zero_shot.py`)

Number the windows in date order: fold k's inner window is window 2k, its test window
2k + 1.

1. **Pool of fold k:** windows 0..2k, every window that ends before fold k's embargo day:
   the inner windows of folds 0..k and the test windows of folds 0..k − 1. At fold k's
   first issue time all of them are history (recursive, real-time selection: Pesaran &
   Timmermann 1995; Lago et al. 2021 re-tune LEAR's penalty at every recalibration).
   Fold 0's pool is its inner window, 29 days; fold 8's is 501 days. Each fold's test
   window stays out of sample for that fold's pick.
2. **Criterion:** skill over CLIPER on daytime hours, each station pooled over the
   pool's days, then the mean over the six stations: the headline skill of the thesis,
   restricted to the pool (`report.Bootstrap.skill`). Separately for NWP and No weather.
3. **Tie-break:** a configuration is tied with the best if its pooled skill is within
   one standard error of the best's, the standard error being that of their *paired*
   skill difference, from the 7-day moving-block bootstrap over the pool's days (10,000
   resamples, seed 0, the scheme of the thesis intervals). Of the tied configurations
   the cheapest is picked: small before base, then the shorter context. The paired
   standard error is used because all candidates are scored on the same days, so the
   day-to-day weather that moves every candidate's skill together is not noise in the
   comparison. The rule leans away from the configurations that won on the test days
   (base, long), so it cannot have been designed to land on them.
4. **Selected row (`chronos2_selected`, Chronos-sel ‡):** for every station, test window
   k is the test-window forecast of fold k's pick, in each setting. It is assembled from
   existing forecasts and scored like any run by `experiments/report.py`. Its time per
   window is that of the configuration picked; running all eight candidates every day,
   which the selection needs in operation, is reported separately.
5. **Declared sensitivity variants** (summary only, never in place of the rule): plain
   argmax over the pool; the classic one-standard-error rule, with the standard error of
   the best configuration's own pooled skill.

## 5. How big was the bias of the hindsight pick?

On the 270 test days only, from existing forecasts:

- **Tibshirani & Tibshirani (2009), adapted to skill:** the mean over the nine test
  windows of (best window skill among the candidates − window skill of the configuration
  best over all test days). For the eight candidates and for the original six.
- **Analytic:** sd(D) / sqrt(2 pi) for the two best configurations, D their paired skill
  difference over the test days (the expected excess of the better of two equal noisy
  estimates).

The gap between Chronos-sel and Chronos-B-long (V3) is *not* the selection bias: it is
the optimism of the hindsight pick plus the real cost, or gain, of choosing in real time
on less data, and Chronos-sel can score above Chronos-B-long.

## 6. Reporting

- Chronos-sel ‡ is reported whatever it shows, with its selection path (appendix table
  `selection_path`).
- Main tables and figures (T1, T3, T4, F1–F6): Chronos-S (fixed before the main run: the
  confirmatory configuration of RQ1 and of P1, P4, P5, P6, P7 and P10), Chronos-sel ‡ (the
  best-practice zero-shot result, chosen without the test days) and Chronos-B-long ‡ (best
  on the test days, in hindsight, next to the optimism estimate). T2 adds Chronos-S-max,
  Chronos-B-mid and Chronos-sel.
- Paired skill differences, NWP, intervals and no tests (`report.POST_HOC`):
  V1 LightGBM vs Chronos-sel; V2 Chronos-S-FT vs Chronos-sel; V3 Chronos-sel vs
  Chronos-B-long; V4 Chronos-sel vs Chronos-S.
- Wording: "zero-shot weights, inference settings selected on earlier days", not "no
  tuning"; the ‡ is not removed; no claim that Chronos-sel or Chronos-B-long beats or
  equals LightGBM.

## 7. Not done, by decision

- **No learning-rate search for fine-tuning.** The recipe's 10⁻⁵ is AutoGluon's
  Chronos-2 `fine_tune_lr` default (applied there with low-rank adapters) and the
  learning rate of the official Chronos-2 quickstart; the smoke test only rejected darts'
  Adam default, 10⁻³, which wrecks the pretrained weights. chronos-forecasting's own
  `Chronos2Pipeline.fit` defaults to 10⁻⁶ for full fine-tuning (checked in the installed
  2.2.2), and AutoGluon 1.6.1's documentation recommends about 10⁻⁶ for full
  fine-tuning; 10⁻⁶ was not tried. A search over the 18 inner windows would take about
  12 h and could not change P2, which is not significant.
- **No selection for the purpose-trained models** (e.g. LightGBM lags 24/48/168 h through
  the same rule): about 20 h of runs. Only the zero-shot side gets a selection, so the
  asymmetry favours it; it is stated as a limitation.
- **No Model Confidence Set** (Hansen, Lunde & Nason 2011): optional and descriptive.
- **Not the development period** (4 Jan – 2 Jun 2022) for selection: it lies inside the
  template year, cannot host 8,192-h contexts, covers other seasons, and its skills run
  high.

## 8. Runs and commands

From `~/Thesis/solar-pv-forecasting` with `conda activate pv` and `export HF_HUB_OFFLINE=1`,
one batch at a time:

```bash
# G: the two new candidates on the test windows (24 runs, about 0.75 h)
python experiments/run_panel.py --config configs/grid.yaml \
    --results results/grid/results.parquet --predictions-dir results/grid/predictions \
    --logs-dir results/grid/logs
# V: the eight candidates and CLIPER on the inner windows (108 runs, about 2.7 h)
python experiments/run_panel.py --config configs/validation.yaml \
    --results results/validation/results.parquet \
    --predictions-dir results/validation/predictions --logs-dir results/validation/logs
# selection, the selected row, the optimism estimates and the selection-path table
python experiments/select_zero_shot.py
# thesis tables and figures, now with Chronos-S-max, Chronos-B-mid and Chronos-sel
python experiments/report.py
```

## 9. Checks after the runs

- every run ok in its ledger;
- grid: 12 prediction files per configuration, each with 6,480 rows, 9 folds, no
  missing forecast, the same time index, folds and truth as the panel's file for that
  station;
- inner windows: 108 files, each with 6,264 rows (9 x 29 x 24), 9 folds, no missing
  forecast; at each station the same time index and truth for all nine models;
- no line with WARNING, Traceback or "without covariates" in the run logs;
- NWP and No-weather forecasts differ for every candidate at every station;
- `results/results.parquet`, `results/extra/` and `results/covariates/` unchanged
  (sha256 before and after).

## Amendment (13 Oct 2026, 13:39)

Everything above is the file hashed at 13:18:41 (`results/validation/PREREGISTRATION.txt`).
After a code review, and before any inner-window forecast existed, the optimism estimates
of section 5 were relabelled (Tibshirani & Tibshirani conservative, the analytic value the
two-way equal-skill case), a centred paired-bootstrap estimate was added, and the wording
of section 6 was made precise: see the 13:39 amendment entry of `DECISIONS.md`.
