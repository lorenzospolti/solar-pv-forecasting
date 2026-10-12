# Experiments and Results {#sec:results}

<!-- TODO(Lorenzo): drafted with Claude on 5 Oct 2026 from the tables and the Results notes; rewrite
it in your own words before submission (UniPV AI guidelines: no unrevised AI-written paragraphs).
Every number is taken from the included tables; regenerate them with `make report`. -->

This chapter answers the four research questions in turn. Every skill score is the RMSE skill
over CLIPER on daytime hours, averaged over the six stations (@sec:method-metrics): a skill of 0.28
means a 28% lower RMSE than CLIPER, and 0 means no better. The text gives rounded values, and an
interval (95%, from a bootstrap over test days, @sec:method-tests) only where a claim depends on it;
the tables give the exact values. Configurations added after the main run had been seen are marked
‡ (post hoc), and comparisons that involve them are reported without tests. Supporting tables are
in Appendix [-@sec:appendix-results].

## Zero-shot accuracy (RQ1) {#sec:results-accuracy}

**With weather forecasts, the best zero-shot configuration is as accurate as the best trained
models; the configuration fixed in advance is not. Without weather forecasts, no model beats
CLIPER.**

@Fig:skill-by-model shows every model in both settings. With weather forecasts, three models lead:
LightGBM (0.28), the zero-shot Chronos-B-long ‡ (0.27) and the fine-tuned Chronos-S-FT (0.26).
They are within 0.03 of one another, and every paired difference among them has an interval that
contains zero (@tbl:significance), so they cannot be told apart. The LSTM and SARIMA follow at
about 0.2. Chronos-S, the zero-shot configuration fixed before the main run, trails at 0.12, well
behind LightGBM. In absolute terms, LightGBM's RMSE is 11% of nameplate capacity, against 15.5% for
CLIPER (@tbl:master-skill): on a 55 kW roof, a typical hourly error of about 6 kW instead of 8.5 kW.

Zero-shot accuracy depends on the configuration more than on the model family
(@tbl:chronos2-configs): a longer context lifts the small checkpoint from 0.12 to 0.21, and the
base checkpoint with the same long context reaches 0.27. Chronos-B-long was picked as the best of
five zero-shot configurations tried after the main run, on the same test days, so this comparison
favours the zero-shot model.

Without weather forecasts, the best models stay within 0.01 of CLIPER, with intervals that contain
zero, and none beats it. TiRex and TimesFM, which take no covariates, are in the same position:
TiRex at 0.00 and TimesFM at −0.05. The LSTM (−0.09) and Smart persistence (−0.13) are worse than
CLIPER.

![Weather forecasts supply nearly all the skill: with them (solid bars), three models reach about
0.27; without them (hatched), no model beats CLIPER, the zero line. Skill over CLIPER of the models
of @tbl:master-skill, with 95% intervals from the moving-block bootstrap over test days. ‡ Post
hoc.](figures/skill_by_model.pdf){#fig:skill-by-model width=100%}

```{.include}
tables/master_skill.md
```

```{.include}
tables/significance.md
```

## Fine-tuning (RQ2) {#sec:results-finetune}

**Fine-tuning lifts the short-context model to the level of LightGBM, but a longer context gives
most of that gain without any training.**

At the context fixed before the main run, fine-tuning adds 0.14 to zero-shot Chronos-S, a
significant gain (@tbl:significance), and three fine-tuning seeds agree within 0.01
(@tbl:chronos2-configs). A longer context gives the zero-shot model about two thirds of that gain
(0.09 of 0.14) ‡. The fine-tuned model stays narrowly ahead of Chronos-S-long ‡ (0.046
[0.003, 0.087]) but cannot be told apart from Chronos-B-long ‡ (−0.016 [−0.051, 0.018]).
Fine-tuning at a long context was not tested: at the recipe's batch size it does not fit in the
laptop's 16 GB. Without weather forecasts, fine-tuning adds nothing measurable (0.01, not
significant).

The cost of fine-tuning, about three minutes per test window, is the same as that of fitting
LightGBM (@sec:results-calibration).

```{.include}
tables/chronos2_configs.md
```

## What weather forecasts add (RQ3) {#sec:results-weather}

**Weather forecasts supply nearly all the skill over CLIPER, for the models that can use them.**

Weather forecasts add about 0.27 to LightGBM and to Chronos-B-long ‡, 0.25 to Chronos-S-FT and
0.12 to Chronos-S; the two gains that were tested, for LightGBM and Chronos-S, are significant
(@tbl:significance). TimesFM and TiRex cannot use weather forecasts through their wrappers
(@sec:method-covariates), so they gain nothing from them. These gains are an upper
bound: the archived weather forecasts reach the scored hours at shorter lead times than a forecast
issued at midnight could use (@sec:data-weather).

## Calibration and compute (RQ4) {#sec:results-calibration}

**With weather forecasts, the fine-tuned model's forecast intervals are the closest to calibrated: LightGBM's are too
narrow and the zero-shot models' too wide. The three leading models score the same on the
quantile score, and all of them run comfortably on the laptop.**

A central 80% interval should contain 80% of the observations. With weather forecasts, LightGBM's
contains 71% [67%, 74%], so it is too narrow; those of Chronos-S and Chronos-B-long ‡ contain
about 85%, too wide; Chronos-S-FT's contains 77% [74%, 80%] (@tbl:probabilistic).
@Fig:reliability shows that the miscalibration lies in the outer quantiles. On the quantile score, which
rewards both accuracy and calibration, Chronos-B-long ‡, Chronos-S-FT and LightGBM tie at 0.029 of
capacity, so none of them wins the probabilistic comparison.

![The fine-tuned model is the closest to the diagonal of perfect calibration; LightGBM's outer
quantiles are too close to the median and the zero-shot models' too far from it. Observed share of
scored hours at or below each forecast quantile against its nominal level, for the probabilistic
models of @tbl:probabilistic. ‡ Post hoc.](figures/reliability.pdf){#fig:reliability width=100%}

```{.include}
tables/probabilistic.md
```

@Fig:skill-vs-compute sets skill against cost. The zero-shot models need no training: Chronos-S
issues a 30-day test window of forecasts in 2.5 seconds and Chronos-B-long ‡ in 12 seconds, while
LightGBM and Chronos-S-FT each spend about three minutes per window on fitting (@tbl:compute).
Fit times vary by 20–25% between sessions, so these two cost the same within noise. Every model
fits in the laptop's memory; SARIMA needs the most, 7.9 GB, and no other model more than 2.2 GB,
although the memory of the GPU is not counted, so the neural models' figures are lower bounds. The
144 runs of the main run took 18.5 hours.

![The best zero-shot configuration matches the trained leaders at a fraction of their cost: skill
over CLIPER against the median fit + predict time per test window (log scale), with weather
forecasts (filled) and without (open), for the models of @tbl:master-skill. ‡ Post
hoc.](figures/skill_vs_compute.pdf){#fig:skill-vs-compute width=100%}

```{.include}
tables/compute.md
```

## Stability across test windows {#sec:results-windows}

**The order of the three leaders changes from window to window; the last place of Chronos-S does
not.**

First place moves between LightGBM (four of the nine windows), Chronos-S-FT (four) and
Chronos-B-long ‡ (one) (@fig:skill-by-fold). LightGBM's small overall lead over the fine-tuned model
comes from one window, May 2023, without which the order reverses (@tbl:p2-robustness). Chronos-S is
last, or level with the LSTM, in every window. The test calendar leaves February and April
untested (@sec:method-protocol).

![The leaders trade places from window to window: skill over CLIPER of each model in each 30-day
test window, averaged over the six stations; columns are labelled by the month that holds the middle
of the window. The values are in @tbl:skill-by-fold. ‡ Post
hoc.](figures/skill_by_fold.pdf){#fig:skill-by-fold width=100%}

## Summary {#sec:results-summary}

@Tbl:results-summary collects the four answers and where to find the evidence for each.

: The answers to the four research questions, with the evidence for each. {#tbl:results-summary}

| Question | Answer | Evidence |
|:--------------------|:-------------------------------------------------------------|:--------------------|
| RQ1. Zero-shot accuracy | Without weather forecasts, no better than CLIPER. With them, the best zero-shot configuration (post hoc) matches LightGBM and the fine-tuned model; the configuration fixed in advance trails them. | @fig:skill-by-model; Tables [-@tbl:master-skill]–[-@tbl:chronos2-configs] |
| RQ2. Fine-tuning | Adds 0.14 at the short context; a longer context gives two thirds of that without training. Costs about three minutes per window, like LightGBM. | Tables [-@tbl:significance], [-@tbl:chronos2-configs], [-@tbl:compute] |
| RQ3. Weather forecasts | Nearly all the skill over CLIPER, for the models that can use them; an upper bound. | @fig:skill-by-model; @tbl:master-skill |
| RQ4. Calibration and compute | With weather forecasts, fine-tuned closest to calibrated, LightGBM too narrow, zero-shot too wide; equal quantile scores. Every model fits in 16 GB; at most four minutes per window. | Figures [-@fig:reliability], [-@fig:skill-vs-compute]; Tables [-@tbl:probabilistic], [-@tbl:compute] |
