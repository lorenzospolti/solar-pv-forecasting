# Discussion {#sec:discussion}

*Scope note: interpret, do not repeat, the Results, in this order. (1) Do foundation
models earn their compute? Chronos-S costs seconds per window but trails LightGBM and
Chronos-S-FT with NWP; Chronos-B-long (about 12\ s per window with NWP, no training) cannot
be told apart from LightGBM (E4) or Chronos-S-FT (E5), a post hoc selection; without weather
nothing beats CLIPER. (2) Zero-shot or fine-tuned: a longer context gives the small zero-shot
model about two thirds of what fine-tuning adds, and Chronos-B-long cannot be told apart from
Chronos-S-FT; fine-tuning at a long context is untested (at the recipe's batch size it does
not fit the laptop's memory). (3) What weather adds: almost all the skill over CLIPER, for the
models that take covariates, as an upper bound. (4) Calibration: LightGBM too narrow,
zero-shot too wide. Then the threats to validity.*

<!-- TODO(Lorenzo): write this chapter yourself; it is where your own interpretation counts.
Points raised by the 12 Oct review of Chapter 1, for you to cover in your own words: (a) Chapter 1's
promise that a zero-shot model needs no training meets the template, which is still fitted on a year of
each station's data; (b) why only Chronos-2 was fine-tuned; (c) where the result falls between FETS
[@obermeier2026fets] and Za'ter & Hodge [@zater2026empirical], which Chapter 1 sets up. -->

## Threats to validity {#sec:threats}

The largest threat concerns the weather inputs. The archived forecasts reach the scored
hours at lead times of about 0–12 hours, mostly from runs that start after the issue
time (@fig:data-leadtime), whereas a forecast issued at midnight would rely on the 06 UTC run of the previous
day, at lead times of 16–28 hours (@sec:data-weather). The NWP results are therefore an
upper bound, and the gap between the NWP and no-weather settings overstates what weather
forecasts add in operation. Other studies, with different lead-time contrasts, put the
cost of longer leads at 4–13% of the RMSE of PV forecasts at one site
[@dehghan2026pipeline] and under 10% of the irradiance error [@mayer2023comparing];
none covers this site. The archive
also changes model version within the test period: IFS Cycle 48r1 replaced the previous cycle on 27 June 2023 [@ecmwf_48r1], three days before
the seventh test window.

Several choices were made after test results had been seen, and each is stated where it
applies: the fine-tuning recipe, chosen on one test fold; the lag inputs of LightGBM,
corrected after a wiring bug showed on a test fold; the point-forecast rule and the
family of comparisons, after the first run, which covered the same test days; the extra
contexts, the base checkpoint and the extra seeds, after the main run; and CLIPER as the
headline reference (@tbl:decisions). Reporting RQ1 with the best zero-shot configuration favours the
zero-shot models. The comparisons that involve the later runs are reported without
tests.

The test sample is small. It holds 270 days in nine 30-day windows, with no test day in
February or April, and all six stations share one campus: their errors correlate at
0.93, so they are worth little more than one independent site. With 270 days the
smallest difference between LightGBM and Chronos-S-FT that the DM test could
detect with 80% power at the 5% level (two-sided) is about 16.5% in mean daily squared
error, against the 5.7% observed;
their non-significant difference is not evidence that they are equal.

Two purpose-trained baselines were not tuned: LightGBM runs with its default settings
and the LSTM was configured once, so neither shows the best its family can reach. The
fine-tuning is light, 60 to 150 optimiser steps at a single context, against 1,000 in the
official example. Three fine-tuning seeds agree closely, but the LSTM was run with one.

The dataset was public before every checkpoint used here was released, and it was later used in a public benchmark; no model card names it, but an overlap would
favour the zero-shot models. Finally, the six stations were selected on the whole record,
test period included, for complete and stable data, so the results may be optimistic for
less well-kept installations.
