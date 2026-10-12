: Robustness of comparison P2 (@tbl:significance): skill over CLIPER of LightGBM minus that of Chronos-S-FT, NWP setting (positive: LightGBM more accurate), with its 95 % interval from a 7-day moving-block bootstrap over the test days (10,000 resamples, the same days for all stations): on all test windows with each fine-tuning seed, and with seed 0 leaving out one test window (named by the month holding its middle) at a time, the interval then resampling the remaining days. ‡ Post hoc: added after the results of the main run had been seen. {#tbl:p2-robustness}

| Test days             | LightGBM − Chronos-S-FT |
|:------------------------|--------------------------------------:|
| All windows (seed 0)  |   0.023 [−0.021, 0.065] |
| All windows, seed 1 ‡ |   0.024 [−0.020, 0.065] |
| All windows, seed 2 ‡ |   0.028 [−0.017, 0.070] |
| Without Jul 2022      |   0.024 [−0.024, 0.069] |
| Without Sep 2022      |   0.016 [−0.034, 0.062] |
| Without Nov 2022      |   0.028 [−0.021, 0.072] |
| Without Jan 2023      |   0.024 [−0.025, 0.069] |
| Without Mar 2023      |   0.039 [−0.004, 0.080] |
| Without May 2023      |  −0.005 [−0.050, 0.035] |
| Without Jul 2023      |   0.032 [−0.018, 0.079] |
| Without Sep 2023      |   0.014 [−0.038, 0.063] |
| Without Nov 2023      |   0.032 [−0.016, 0.075] |
