: Pairwise comparisons. Skill difference: skill over CLIPER of A minus that of B (positive: A more accurate), with its 95 % interval from a 7-day moving-block bootstrap over the test days (10,000 resamples, the same days for all stations). P1–P11 were fixed before the main run, after a first run on the same test days had been examined: Diebold-Mariano test on daily squared-error losses, Holm-adjusted over the 11 comparisons; verdict at Holm p < 0.05 (n.s.: no significant difference). E1–E5 are post hoc (they involve configurations added after the main run) and are reported with intervals only. Full test statistics: @tbl:significance-detail. † No covariates: the forecasts are identical in both weather settings, so the model is reported once, under No weather. ‡ Post hoc: added after the results of the main run had been seen. {#tbl:significance}

| ID  | Comparison (A vs B)                         | Skill difference (A − B) | Holm p | Verdict  |
|:---------|:----------------------------------------|--------------------------------------:|-----------------:|:-------------------|
| P1  | NWP: LightGBM vs Chronos-S                  |     0.162 [0.111, 0.210] | <0.001 | A better |
| P2  | NWP: LightGBM vs Chronos-S-FT               |    0.023 [−0.021, 0.065] |  1.000 | n.s.     |
| P3  | NWP: Chronos-S-FT vs SARIMA                 |     0.058 [0.015, 0.099] |  0.040 | A better |
| P4  | NWP: Chronos-S-FT vs Chronos-S              |     0.139 [0.082, 0.191] | <0.001 | A better |
| P5  | No weather: Chronos-S-FT vs Chronos-S       |    0.011 [−0.016, 0.042] |  1.000 | n.s.     |
| P6  | No weather: SARIMA vs Chronos-S             |    0.010 [−0.027, 0.041] |  1.000 | n.s.     |
| P7  | No weather: Chronos-S vs TiRex †            |   −0.005 [−0.032, 0.021] |  1.000 | n.s.     |
| P8  | No weather: TiRex † vs TimesFM †            |     0.050 [0.012, 0.090] |  0.040 | A better |
| P9  | LightGBM: NWP vs No weather                 |     0.272 [0.204, 0.337] | <0.001 | A better |
| P10 | Chronos-S: NWP vs No weather                |     0.123 [0.089, 0.152] | <0.001 | A better |
| P11 | No weather: CLIPER † vs Smart persistence † |     0.131 [0.082, 0.173] | <0.001 | A better |
| E1  | NWP: Chronos-S-long ‡ vs Chronos-S          |     0.093 [0.066, 0.118] |      — | post hoc |
| E2  | NWP: LightGBM vs Chronos-S-long ‡           |     0.069 [0.026, 0.110] |      — | post hoc |
| E3  | NWP: Chronos-S-FT vs Chronos-S-long ‡       |     0.046 [0.003, 0.087] |      — | post hoc |
| E4  | NWP: LightGBM vs Chronos-B-long ‡           |    0.007 [−0.029, 0.041] |      — | post hoc |
| E5  | NWP: Chronos-S-FT vs Chronos-B-long ‡       |   −0.016 [−0.051, 0.018] |      — | post hoc |
