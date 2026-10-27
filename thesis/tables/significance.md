: Pairwise comparisons. Skill difference: skill over CLIPER of A minus that of B (positive: A more accurate), with its 95 % interval from a 7-day moving-block bootstrap over the test days (10,000 resamples, the same days for all stations). P1–P11 form the tested family: Diebold-Mariano test on daily squared-error losses, Holm-adjusted over the 11 comparisons; verdict at Holm p < 0.05 (n.s.: no significant difference). W1–W4 are outside the tested family and are reported with intervals only (verdict: no test). Full test statistics: @tbl:significance-detail. † No covariates: the forecasts are identical in both weather settings, so the model is reported once, under No weather. {#tbl:significance}

| ID  | Comparison (A vs B)                         | Skill difference (A − B) | Holm p | Verdict  |
|:---------|:------------------------------------|-------------------------------------------:|-----------------:|:-------------------|
| P1  | NWP: LightGBM vs Chronos-B                  |    0.011 [−0.026, 0.047] |  1.000 | n.s.     |
| P2  | NWP: LightGBM vs Chronos-S-FT               |    0.023 [−0.021, 0.065] |  1.000 | n.s.     |
| P3  | NWP: Chronos-S-FT vs SARIMA                 |     0.058 [0.015, 0.099] |  0.042 | A better |
| P4  | NWP: Chronos-S-FT vs Chronos-S              |     0.139 [0.082, 0.191] | <0.001 | A better |
| P5  | No weather: Chronos-S-FT vs Chronos-S       |    0.011 [−0.017, 0.042] |  1.000 | n.s.     |
| P6  | No weather: SARIMA vs Chronos-B             |    0.002 [−0.045, 0.043] |  1.000 | n.s.     |
| P7  | NWP: LightGBM vs TimesFM-3                  |  −0.033 [−0.066, −0.001] |  0.302 | n.s.     |
| P8  | NWP: LightGBM vs TiRex-2                    |     0.095 [0.059, 0.135] | <0.001 | A better |
| P9  | LightGBM: NWP vs No weather                 |     0.272 [0.204, 0.337] | <0.001 | A better |
| P10 | Chronos-B: NWP vs No weather                |     0.266 [0.205, 0.321] | <0.001 | A better |
| P11 | No weather: CLIPER † vs Smart persistence † |     0.131 [0.082, 0.173] | <0.001 | A better |
| W1  | TimesFM-3: NWP vs No weather                |     0.291 [0.228, 0.351] |      — | no test  |
| W2  | TiRex-2: NWP vs No weather                  |     0.180 [0.108, 0.246] |      — | no test  |
| W3  | NWP: TimesFM-3 vs Chronos-B                 |     0.045 [0.017, 0.071] |      — | no test  |
| W4  | NWP: TiRex-2 vs Chronos-B                   |  −0.083 [−0.136, −0.037] |      — | no test  |
