: Test statistics of the tested family of @tbl:significance, on daily losses over the same 270 days: each local day's loss is the station average of the mean ((forecast − truth) / C)² over the day's scored daytime hours; Loss A and B are their means over the days, and Rel. diff. is the mean of d = loss A − loss B relative to loss B (negative: A better). DM: Diebold-Mariano statistic with Newey-West variance and the Harvey-Leybourne-Newbold correction, two-sided p from Student's t; Holm p adjusts over the 11 comparisons that could be run; Bootstrap p: moving-block bootstrap (7-day blocks, 10,000 resamples). Newey-West lags and bootstrap blocks run over the days in date order, so they also span the gaps between test windows. Verdict at Holm p < 0.05; n.s.: no significant difference. † No covariates: the forecasts are identical in both weather settings, so the model is reported once, under No weather. {#tbl:significance-detail}

| ID  | Comparison (A vs B)                         | Loss A | Loss B | Rel. diff. (%) |    DM |      p | Holm p | Bootstrap p | Verdict  |
|:---------|:------------------------------------|----------------:|----------------:|--------------:|--------------:|-----------------:|-----------------:|-------------------------:|:-------------------|
| P1  | NWP: LightGBM vs Chronos-B                  | 0.0122 | 0.0126 |           −3.6 | −0.70 |  0.484 |  1.000 |       0.476 | n.s.     |
| P2  | NWP: LightGBM vs Chronos-S-FT               | 0.0122 | 0.0129 |           −5.8 | −0.98 |  0.330 |  1.000 |       0.330 | n.s.     |
| P3  | NWP: Chronos-S-FT vs SARIMA                 | 0.0129 | 0.0151 |          −14.4 | −2.72 |  0.007 |  0.042 |       0.007 | A better |
| P4  | NWP: Chronos-S-FT vs Chronos-S              | 0.0129 | 0.0185 |          −30.2 | −4.58 | <0.001 | <0.001 |      <0.001 | A better |
| P5  | No weather: Chronos-S-FT vs Chronos-S       | 0.0234 | 0.0239 |           −2.1 | −0.70 |  0.485 |  1.000 |       0.491 | n.s.     |
| P6  | No weather: SARIMA vs Chronos-B             | 0.0233 | 0.0236 |           −1.2 | −0.26 |  0.798 |  1.000 |       0.799 | n.s.     |
| P7  | NWP: LightGBM vs TimesFM-3                  | 0.0122 | 0.0111 |            9.6 |  1.89 |  0.060 |  0.302 |       0.056 | n.s.     |
| P8  | NWP: LightGBM vs TiRex-2                    | 0.0122 | 0.0157 |          −22.4 | −4.50 | <0.001 | <0.001 |      <0.001 | A better |
| P9  | LightGBM: NWP vs No weather                 | 0.0122 | 0.0236 |          −48.3 | −6.32 | <0.001 | <0.001 |      <0.001 | A better |
| P10 | Chronos-B: NWP vs No weather                | 0.0126 | 0.0236 |          −46.5 | −6.32 | <0.001 | <0.001 |      <0.001 | A better |
| P11 | No weather: CLIPER † vs Smart persistence † | 0.0237 | 0.0303 |          −21.7 | −4.43 | <0.001 | <0.001 |      <0.001 | A better |
