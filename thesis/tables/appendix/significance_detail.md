: Test statistics of the comparisons of @tbl:significance fixed before the main run, on daily losses over the same 270 days: each local day's loss is the station average of the mean ((forecast − truth) / C)² over the day's scored daytime hours; Loss A and B are their means over the days, and Rel. diff. is the mean of d = loss A − loss B relative to loss B (negative: A better). DM: Diebold-Mariano statistic with Newey-West variance and the Harvey-Leybourne-Newbold correction, two-sided p from Student's t; Holm p adjusts over the 11 comparisons that could be run; Bootstrap p: moving-block bootstrap (7-day blocks, 10,000 resamples). Newey-West lags and bootstrap blocks run over the days in date order, so they also span the gaps between test windows. Verdict at Holm p < 0.05; n.s.: no significant difference. † No covariates: the forecasts are identical in both weather settings, so the model is reported once, under No weather. {#tbl:significance-detail}

| ID  | Comparison (A vs B)                         | Loss A | Loss B | Rel. diff. (%) |    DM |      p | Holm p | Bootstrap p | Verdict  |
|:---------|:------------------------------------|----------------:|----------------:|--------------:|--------------:|-----------------:|-----------------:|-------------------------:|:-------------------|
| P1  | NWP: LightGBM vs Chronos-S                  | 0.0122 | 0.0185 |          −34.2 | −5.37 | <0.001 | <0.001 |      <0.001 | A better |
| P2  | NWP: LightGBM vs Chronos-S-FT               | 0.0122 | 0.0129 |           −5.7 | −0.96 |  0.339 |  1.000 |       0.338 | n.s.     |
| P3  | NWP: Chronos-S-FT vs SARIMA                 | 0.0129 | 0.0151 |          −14.5 | −2.74 |  0.007 |  0.040 |       0.007 | A better |
| P4  | NWP: Chronos-S-FT vs Chronos-S              | 0.0129 | 0.0185 |          −30.2 | −4.58 | <0.001 | <0.001 |      <0.001 | A better |
| P5  | No weather: Chronos-S-FT vs Chronos-S       | 0.0234 | 0.0239 |           −2.2 | −0.74 |  0.462 |  1.000 |       0.467 | n.s.     |
| P6  | No weather: SARIMA vs Chronos-S             | 0.0233 | 0.0239 |           −2.6 | −0.74 |  0.463 |  1.000 |       0.467 | n.s.     |
| P7  | No weather: Chronos-S vs TiRex †            | 0.0239 | 0.0236 |            1.4 |  0.50 |  0.618 |  1.000 |       0.612 | n.s.     |
| P8  | No weather: TiRex † vs TimesFM †            | 0.0236 | 0.0262 |           −9.7 | −2.72 |  0.007 |  0.040 |       0.008 | A better |
| P9  | LightGBM: NWP vs No weather                 | 0.0122 | 0.0236 |          −48.3 | −6.32 | <0.001 | <0.001 |      <0.001 | A better |
| P10 | Chronos-S: NWP vs No weather                | 0.0185 | 0.0239 |          −22.7 | −5.86 | <0.001 | <0.001 |      <0.001 | A better |
| P11 | No weather: CLIPER † vs Smart persistence † | 0.0237 | 0.0303 |          −21.7 | −4.43 | <0.001 | <0.001 |      <0.001 | A better |
