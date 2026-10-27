: Probabilistic accuracy on daytime hours of the models that issue quantiles, mean over the stations of each station's value pooled over all test days. QS / C: the quantile score, the mean pinball loss over the nine quantile levels 0.1, …, 0.9, divided by capacity C (lower is better); twice the QS is a common approximation of the CRPS. PICP-80: share of hours inside the 10–90 % interval (nominal 0.8), with its 95 % interval from a 7-day moving-block bootstrap over the test days (10,000 resamples, the same days for all stations). Width-80 / C: the interval's mean width divided by C. † No covariates: the forecasts are identical in both weather settings, so the model is reported once, under No weather. {#tbl:probabilistic}

| Setting    | Model        | QS / C |              PICP-80 | Width-80 / C |
|:--------------------|:------------------------------------|----------------:|----------------------------------:|------------------:|
| NWP        | SARIMA       | 0.0342 | 0.751 [0.717, 0.781] |        0.268 |
|            | LightGBM     | 0.0294 | 0.707 [0.673, 0.739] |        0.213 |
|            | Chronos-B    | 0.0295 | 0.862 [0.837, 0.885] |        0.277 |
|            | Chronos-S    | 0.0360 | 0.849 [0.823, 0.873] |        0.327 |
|            | TimesFM-3    | 0.0275 | 0.854 [0.834, 0.870] |        0.260 |
|            | TiRex-2      | 0.0325 | 0.778 [0.748, 0.803] |        0.236 |
|            | Chronos-S-FT | 0.0294 | 0.771 [0.737, 0.802] |        0.225 |
| No weather | SARIMA       | 0.0424 | 0.719 [0.677, 0.754] |        0.296 |
|            | LightGBM     | 0.0417 | 0.706 [0.665, 0.745] |        0.309 |
|            | Chronos-B    | 0.0408 | 0.855 [0.822, 0.886] |        0.368 |
|            | Chronos-S    | 0.0413 | 0.813 [0.779, 0.843] |        0.334 |
|            | TimesFM-3    | 0.0404 | 0.841 [0.810, 0.869] |        0.374 |
|            | TiRex-2      | 0.0410 | 0.780 [0.744, 0.810] |        0.312 |
|            | Chronos-S-FT | 0.0405 | 0.781 [0.747, 0.812] |        0.323 |
