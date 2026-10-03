: Probabilistic accuracy on daytime hours, mean over stations of each station's pooled value: CRPS (2 × mean pinball loss over the nine quantiles) / C, CRPS skill over SP (1 − CRPS / MAE of SP), coverage of the 10–90 % interval (PICP-80, nominal 0.8) and its mean width / C, and the observed share of hours at or below the 0.1, 0.5 and 0.9 quantiles (Cov.). Deterministic models enter with CRPS = MAE. † No covariates: the forecasts are identical in both weather settings, so the model is reported once, under No weather. {#tbl:probabilistic}

| Setting    | Model                    | CRPS / C | CRPS skill vs SP | PICP-80 | Width-80 / C | Cov. q0.1 | Cov. q0.5 | Cov. q0.9 | Note                       |
|:--------------------|:----------------------------------|---------------:|-----------------:|----------------:|------------------:|-------------:|-------------:|-------------:|:---------------------------------|
| NWP        | SARIMA                   |    0.068 |            0.392 |   0.751 |        0.268 |     0.127 |     0.470 |     0.877 |                            |
|            | LightGBM                 |    0.059 |            0.477 |   0.707 |        0.213 |     0.174 |     0.537 |     0.881 |                            |
|            | LSTM                     |    0.085 |            0.243 |       — |            — |         — |         — |         — | deterministic (CRPS = MAE) |
|            | Chronos-2 (zero-shot)    |    0.072 |            0.362 |   0.849 |        0.327 |     0.081 |     0.506 |     0.928 |                            |
| No weather | Persistence †            |    0.113 |           −0.001 |       — |            — |         — |         — |         — | deterministic (CRPS = MAE) |
|            | Seasonal naive (168 h) † |    0.169 |           −0.500 |       — |            — |         — |         — |         — | deterministic (CRPS = MAE) |
|            | Smart persistence †      |    0.113 |            0.000 |       — |            — |         — |         — |         — | deterministic (CRPS = MAE) |
|            | Climatology †            |    0.144 |           −0.276 |       — |            — |         — |         — |         — | deterministic (CRPS = MAE) |
|            | CLIPER †                 |    0.115 |           −0.019 |       — |            — |         — |         — |         — | deterministic (CRPS = MAE) |
|            | SARIMA                   |    0.085 |            0.248 |   0.719 |        0.296 |     0.145 |     0.448 |     0.863 |                            |
|            | LightGBM                 |    0.083 |            0.260 |   0.706 |        0.309 |     0.165 |     0.484 |     0.870 |                            |
|            | Chronos-2 (zero-shot)    |    0.083 |            0.267 |   0.813 |        0.334 |     0.111 |     0.499 |     0.923 |                            |
|            | TimesFM-2.5 †            |    0.088 |            0.218 |   0.754 |        0.327 |     0.148 |     0.472 |     0.901 |                            |
|            | TiRex †                  |    0.082 |            0.276 |   0.805 |        0.314 |     0.114 |     0.483 |     0.917 |                            |
