: Compute cost per test window of 30 days: median wall-clock seconds over the station-folds to fit the model, to issue the window's day-ahead forecasts, and both (time per window: the median of fit + predict, not the sum of the two medians), and the maximum over the station-folds of the process's peak resident set size (one process per run, so it includes the data and the framework; 1 GB = 1,024³ bytes). Memory does not count GPU (Metal) memory, so it is a lower bound for the neural models. † No covariates: the forecasts are identical in both weather settings, so the model is reported once, under No weather. {#tbl:compute}

| Setting    | Model               | Fit (s) | Predict (s) | Time per window (s) | Max memory (GB) |
|:--------------------|:------------------------------------|-------------:|------------------:|-------------------:|--------------------:|
| NWP        | SARIMA              |   102.4 |         0.7 |               103.0 |             7.9 |
|            | LightGBM            |   164.9 |         2.1 |               167.0 |             1.1 |
|            | LSTM                |   140.7 |         0.6 |               141.2 |             0.7 |
|            | Chronos-B           |     0.0 |        44.9 |                44.9 |             1.4 |
|            | Chronos-S           |     0.0 |         2.4 |                 2.4 |             0.9 |
|            | TimesFM-3           |     0.0 |       129.8 |               129.8 |             3.6 |
|            | TiRex-2             |     0.0 |        12.9 |                12.9 |             1.1 |
|            | Chronos-S-FT        |   151.0 |         2.6 |               153.6 |             1.3 |
| No weather | Persistence †       |     0.0 |         0.0 |                 0.0 |             0.3 |
|            | Seasonal naive †    |     0.0 |         0.0 |                 0.0 |             0.3 |
|            | Smart persistence † |     0.0 |         0.0 |                 0.0 |             0.3 |
|            | Climatology †       |     0.0 |         0.0 |                 0.0 |             0.3 |
|            | CLIPER †            |     0.0 |         0.0 |                 0.0 |             0.3 |
|            | SARIMA              |    96.0 |         0.7 |                96.6 |             7.8 |
|            | LightGBM            |   124.8 |         1.9 |               126.7 |             0.8 |
|            | LSTM                |   136.4 |         0.6 |               137.0 |             0.7 |
|            | Chronos-B           |     0.0 |        31.0 |                31.0 |             1.4 |
|            | Chronos-S           |     0.0 |         2.3 |                 2.3 |             0.8 |
|            | TimesFM-3           |     0.0 |        96.8 |                96.8 |             3.3 |
|            | TiRex-2             |     0.0 |        11.7 |                11.7 |             1.1 |
|            | Chronos-S-FT        |    92.5 |         2.4 |                95.0 |             2.0 |
