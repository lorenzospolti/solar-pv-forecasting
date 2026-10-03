: Compute cost per test window: median wall-clock seconds to fit, to issue the window's 30 day-ahead forecasts, and both, and median peak resident memory of the run's process (one process per run, so it includes the data and the framework), over the station-folds of each model and setting. † No covariates: the forecasts are identical in both weather settings, so the model is reported once, under No weather. {#tbl:compute}

| Setting    | Model                    | Fit (s) | Predict (s) | Fit + predict (s) | Peak RSS (MB) | Station-folds |
|:--------------------|:----------------------------------|-------------:|------------------:|-----------------:|-------------:|--------------------:|
| NWP        | SARIMA                   |   102.5 |         0.7 |             103.2 |          4148 |            54 |
|            | LightGBM                 |   170.8 |         2.2 |             172.9 |           865 |            54 |
|            | LSTM                     |   234.7 |         0.6 |             235.3 |           669 |            54 |
|            | Chronos-2 (zero-shot)    |     0.0 |         2.5 |               2.5 |           811 |            54 |
| No weather | Persistence †            |     0.0 |         0.0 |               0.0 |           288 |            54 |
|            | Seasonal naive (168 h) † |     0.0 |         0.0 |               0.0 |           287 |            54 |
|            | Smart persistence †      |     0.0 |         0.0 |               0.0 |           288 |            54 |
|            | Climatology †            |     0.0 |         0.0 |               0.0 |           288 |            54 |
|            | CLIPER †                 |     0.0 |         0.0 |               0.0 |           289 |            54 |
|            | SARIMA                   |    95.1 |         0.7 |              95.8 |          4135 |            54 |
|            | LightGBM                 |   128.6 |         1.9 |             130.6 |           650 |            54 |
|            | Chronos-2 (zero-shot)    |     0.0 |         2.4 |               2.4 |           757 |            54 |
|            | TimesFM-2.5 †            |     0.0 |         6.5 |               6.5 |          2266 |            54 |
|            | TiRex †                  |     0.0 |        11.0 |              11.0 |           950 |            54 |
