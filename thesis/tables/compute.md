: Compute cost per test window of 30 days: median wall-clock seconds over the station-folds to fit the model and to issue the window's day-ahead forecasts, and the maximum over the station-folds of the process's peak resident set size (one process per run, so it includes the data and the framework; 1 GB = 1,024³ bytes). Memory does not count GPU (Metal) memory, so it is a lower bound for the neural models. † No covariates: the forecasts are identical in both weather settings, so the model is reported once, under No weather. ‡ Post hoc: added after the results of the main run had been seen. {#tbl:compute}

| Setting    | Model            | Fit (s) | Predict (s) | Max memory (GB) |
|:--------------------|:----------------------------------------|-------------:|------------------:|--------------------:|
| NWP        | SARIMA           |   102.5 |         0.7 |             7.9 |
|            | LightGBM         |   170.8 |         2.2 |             1.2 |
|            | LSTM             |   234.7 |         0.6 |             0.7 |
|            | Chronos-S        |     0.0 |         2.5 |             0.9 |
|            | Chronos-B-long ‡ |     0.0 |        11.8 |             1.4 |
|            | Chronos-S-FT     |   183.8 |         2.6 |             1.1 |
| No weather | SARIMA           |    95.1 |         0.7 |             7.9 |
|            | LightGBM         |   128.6 |         1.9 |             0.9 |
|            | LSTM             |   138.2 |         0.6 |             0.6 |
|            | Chronos-S        |     0.0 |         2.4 |             0.8 |
|            | Chronos-B-long ‡ |     0.0 |         9.6 |             1.4 |
|            | Chronos-S-FT     |    87.0 |         2.3 |             2.0 |
|            | TimesFM †        |     0.0 |         6.5 |             2.2 |
|            | TiRex †          |     0.0 |        11.0 |             1.1 |
