: Accuracy on daytime hours: skill over CLIPER (1 − RMSE / RMSE of CLIPER), each station pooled over all test days, mean over the 6 stations, with its 95 % interval from a 7-day moving-block bootstrap over the test days (10,000 resamples, the same days for all stations); nRMSE = RMSE / capacity, mean over stations. Probabilistic models are scored on the mean of their nine quantiles. The full table, with skill over Smart persistence and the range across stations, is @tbl:master-skill-full. † No covariates: the forecasts are identical in both weather settings, so the model is reported once, under No weather. ‡ Post hoc: added after the results of the main run had been seen. {#tbl:master-skill}

| Model               |          Skill (NWP) |      Skill (No weather) | nRMSE (NWP) | nRMSE (No weather) |
|:-----------------------------------------|----------------------------------:|-------------------------------------------:|--------------------:|----------------------:|
| Smart persistence † |                    — | −0.131 [−0.173, −0.082] |           — |              0.175 |
| CLIPER †            |                    — |                       0 |           — |              0.155 |
| SARIMA              | 0.200 [0.160, 0.235] |   0.005 [−0.025, 0.036] |       0.124 |              0.154 |
| LightGBM            | 0.280 [0.231, 0.326] |   0.008 [−0.050, 0.061] |       0.111 |              0.153 |
| LSTM                | 0.211 [0.152, 0.267] | −0.094 [−0.187, −0.018] |       0.122 |              0.169 |
| Chronos-S           | 0.118 [0.073, 0.164] |  −0.004 [−0.045, 0.041] |       0.136 |              0.155 |
| Chronos-B-long ‡    | 0.274 [0.227, 0.316] |   0.008 [−0.037, 0.059] |       0.112 |              0.153 |
| Chronos-S-FT        | 0.257 [0.196, 0.315] |   0.007 [−0.030, 0.051] |       0.115 |              0.153 |
| TimesFM †           |                    — |  −0.049 [−0.099, 0.002] |           — |              0.162 |
| TimesFM-XReg ‡      | 0.242 [0.189, 0.290] |                       — |       0.117 |                  — |
| TiRex †             |                    — |   0.001 [−0.046, 0.055] |           — |              0.154 |
| TiRex-XReg ‡        | 0.247 [0.198, 0.293] |                       — |       0.116 |                  — |
