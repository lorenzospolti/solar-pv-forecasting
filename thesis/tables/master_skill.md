: Accuracy on daytime hours: skill over CLIPER (1 − RMSE / RMSE of CLIPER), each station pooled over all test days, mean over the 6 stations, with its 95 % interval from a 7-day moving-block bootstrap over the test days (10,000 resamples, the same days for all stations); nRMSE = RMSE / capacity, mean over stations. Probabilistic models are scored on the mean of their nine quantiles. The full table, with skill over Smart persistence and the range across stations, is @tbl:master-skill-full. † No covariates: the forecasts are identical in both weather settings, so the model is reported once, under No weather. {#tbl:master-skill}

| Model               |          Skill (NWP) |      Skill (No weather) | nRMSE (NWP) | nRMSE (No weather) |
|:------------------------------------|----------------------------------:|-------------------------------------------:|--------------------:|----------------------:|
| Persistence †       |                    — | −0.132 [−0.174, −0.083] |           — |              0.175 |
| Seasonal naive †    |                    — | −0.561 [−0.702, −0.414] |           — |              0.241 |
| Smart persistence † |                    — | −0.131 [−0.173, −0.082] |           — |              0.175 |
| Climatology †       |                    — | −0.186 [−0.255, −0.118] |           — |              0.183 |
| CLIPER †            |                    — |                       0 |           — |              0.155 |
| SARIMA              | 0.200 [0.160, 0.235] |   0.005 [−0.025, 0.036] |       0.124 |              0.154 |
| LightGBM            | 0.280 [0.231, 0.326] |   0.008 [−0.050, 0.061] |       0.111 |              0.153 |
| LSTM                | 0.211 [0.152, 0.267] | −0.094 [−0.187, −0.018] |       0.122 |              0.169 |
| Chronos-B           | 0.269 [0.224, 0.312] |   0.003 [−0.042, 0.058] |       0.113 |              0.154 |
| Chronos-S           | 0.118 [0.073, 0.164] |  −0.004 [−0.045, 0.041] |       0.136 |              0.155 |
| TimesFM-3           | 0.314 [0.259, 0.363] |   0.022 [−0.011, 0.058] |       0.106 |              0.151 |
| TiRex-2             | 0.186 [0.122, 0.243] |   0.005 [−0.034, 0.044] |       0.126 |              0.154 |
| Chronos-S-FT        | 0.257 [0.196, 0.314] |   0.006 [−0.030, 0.051] |       0.115 |              0.153 |
