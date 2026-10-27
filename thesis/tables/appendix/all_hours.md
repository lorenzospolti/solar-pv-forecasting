: Robustness of the headline to the scoring mask: skill over Smart persistence and nRMSE scored on all hours with valid truth (night included); skill as the mean over stations [min, max], nRMSE as the mean. No model's mean skill moves by more than 0.00005 from its daytime value (@tbl:master-skill-full): at night both truth and forecasts are close to zero for every model, so the night adds almost nothing to either RMSE in the skill ratio. The night zeros do lower nRMSE, by 32 % on average, which is why accuracy is reported on daytime hours. † No covariates: the forecasts are identical in both weather settings, so the model is reported once, under No weather. {#tbl:all-hours}

| Model               | Skill vs SP, all hours (NWP) | Skill vs SP, all hours (No weather) | nRMSE, all hours (NWP) | nRMSE, all hours (No weather) |
|:------------------------------------|----------------------------------:|-------------------------------------------:|----------------------:|----------------------:|
| Persistence †       |                            — |             −0.001 [−0.001, −0.001] |                      — |                         0.119 |
| Seasonal naive †    |                            — |             −0.380 [−0.420, −0.362] |                      — |                         0.165 |
| Smart persistence † |                            — |                0.000 [0.000, 0.000] |                      — |                         0.119 |
| Climatology †       |                            — |             −0.049 [−0.074, −0.037] |                      — |                         0.125 |
| CLIPER †            |                            — |                0.116 [0.110, 0.120] |                      — |                         0.105 |
| SARIMA              |         0.292 [0.286, 0.299] |                0.121 [0.114, 0.125] |                  0.084 |                         0.105 |
| LightGBM            |         0.364 [0.343, 0.377] |                0.123 [0.110, 0.134] |                  0.076 |                         0.105 |
| LSTM                |         0.302 [0.276, 0.325] |               0.033 [−0.002, 0.067] |                  0.083 |                         0.115 |
| Chronos-B           |         0.354 [0.339, 0.363] |                0.119 [0.112, 0.126] |                  0.077 |                         0.105 |
| Chronos-S           |         0.220 [0.215, 0.227] |                0.112 [0.101, 0.122] |                  0.093 |                         0.106 |
| TimesFM-3           |         0.393 [0.389, 0.401] |                0.136 [0.127, 0.148] |                  0.072 |                         0.103 |
| TiRex-2             |         0.280 [0.263, 0.296] |                0.120 [0.116, 0.126] |                  0.086 |                         0.105 |
| Chronos-S-FT        |         0.343 [0.333, 0.363] |                0.121 [0.113, 0.130] |                  0.078 |                         0.105 |
