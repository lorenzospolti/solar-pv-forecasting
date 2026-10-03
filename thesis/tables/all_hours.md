: Robustness of the headline to the scoring mask: skill over smart persistence and nRMSE scored on all hours with valid truth (night included); skill as the mean over stations [min, max], nRMSE as the mean. No model's mean skill moves by more than 0.00005 from its daytime value (@tbl:master-skill): at night both truth and forecasts are close to zero for every model, so the night adds almost nothing to either RMSE in the skill ratio. The night zeros do lower nRMSE, by 32 % on average, which is why accuracy is reported on daytime hours. † No covariates: the forecasts are identical in both weather settings, so the model is reported once, under No weather. {#tbl:all-hours}

| Model                    | Skill vs SP, all hours (NWP) | Skill vs SP, all hours (No weather) | nRMSE, all hours (NWP) | nRMSE, all hours (No weather) |
|:----------------------------------|----------------------------------:|-------------------------------------------:|----------------------:|----------------------:|
| Persistence †            |                            — |             −0.001 [−0.001, −0.001] |                      — |                         0.119 |
| Seasonal naive (168 h) † |                            — |             −0.380 [−0.420, −0.362] |                      — |                         0.165 |
| Smart persistence †      |                            — |                0.000 [0.000, 0.000] |                      — |                         0.119 |
| Climatology †            |                            — |             −0.049 [−0.074, −0.037] |                      — |                         0.125 |
| CLIPER †                 |                            — |                0.116 [0.110, 0.120] |                      — |                         0.105 |
| SARIMA                   |         0.292 [0.286, 0.299] |                0.121 [0.114, 0.125] |                  0.084 |                         0.105 |
| LightGBM                 |         0.364 [0.343, 0.377] |                0.123 [0.110, 0.134] |                  0.076 |                         0.105 |
| LSTM                     |         0.302 [0.276, 0.325] |                                   — |                  0.083 |                             — |
| Chronos-2 (zero-shot)    |         0.220 [0.215, 0.227] |                0.112 [0.101, 0.122] |                  0.093 |                         0.106 |
| TimesFM-2.5 †            |                            — |                0.073 [0.053, 0.089] |                      — |                         0.111 |
| TiRex †                  |                            — |                0.116 [0.109, 0.125] |                      — |                         0.105 |
