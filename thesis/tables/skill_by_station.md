: Skill over smart persistence per station (daytime hours, all test windows pooled) and its mean over stations. † No covariates: the forecasts are identical in both weather settings, so the model is reported once, under No weather. {#tbl:skill-station}

| Setting    | Model                    | CYT (Station 2) | LSK North |    SQ1 |   SQ15 |  SQ567 | UG Hall6 |   Mean |
|:--------------------|:----------------------------------|--------------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|
| NWP        | SARIMA                   |           0.293 |     0.288 |  0.290 |  0.286 |  0.299 |    0.298 |  0.292 |
|            | LightGBM                 |           0.377 |     0.365 |  0.352 |  0.343 |  0.372 |    0.372 |  0.364 |
|            | LSTM                     |           0.325 |     0.282 |  0.311 |  0.276 |  0.298 |    0.321 |  0.302 |
|            | Chronos-2 (zero-shot)    |           0.223 |     0.227 |  0.216 |  0.217 |  0.225 |    0.215 |  0.220 |
| No weather | Persistence †            |          −0.001 |    −0.001 | −0.001 | −0.001 | −0.001 |   −0.001 | −0.001 |
|            | Seasonal naive (168 h) † |          −0.368 |    −0.380 | −0.420 | −0.362 | −0.379 |   −0.373 | −0.380 |
|            | Climatology †            |          −0.051 |    −0.042 | −0.074 | −0.037 | −0.045 |   −0.044 | −0.049 |
|            | CLIPER †                 |           0.114 |     0.118 |  0.110 |  0.120 |  0.117 |    0.116 |  0.116 |
|            | SARIMA                   |           0.118 |     0.124 |  0.114 |  0.118 |  0.125 |    0.125 |  0.121 |
|            | LightGBM                 |           0.134 |     0.126 |  0.110 |  0.132 |  0.121 |    0.116 |  0.123 |
|            | Chronos-2 (zero-shot)    |           0.122 |     0.121 |  0.101 |  0.107 |  0.114 |    0.108 |  0.112 |
|            | TimesFM-2.5 †            |           0.075 |     0.089 |  0.053 |  0.074 |  0.076 |    0.069 |  0.073 |
|            | TiRex †                  |           0.116 |     0.115 |  0.109 |  0.109 |  0.125 |    0.125 |  0.116 |
