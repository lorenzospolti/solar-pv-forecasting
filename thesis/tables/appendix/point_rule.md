: Sensitivity to the point-forecast rule: skill over CLIPER on daytime hours (mean over stations) when a probabilistic model is scored on the mean of its nine quantiles (the rule used throughout) and on its median (the 0.5 quantile), and the difference (mean rule minus median rule). † No covariates: the forecasts are identical in both weather settings, so the model is reported once, under No weather. ‡ Post hoc: added after the results of the main run had been seen. {#tbl:point-rule}

| Setting    | Model            | Skill, mean rule | Skill, median rule | Difference |
|:--------------------|:-----------------------------------------|-----------------:|------------------:|-------------------------:|
| NWP        | SARIMA           |            0.200 |              0.200 |      0.000 |
|            | LightGBM         |            0.280 |              0.272 |      0.008 |
|            | Chronos-S        |            0.118 |              0.086 |      0.032 |
|            | Chronos-B-long ‡ |            0.274 |              0.269 |      0.005 |
|            | Chronos-S-FT     |            0.257 |              0.242 |      0.016 |
|            | TimesFM-XReg ‡   |            0.242 |              0.242 |      0.000 |
|            | TiRex-XReg ‡     |            0.247 |              0.243 |      0.003 |
| No weather | SARIMA           |            0.005 |              0.005 |      0.000 |
|            | LightGBM         |            0.008 |             −0.031 |      0.039 |
|            | Chronos-S        |           −0.004 |             −0.030 |      0.026 |
|            | Chronos-B-long ‡ |            0.008 |             −0.028 |      0.036 |
|            | Chronos-S-FT     |            0.007 |             −0.017 |      0.024 |
|            | TimesFM †        |           −0.049 |             −0.078 |      0.029 |
|            | TiRex †          |            0.001 |             −0.027 |      0.027 |
