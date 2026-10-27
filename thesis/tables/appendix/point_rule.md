: Sensitivity to the point-forecast rule: skill over CLIPER on daytime hours (mean over stations) when a probabilistic model is scored on the mean of its nine quantiles (the rule used throughout) and on its median (the 0.5 quantile), and the difference (mean rule minus median rule). † No covariates: the forecasts are identical in both weather settings, so the model is reported once, under No weather. {#tbl:point-rule}

| Setting    | Model        | Skill, mean rule | Skill, median rule | Difference |
|:--------------------|:------------------------------------|-----------------:|------------------:|-------------------------:|
| NWP        | SARIMA       |            0.200 |              0.200 |      0.000 |
|            | LightGBM     |            0.280 |              0.272 |      0.008 |
|            | Chronos-B    |            0.269 |              0.264 |      0.005 |
|            | Chronos-S    |            0.118 |              0.086 |      0.032 |
|            | TimesFM-3    |            0.314 |              0.304 |      0.009 |
|            | TiRex-2      |            0.186 |              0.176 |      0.010 |
|            | Chronos-S-FT |            0.257 |              0.242 |      0.016 |
| No weather | SARIMA       |            0.005 |              0.005 |      0.000 |
|            | LightGBM     |            0.008 |             −0.031 |      0.039 |
|            | Chronos-B    |            0.003 |             −0.038 |      0.042 |
|            | Chronos-S    |           −0.004 |             −0.030 |      0.026 |
|            | TimesFM-3    |            0.022 |              0.000 |      0.022 |
|            | TiRex-2      |            0.005 |             −0.002 |      0.007 |
|            | Chronos-S-FT |            0.006 |             −0.017 |      0.023 |
