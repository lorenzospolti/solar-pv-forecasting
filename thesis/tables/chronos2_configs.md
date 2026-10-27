: The Chronos-2 configurations. Names: S and B for the small and base checkpoints, -FT for fine-tuning on each training window; the others are zero-shot. Context: the maximum history the model reads. Skill over CLIPER on daytime hours as in @tbl:master-skill, with its 95 % interval from a 7-day moving-block bootstrap over the test days (10,000 resamples, the same days for all stations). PICP-80: share of hours inside the 10–90 % interval (nominal 0.8). Time: median over the station-folds of fit + predict seconds per 30-day test window. {#tbl:chronos2-configs}

| Configuration | Checkpoint | Context (h) |          Skill (NWP) |     Skill (No weather) | PICP-80 (NWP) | Time per window (s) |
|:------------------------------------|:-----------------------------|--------------------:|----------------------------------:|--------------------------------------:|------------------:|-------------------:|
| Chronos-B     | base       |       8,192 | 0.269 [0.224, 0.312] |  0.003 [−0.042, 0.058] |         0.862 |                44.9 |
| Chronos-S     | small      |         336 | 0.118 [0.073, 0.164] | −0.004 [−0.045, 0.041] |         0.849 |                 2.4 |
| Chronos-S-FT  | small      |         336 | 0.257 [0.196, 0.314] |  0.006 [−0.030, 0.051] |         0.771 |               153.6 |
