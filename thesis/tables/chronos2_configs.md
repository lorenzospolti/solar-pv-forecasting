: The Chronos-2 configurations. Names: S and B for the small and base checkpoints, -mid and -long for contexts longer than the one fixed before the main run, -FT for fine-tuning on each training window (repeated with seeds 1 and 2 in the NWP setting only). Skill over CLIPER on daytime hours as in @tbl:master-skill, with its 95 % interval from a 7-day moving-block bootstrap over the test days (10,000 resamples, the same days for all stations). PICP-80: share of hours inside the 10–90 % interval (nominal 0.8). Time: median over the station-folds of fit + predict seconds per 30-day test window. ‡ Post hoc: added after the results of the main run had been seen. {#tbl:chronos2-configs}

| Configuration          | Checkpoint | Context (h) |          Skill (NWP) |     Skill (No weather) | PICP-80 (NWP) | Time per window (s) |
|:----------------------------------------|:-----------------------------|--------------------:|----------------------------------:|--------------------------------------:|------------------:|-------------------:|
| Chronos-S              | small      |         336 | 0.118 [0.073, 0.164] | −0.004 [−0.045, 0.041] |         0.849 |                 2.5 |
| Chronos-S-mid ‡        | small      |       1,024 | 0.187 [0.144, 0.231] |  0.007 [−0.035, 0.057] |         0.871 |                 3.1 |
| Chronos-S-long ‡       | small      |       2,048 | 0.211 [0.168, 0.253] |  0.008 [−0.035, 0.058] |         0.886 |                 4.1 |
| Chronos-B ‡            | base       |         336 | 0.163 [0.120, 0.207] | −0.001 [−0.050, 0.050] |         0.832 |                 5.5 |
| Chronos-B-long ‡       | base       |       2,048 | 0.274 [0.227, 0.316] |  0.008 [−0.037, 0.059] |         0.843 |                11.9 |
| Chronos-S-FT           | small      |         336 | 0.257 [0.196, 0.315] |  0.007 [−0.030, 0.051] |         0.770 |               186.5 |
| Chronos-S-FT, seed 1 ‡ | small      |         336 | 0.256 [0.194, 0.314] |                      — |         0.760 |               149.7 |
| Chronos-S-FT, seed 2 ‡ | small      |         336 | 0.252 [0.191, 0.311] |                      — |         0.762 |               147.2 |
