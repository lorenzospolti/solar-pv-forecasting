# Method {#sec:method}

This chapter describes how the forecasts were produced and scored; all models run
through the same code, and the data are described in Chapter [-@sec:data].

## Design overview and research questions {#sec:method-overview}

The experiment crosses four factors: the six stations of the panel (@sec:data-panel), two
weather settings, twelve models and nine test windows, called folds. Five models are
simple reference forecasts. Three are purpose-trained on each station's history: a
seasonal autoregressive integrated moving-average model (SARIMA), the gradient-boosted
tree model LightGBM and a long short-term memory network (LSTM). Four are pretrained
time-series foundation models: Chronos-2, TimesFM-2.5 and TiRex, used without any
training on the station ("zero-shot"), and Chronos-2 fine-tuned on each station. The
*NWP setting* adds archived numerical weather prediction (NWP) forecasts to the
clear-sky template and calendar features; the *no-weather setting* does not. Every model
is scored on the same quantity, power, and on the same hours.

The design addresses four research questions:

- **RQ1.** How accurate are zero-shot foundation models for day-ahead hourly
  photovoltaic (PV) power, compared with purpose-trained models and reference forecasts?
- **RQ2.** Does fine-tuning a foundation model on a station's own history improve on its
  zero-shot forecasts enough to justify the extra compute?
- **RQ3.** What do weather forecasts add as inputs, and which models can use them?
- **RQ4.** How well calibrated are the probabilistic forecasts, and how much time and
  memory does each model need on a laptop?

<!-- TODO(Lorenzo): finalise the wording of RQ1-RQ4 together with the Introduction and keep the two chapters identical. -->

The reported results come from one complete run of this design, *panel v2*: 144 runs,
one per station, weather setting and model, each over the nine folds. An earlier run
with older data processing and fitting code, panel v1, is not reported; its results
prompted the two references added in @sec:method-models and informed the choice of
comparisons in @sec:method-tests.

## Forecasting task {#sec:method-task}

A forecast is issued once a day at local midnight, 00:00 Hong Kong Time (HKT, UTC+8, no
daylight saving time), and consists of the 24 hourly mean powers $\hat{P}_t$ of the local
day that begins then, labelled by the start of their interval (@sec:data-time) from 00:00
to 23:00. The daytime mask (@sec:data-clearsky) keeps only the hours labelled 06:00 to
18:00 in the test windows, so the scored hours lie 6 to 19 hours after the issue time.

At the issue time a model receives the *history*, every observation strictly before
local midnight (power $P_t$, clear-sky index $k_t$ and covariates), and the *future
frame*, the 24 target timestamps with the covariates known in advance
(@sec:method-covariates). The runner deletes power and clear-sky index from the future
frame, so a model cannot read the values it must predict (`pvfc.runner`).

Two simplifications make the task easier than in operation. First, the last hour before
midnight is assumed to be available at the issue time, whereas an operator who submits a
schedule earlier in the day forecasts further ahead [@yang2020verification]. Second, the
archived weather forecasts come from model runs at most a few hours old
(@sec:data-weather), closer to an analysis of the current weather than to a forecast
frozen at midnight, so the NWP setting is optimistic.

## Target: the clear-sky index {#sec:method-target}

All models except Persistence and Seasonal naive forecast the clear-sky index rather than
power. With the clear-sky power template $P^{\mathrm{cs}}_t$ of @sec:data-clearsky,

$$
k_t =
\begin{cases}
\min\left\{\max\left\{P_t / P^{\mathrm{cs}}_t,\ 0\right\},\ 1.3\right\} & \text{if } P^{\mathrm{cs}}_t > \varepsilon,\\
0 & \text{otherwise,}
\end{cases}
$$ {#eq:k}

with $\varepsilon = 10^{-6}$ kW. Dividing by the template removes most of the
deterministic daily and seasonal cycle and leaves the weather-driven part to the models
[@yang2020verification]; the clip at 1.3 bounds the ratio at sunrise and sunset, when
the template is small. The index is 0 at night and missing wherever daytime power is
missing. A forecast $\hat{k}_t$, like each quantile forecast $\hat{q}_{\tau,t}$, is
converted back to power with

$$
\hat{P}_t = \max\left\{\hat{k}_t,\ 0\right\} P^{\mathrm{cs}}_t ,
$$ {#eq:reconstruct}

which is 0 at night, applies no upper clip and preserves the order of the quantiles.

The models built with the darts library [@herzen2022darts], all except the references,
need a series without gaps. A missing hour of the index (a feed gap or an outage hour,
@sec:data-qc) therefore takes the value of the same hour on the most recent earlier day
that has one, at most seven days back, the rule of the references (@sec:method-models);
an hour still missing, on the first day of a series or in a longer gap, takes the
series' mean daytime index, or 0 outside the daytime mask
(`pvfc.normalize.fill_index_gaps`). Filled hours are inputs only, never scored.

<!-- TODO(Lorenzo): refer to the raw-power vs clear-sky-index ablation here once it has run (planned for 6-12 Oct). -->

## Covariates and weather settings {#sec:method-covariates}

All covariates are *future* covariates: their values for the 24 target hours are treated
as known at the issue time, so models may use them over the horizon as well as over the
context (@tbl:method-covariates). This is exact for the template and the calendar and,
for the weather forecasts, the optimistic approximation of @sec:method-task.

: Covariates supplied for the context window and the 24 target hours, by weather
setting, with their names in the code. {#tbl:method-covariates}

| Covariate | Code name | Source | NWP | No weather |
|:--------------------------------------|:--------------------|:------------------------|:--------:|:-----------:|
| Clear-sky power template $P^{\mathrm{cs}}_t$ (kW) | `p_clear` | Template (@sec:data-clearsky) | yes | yes |
| Hour of day, sine and cosine | `hour_sin`, `hour_cos` | Timestamp | yes | yes |
| Day of year, sine and cosine | `doy_sin`, `doy_cos` | Timestamp | yes | yes |
| Global horizontal irradiance (W/m²) | `ghi` | Archived NWP forecast | yes | no |
| Air temperature at 2 m (°C) | `temp_air` | Archived NWP forecast | yes | no |
| Wind speed at 10 m (km/h) | `wind_speed` | Archived NWP forecast | yes | no |
| Relative humidity at 2 m (%) | `rh` | Archived NWP forecast | yes | no |

The weather covariates are four variables of the Open-Meteo archive of historical
forecasts for the campus [@openmeteo], described in @sec:data-weather. The NWP setting
supplies all nine covariates, the no-weather setting only the template and the calendar
features; the latter isolates what the weather forecasts add (RQ3) and does not depend
on their quality.

@Tbl:method-models lists the covariates that each model receives. TimesFM-2.5 and TiRex
receive none, because their darts wrappers do not accept covariates; their forecasts are
the same in both settings and are reported once, under the no-weather setting. Among the
foundation models, only Chronos-2 thus uses the weather forecasts. For TimesFM-2.5 this
is a limit of darts 0.47.0, which does not expose the separate covariate regression
(XReg) of the model's own package;^[TimesFM repository, README, update of 29 October
2025; <https://github.com/google-research/timesfm>, accessed 2 October 2026.] the TiRex
version used here (tirex-ts 1.4.2) takes none.

## Models {#sec:method-models}

@tbl:method-models summarises the twelve models. Their settings were fixed in
`configs/experiment.yaml` before panel v1 and are unchanged in panel v2 (Climatology and
CLIPER, added in between, have none to tune). Two settings drew on a test fold: the
fine-tuning recipe of Chronos-2 (fine-tuned), discussed below, and the lag inputs of
LightGBM, fixed after a check on a real fold exposed a wiring bug that a regression test
now guards against.

### Reference forecasts

*Persistence* repeats the power of the same hour on the previous day, $\hat{P}_t =
P_{t-24\,\mathrm{h}}$, and *Seasonal naive* that of one week earlier, $\hat{P}_t =
P_{t-168\,\mathrm{h}}$. *Smart persistence* repeats the clear-sky index instead,
$\hat{P}_t = k_{t-24\,\mathrm{h}}\, P^{\mathrm{cs}}_t$, which corrects for the change
in solar geometry; with climatology, it is one of the two popular day-ahead references
named by @yang2020verification.^[They call it clear-sky persistence and reserve "smart
persistence" for persistence of a cloudiness index; the shorter name is kept here.] When
the lagged value is missing, these three rules step back one day at a time, up to seven
days, to the same hour (0 if none is found), never to the last value of the history,
which at midnight is a night-time zero.

<!-- TODO(Lorenzo): say why a weekly Seasonal naive is kept at all (e.g. as a second, deliberately weak reference); the config only records that a 24-h season would duplicate Persistence. -->

At a 24-hour lag Smart persistence and Persistence are almost identical, because the
template changes little from one day to the next: over the panel their skill scores
differ by less than 0.001. Two further references were therefore added. *Climatology*
forecasts the mean daytime clear-sky index of the training window, $\bar{k}$, at every
hour: $\hat{P}_t = \bar{k}\, P^{\mathrm{cs}}_t$. *CLIPER* (climatology--persistence)
combines the two [@yang2019cliper]:

$$
\hat{k}_t = \rho\, k_{t-24\,\mathrm{h}} + (1-\rho)\, \bar{k}, \qquad
\hat{P}_t = \hat{k}_t\, P^{\mathrm{cs}}_t ,
$$ {#eq:cliper}

where $\rho$ is the correlation between $k_t$ and $k_{t-24\,\mathrm{h}}$ over the pairs
of daytime hours in the training window, matched by timestamp and clipped to $[0, 1]$;
night hours are left out because their zeros would inflate $\rho$. For a stationary
index with variance $\sigma^2$, climatology has a mean squared error (MSE) of $\sigma^2$,
persistence $2(1-\rho)\,\sigma^2$ and the combination $(1-\rho^2)\,\sigma^2$, never more
than either [@yang2019cliper], so persistence is worse than climatology whenever
$\rho < 0.5$. In the training windows $\rho$ lies between 0.44 and 0.51: Smart
persistence is a weak bar. @yang2020verification strongly recommend skill over this
combination, with a single climatological mean as here, for all solar forecasting
studies. They estimate the mean and $\rho$ on the verified sample; here both come from
the training window, so the guarantee holds only approximately.

<!-- TODO(Lorenzo): eq:cliper follows the code (src/pvfc/models/baselines.py): one mean for all hours, Yang's single-valued climatology. The protocol spec had one mean per hour of day, which the code docstring says moves CLIPER's mean skill by 0.0001 (panel v1). Confirm the single-valued form; otherwise change eq:cliper, tbl:method-models and the code. -->

### Purpose-trained models

*SARIMA* models the index as a linear regression on the covariates with seasonal
autoregressive integrated moving-average errors. Its orders are chosen by the stepwise
procedure of @hyndman2008automatic, in its statsforecast implementation: seasonal
period 24 hours, at most three non-seasonal and two seasonal autoregressive and
moving-average terms, selection by the corrected Akaike information criterion (AICc),
final maximum-likelihood fit. It is fitted once per fold
on the whole training window; at each issue time statsforecast's `forward()` applies it
to the last 60 days of history, updating its state but not its orders or coefficients.
Its quantiles come from its Gaussian prediction intervals.

*LightGBM* [@ke2017lightgbm] is a direct multi-horizon regressor: one gradient-boosted
tree ensemble (booster) per target hour and quantile level, 216 per fold, each with
LightGBM's default settings (100 trees of at most 31 leaves, learning rate 0.1) and the
pinball loss of its quantile (@sec:method-metrics) as objective. Each sees the index over the
previous local day and every covariate at each of the 24 target hours.

The *LSTM* [@hochreiter1997lstm], built with darts' `RNNModel`, has one layer of 32
units and a linear output. At each step it reads the previous hour's index and the
covariates of the hour it predicts. It is trained for 30 epochs on the squared error
over 192-hour sequences (batches of 32, Adam [@kingma2015adam] at darts' default
learning rate of $10^{-3}$, no early stopping, inputs scaled to $[0, 1]$ on the training
window). At the issue time it reads the last 168 hours and forecasts the 24 hours one
step at a time, feeding back its own predictions, as point forecasts only. It was
configured once and not tuned: a common deep-learning baseline, not an optimised
competitor.

<!-- TODO(Lorenzo): confirm where the LSTM settings (168 h input, 32 units, 30 epochs) come from, so the text can say so. -->

### Foundation models

Foundation models, pretrained on large collections of time series, forecast a new series
from its context window without training on it. *Chronos-2* [@ansari2025chronos2]
is an encoder-only transformer whose group attention relates the target to its
covariates, so it can use covariates without task-specific training. We use its small
checkpoint (`autogluon/chronos-2-small`, 28 M parameters), which trails the authors'
120 M-parameter base model by as little as one percentage point of skill on the
GIFT-Eval benchmark, at nearly twice the inference speed [@ansari2025chronos2], with a
336-hour context. *TimesFM-2.5* (September 2025)
[@das2024timesfm] is a decoder-only transformer with about 200 M parameters, and *TiRex*
[@auer2025tirex] a recurrent model with 35 M parameters based on xLSTM, an extended LSTM
[@beck2024xlstm]; both read a 512-hour context. All three output the nine quantiles
$\tau = 0.1, \dots, 0.9$ of the 24 target hours in one pass. These contexts are far
shorter than the models allow (8,192 hours for Chronos-2-small, 16,384 for
TimesFM-2.5).

<!-- TODO(Lorenzo): confirm that speed and memory on the laptop were the reasons for the small Chronos-2 checkpoint; otherwise state the actual reason. -->

<!-- TODO(Lorenzo): state why 336 h (Chronos-2) and 512 h (TimesFM-2.5, TiRex) were chosen, and point to the context-length sweep once it has run. -->

The pretraining corpora named on the model cards do not include the HKUST dataset,
though an overlap cannot be excluded. Newer versions with covariate support,
TiRex-2 (July 2026) [@podest2026tirex2] and TimesFM 3.0 (August 2026),^[TimesFM
repository, README, "Update — August 2026"; <https://github.com/google-research/timesfm>,
accessed 2 October 2026.] are not wrapped by darts 0.47.0, used here.

<!-- TODO(Lorenzo): confirm when and why the model set was frozen (date and reason), and consider TiRex-2 and TimesFM 3.0, both with covariates, as future work in the Discussion. -->

*Chronos-2 (fine-tuned)* starts in every fold from a fresh copy of the pretrained
checkpoint and updates all its weights on the station's training window, with Adam at a
learning rate of $10^{-5}$, five epochs and batches of 32. Each training example is one
day, a 336-hour context ending at local midnight followed by the next day's 24 hours; a
fold provides 354 to 933 such windows, 12 to 30 optimiser steps per epoch. The loss is
the pinball loss summed over the checkpoint's 13 quantile levels, as darts implements
fine-tuning, with no validation split and no early stopping.

The rate of $10^{-5}$ is the default of AutoGluon's Chronos-2 model, which by default
trains low-rank adapters (LoRA) rather than all the weights. For full fine-tuning, as
here, AutoGluon recommends a lower rate such as $10^{-6}$, the default of the
chronos-forecasting library that it wraps; both default to 1,000 optimiser steps, against
60 to 150 per fold here.^[AutoGluon 1.6.1 model zoo, `Chronos2` fine-tuning
hyperparameters (<https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-model-zoo.html>,
accessed 2 October 2026); chronos-forecasting 2.2.2, `Chronos2Pipeline.fit`.] The recipe
was chosen in a smoke test on the first fold of the development station, CYT (Station 2),
where it beat the zero-shot model and two runs at PyTorch's default rate for Adam,
$10^{-3}$: skill over Smart persistence +0.35 in 122 s, against +0.24 zero-shot, +0.25 in
574 s (one epoch over all hourly windows) and +0.08 (five epochs over the daily windows),
scored on the median and on the data processing of the time (`configs/experiment.yaml`).
A rate of $10^{-6}$ was not tried; a sensitivity check is planned. No other station-fold
influenced the recipe.

<!-- TODO(Lorenzo): add lr 1e-6 (AutoGluon's recommendation for full fine-tuning, chronos-forecasting's default) to the fine-tuning sensitivity ablation of 6-12 Oct and report the result here. -->

: The twelve models. *Size*: parameters, counted from the checkpoints for the
foundation models (TimesFM-2.5 uses 203 M of the 231 M stored); the LSTM has 5,153
without weather inputs; SARIMA has at most ten autoregressive and moving-average (ARMA)
terms plus regressor coefficients. *Context*: input window at the issue time. *All*:
every covariate of the setting. *Training per fold*: what is estimated in each fold,
from scratch. *Licence*: of the pretrained weights. {#tbl:method-models}

| Model | Family | Size | Context | Covariates | Output | Training per fold | Licence |
|:-------------------|:--------------|:-----------|:---------|:--------------|:------------|:--------------|:----------------|
| Persistence | Reference | -- | $P_{t-24\,\mathrm{h}}$ | -- | Point | -- | -- |
| Seasonal naive | Reference | -- | $P_{t-168\,\mathrm{h}}$ | -- | Point | -- | -- |
| Smart persistence | Reference | -- | $k_{t-24\,\mathrm{h}}$ | $P^{\mathrm{cs}}$ | Point | -- | -- |
| Climatology | Reference | 1 | -- | $P^{\mathrm{cs}}$ | Point | $\bar{k}$ | -- |
| CLIPER | Reference | 2 | $k_{t-24\,\mathrm{h}}$ | $P^{\mathrm{cs}}$ | Point | $\bar{k}$ and $\rho$ | -- |
| SARIMA | Statistical | ≤ 10 ARMA | 60 days | All | Quantiles | Order search and fit | -- |
| LightGBM | Boosted trees | 216 × 100 trees | 24 h | All | Quantiles | 216 boosters | -- |
| LSTM | Recurrent network | 5,665 | 168 h | All | Point | 30 epochs | -- |
| Chronos-2 (zero-shot) | Foundation (encoder) | 28 M | 336 h | All | Quantiles | None | Apache-2.0 |
| Chronos-2 (fine-tuned) | Foundation (encoder) | 28 M | 336 h | All | Quantiles | 5 epochs | Apache-2.0 |
| TimesFM-2.5 | Foundation (decoder) | 203 M | 512 h | None | Quantiles | None | Apache-2.0 |
| TiRex | Foundation (xLSTM) | 35 M | 512 h | None | Quantiles | None | NXAI Community |

## Evaluation protocol {#sec:method-protocol}

### Walk-forward folds

The models are evaluated by walk-forward validation on whole local days, a form of
rolling-origin evaluation with an expanding training window [@tashman2000out]. The folds
are anchored to the calendar: the first test day is 5 July 2022 for every station,
whatever the start of its record, so all stations are scored on the same days. Nine
30-day test windows start 60 days apart, each preceded by a one-day *embargo* that
belongs neither to the training nor to the test window. Every training window starts at
the beginning of the station's record and holds at least 365 days, the fit period of the
clear-sky template (@sec:data-clearsky): 368 to 948 days. The folds (@tbl:method-folds)
contain 270 issue days and 6,480 target hours per station, 3,041 of them in the daytime
mask; they cover all four seasons, unevenly (four windows in autumn, one in winter).

: Test windows of the nine folds (local dates, HKT) and training days, which vary
across stations; meteorological seasons, by the majority of a window's days.
{#tbl:method-folds}

| Fold | Embargo day | Test window (30 days) | Season | Training days |
|:---:|:------------|:-------------------------|:-------|------:|
| 0 | 4 Jul 2022 | 5 Jul -- 3 Aug 2022 | Summer | 368--468 |
| 1 | 2 Sep 2022 | 3 Sep -- 2 Oct 2022 | Autumn | 428--528 |
| 2 | 1 Nov 2022 | 2 Nov -- 1 Dec 2022 | Autumn | 488--588 |
| 3 | 31 Dec 2022 | 1 Jan -- 30 Jan 2023 | Winter | 548--648 |
| 4 | 1 Mar 2023 | 2 Mar -- 31 Mar 2023 | Spring | 608--708 |
| 5 | 30 Apr 2023 | 1 May -- 30 May 2023 | Spring | 668--768 |
| 6 | 29 Jun 2023 | 30 Jun -- 29 Jul 2023 | Summer | 728--828 |
| 7 | 28 Aug 2023 | 29 Aug -- 27 Sep 2023 | Autumn | 788--888 |
| 8 | 27 Oct 2023 | 28 Oct -- 26 Nov 2023 | Autumn | 848--948 |

### Fitting and issuing

Each model is fitted once per fold, on the training window, and then issues the 30 daily
forecasts of the test window without being re-fitted. Its inputs at each issue time run
up to local midnight, including the embargo day and the earlier test days: the embargo
separates the last training target from the first test day but withholds no input data,
as for a forecaster that is re-trained now and then but run every day.

Every fit starts from scratch, so no fold inherits anything estimated in an earlier one:
the references, SARIMA and LightGBM are estimated anew, the LSTM starts from the same
seeded initialisation and Chronos-2 (fine-tuned) from a fresh copy of the pretrained
checkpoint. The zero-shot foundation models are never trained; their checkpoint is loaded
in the first fold and kept.

### Comparability

One splitter produces the folds for all models, the runner stops if a model's
predictions miss any hour of a test window, and the reporting script excludes any run
whose scored hours differ from those of the station's other runs, so every model is
scored on exactly the same timestamps. Each run has its own operating-system process,
which isolates crashes and memory measurements.

## Metrics {#sec:method-metrics}

### Scored hours and point forecasts

The headline scores use the daytime mask of @sec:data-clearsky, the hours whose apparent
solar zenith angle at mid-hour is below 85°, as @yang2020verification recommend. Within
the mask they use every hour with a valid power value after quality control
(@sec:data-qc); outage hours and other gaps are missing data and never scored. This
leaves between 2,914 and all 3,041 daytime test hours per station. Outside the mask lie
the night, where all models forecast zero or almost zero, and about 500 low-sun hours per
station holding 0.2–0.3% of the energy measured in the test windows; scores over all
hours are kept as a sensitivity check. $\mathcal{H}_s$ denotes the scored hours of
station $s$, pooled over the nine test windows.

<!-- TODO(Lorenzo): ETL v4 (like v3) flags outage hours only where P_cs >= 0.1 C, so zero readings at the low-sun edges of outage days and feed gaps escape the rule and are scored: 30 hours in the test windows (SQ1 9, all at 06:00 on 8-16 May 2023; CYT (Station 2) 11 and SQ15 10, mostly the 17:00 hour after their 07:00-17:00 feed gaps), 0.17% of the 17,919 scored hours. sec:data-qc now states this, with its own TODO; decide there whether to extend the outage rule to whole outage days in a later ETL. Evidence: scratchpad ch4/v4_stats.py. -->

For the deterministic models, the references and the LSTM, the point forecast is the
model's own output; for the probabilistic models it is the mean of the nine quantiles,

$$
\hat{P}_t = \frac{1}{9} \sum_{\tau \in \mathcal{T}} \hat{q}_{\tau,t}, \qquad
\mathcal{T} = \{0.1, 0.2, \dots, 0.9\},
$$ {#eq:point}

an estimate of the predictive mean that ignores the tails beyond the outer quantiles. The
expected squared error is minimised by the predictive mean, not the median
[@gneiting2011making], so scoring the probabilistic models on their median would
handicap them against the LSTM, trained on the squared error; the mean absolute error,
minimised by the median, is secondary for them.

All metrics are computed from the saved forecasts, which keep each model's raw output
(`experiments/rescore.py`, `experiments/report.py`), so a change of scoring rule needs no
re-run. The rules here are scoring version 2 (version 1 scored the probabilistic models
on their median).

### Point metrics and skill

The point metrics are the root mean squared error (RMSE), the mean absolute error (MAE)
and the mean bias error (MBE, positive for over-forecasts), each pooled over
$\mathcal{H}_s$; for the RMSE,

$$
\mathrm{RMSE}_s = \sqrt{\frac{1}{|\mathcal{H}_s|} \sum_{t \in \mathcal{H}_s}
\left(\hat{P}_{s,t} - P_{s,t}\right)^2 } .
$$

Divided by the station's nameplate capacity $C_s$, never by the observed power, which is
close to zero at dawn and dusk, they give the normalised errors nRMSE and nMAE, comparable
across station sizes. The main measure is the RMSE skill score over a reference,

$$
\mathrm{SS}_s = 1 - \frac{\mathrm{RMSE}_s^{\,\mathrm{model}}}{\mathrm{RMSE}_s^{\,\mathrm{ref}}},
$$ {#eq:skill}

computed per station on the same hours for model and reference: a positive score means
that the model beats the reference, and $\mathrm{SS}_s = 0.2$ a 20% lower RMSE. Smart
persistence, among the most common day-ahead references [@yang2020verification], gives
the headline numbers; CLIPER, the training-window version of the standard of reference
recommended by @yang2020verification, is the stricter second reference. The reported
value is the mean of $\mathrm{SS}_s$ over the six stations with its range; the other
metrics are aggregated in the same way (@tbl:master-skill).

### Probabilistic metrics

The quantile forecasts are scored with the pinball loss
$\ell_\tau(y, q) = \max\{\tau\,(y - q),\ (\tau - 1)\,(y - q)\}$, whose expectation is
minimised by the true $\tau$-quantile [@gneiting2011making], and with the continuous
ranked probability score (CRPS), a proper scoring rule for the whole predictive
distribution [@gneiting2007strictly], approximated as twice the mean pinball loss,

$$
\mathrm{CRPS}_s \approx \frac{1}{|\mathcal{H}_s|} \sum_{t \in \mathcal{H}_s}
\frac{2}{9} \sum_{\tau \in \mathcal{T}} \ell_\tau\!\left(P_{s,t},\ \hat{q}_{\tau,s,t}\right),
$$ {#eq:crps}

a discrete version of $\mathrm{CRPS} = 2\int_0^1 \ell_\tau\,\mathrm{d}\tau$
[@gneiting2011comparing], divided by $C_s$ like the point errors. If all nine quantiles
equal the point forecast, @eq:crps reduces to the absolute error, because the levels are
symmetric about 0.5: the CRPS of a deterministic forecast is its MAE, so all models can
be compared on it, also through a CRPS skill score over Smart persistence. Calibration is
measured for the probabilistic models by the prediction interval coverage probability
(PICP) of the central 80% interval $[\hat{q}_{0.1,t}, \hat{q}_{0.9,t}]$, the share of
observations inside it (ideally 0.8), reported with the interval's mean width divided by
$C_s$, and by the share of observations at or below each $\hat{q}_{\tau,t}$ (ideally
$\tau$), the data of a reliability diagram (@tbl:probabilistic). Crossing quantiles are sorted before scoring
[@chernozhukov2010quantile].

## Statistical testing {#sec:method-tests}

The six stations share one campus and one weather regime: their hourly absolute Smart
persistence errors, divided by capacity, have a mean pairwise correlation of 0.93.
Treating station-hours as independent would overstate the evidence, so the tests use
daily losses. For a model $A$ and a test day $d$, the daily loss averages the
capacity-normalised squared error over the scored hours $\mathcal{H}_{s,d}$ of each
station-day, then with equal weight over the stations $\mathcal{S}_d$ with scored hours
that day (all six on 268 of the 270 days):

$$
L^{A}_{d} = \frac{1}{|\mathcal{S}_d|} \sum_{s \in \mathcal{S}_d} \frac{1}{|\mathcal{H}_{s,d}|}
\sum_{t \in \mathcal{H}_{s,d}} \left( \frac{\hat{P}^{A}_{s,t} - P_{s,t}}{C_s} \right)^{2} .
$$ {#eq:daily-loss}

The Diebold--Mariano (DM) test [@diebold1995comparing] compares models $A$ and $B$
through the loss differential $D_d = L^{A}_{d} - L^{B}_{d}$ over the $n = 270$ test days,
taken in date order, under the null hypothesis that its expectation is zero. Consecutive
days share weather, so the variance of $\bar{D}$ is estimated with the
heteroskedasticity- and autocorrelation-consistent (HAC) estimator of @newey1987simple,
with Bartlett weights and $m = \lfloor n^{1/3} \rfloor = 6$ lags, a choice that follows
the $n^{1/3}$ growth of the lag window that is optimal for this kernel
[@andrews1991heteroskedasticity]:

$$
\hat{\gamma}_j = \frac{1}{n} \sum_{d=j+1}^{n} \left(D_d - \bar{D}\right)\left(D_{d-j} - \bar{D}\right),
\qquad
\hat{\sigma}^2 = \hat{\gamma}_0 + 2 \sum_{j=1}^{m} \left(1 - \frac{j}{m+1}\right) \hat{\gamma}_j,
$$
$$
\mathrm{DM}^{*} = \sqrt{\frac{n + 1 - 2h + h(h-1)/n}{n}}\;
\frac{\bar{D}}{\sqrt{\hat{\sigma}^2 / n}}, \qquad h = 1 .
$$ {#eq:dm}

The square-root factor is the small-sample correction of @harvey1997testing for a
horizon of $h = 1$ day, and the two-sided p-value comes from Student's t distribution
with $n - 1$ degrees of freedom; with $n = 270$ both corrections are small. A
moving-block bootstrap of the differentials [@kunsch1989jackknife], with blocks of seven
days and 10,000 resamples, gives a second p-value that relies neither on the t
approximation nor on the choice of lags. Lags and blocks also span the gaps between test
windows.

The tests form one family of eleven comparisons, each tied to a research question:

1. LightGBM vs Chronos-2 (zero-shot), NWP setting (RQ1);
2. LightGBM vs Chronos-2 (fine-tuned), NWP setting (RQ1, RQ2);
3. Chronos-2 (fine-tuned) vs SARIMA, NWP setting (RQ1);
4. Chronos-2 (fine-tuned) vs Chronos-2 (zero-shot), NWP setting (RQ2);
5. Chronos-2 (fine-tuned) vs Chronos-2 (zero-shot), no-weather setting (RQ2);
6. SARIMA vs Chronos-2 (zero-shot), no-weather setting (RQ1);
7. Chronos-2 (zero-shot) vs TiRex, no-weather setting (RQ1);
8. TiRex vs TimesFM-2.5, no-weather setting (RQ1);
9. LightGBM, NWP vs no-weather setting (RQ3);
10. Chronos-2 (zero-shot), NWP vs no-weather setting (RQ3);
11. CLIPER vs Smart persistence (choice of reference).

The family was chosen after an exploratory look at day-level tests on the panel v1
forecasts (eight of the ten pairs tested there kept, items 1, 10 and 11 added) and fixed
in the reporting script on 2 October 2026, before panel v2 ran. It is therefore
pre-specified for the reported results, although panel v1 covered the same test days.
The p-values are adjusted with the Holm--Bonferroni procedure [@holm1979simple], which
keeps the probability of at least one false rejection in the family at or below 5%
(@tbl:significance).

## Compute measurement {#sec:method-compute}

Compute cost is measured per fold as wall-clock time around the model alone, without the
reference forecast. The training time is that of the model's `fit()` call, which for the
LSTM and Chronos-2 (fine-tuned) includes building the model, for the latter reloading
the pretrained checkpoint; the zero-shot foundation models load their checkpoint only in
the first fold, in a few seconds, so their median training time is close to zero. The
prediction time covers the fold's 30 daily forecasts. @Tbl:compute reports medians over
the 54 station-folds of each model and setting.

Peak memory is the process's peak resident set size reported by the operating system
(`pvfc.utils.peak_rss_mb`), a high-water mark over the life of the process. It leaves
out the memory that PyTorch allocates for the graphics processor (GPU) through Apple's
Metal framework, although the GPU and the central processor (CPU) share one physical
memory (@sec:appendix-repro), so for the five neural models, the LSTM and the four
foundation-model variants, the reported peak is a lower bound.

All runs used one fanless laptop, an Apple MacBook Air with an M4 chip and 16 GB of
memory shared by CPU and GPU; timings are not corrected for thermal throttling. The
neural models ran on the GPU through PyTorch's Metal Performance Shaders (MPS) backend,
with a fallback to the CPU for unsupported operations, the others on the CPU. The 144
runs of panel v2 were executed one at a time, from 21:37 on 2 October 2026.

<!-- TODO(Lorenzo): panel v2 wall time from results/logs/panel.out -->

## Reproducibility {#sec:method-repro}

Every run is identified by a hash of its configuration: the data, including the version
of the data processing (`data.etl_version`, here 4), the folds, the model and the seed.
The hash keys the rows of the results store and names the file of saved forecasts, so
runs on differently processed data never share a hash, and `experiments/run.py` refuses
a configuration whose version differs from the one the code builds
(`pvfc.data.ETL_VERSION`). All random number generators are seeded with 0, so the LSTM
starts from the same weights in every fold; with a single seed, the sensitivity of the
neural models to their initialisation was not measured. Regression tests (`tests/`)
check the protocol properties of this chapter; @sec:appendix-repro gives the full
checklist.

<!-- TODO(Lorenzo): decide whether to run extra seeds for the LSTM and Chronos-2 (fine-tuned); if not, keep the single-seed caveat. Before any multi-seed run, pass the configuration's seed to the models: darts_wrapper takes random_state from the model params (default 0), and run.py passes only m["params"], so changing `seed` in experiment.yaml would not change the LSTM or fine-tuning initialisation. -->
