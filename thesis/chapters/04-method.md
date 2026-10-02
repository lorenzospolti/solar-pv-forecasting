# Method {#sec:method}

This chapter describes how the forecasts were produced and scored; the data (stations,
quality control, clear-sky template and weather feed) are described in Chapter
[-@sec:data]. All models run through the same code, so what follows applies to every
model unless stated otherwise.

## Design overview and research questions {#sec:method-overview}

The experiment crosses four factors: the six stations of the panel (@sec:data-panel), two
weather settings, twelve models and nine test windows, called folds. Five models are
simple reference forecasts. Three are purpose-trained on each station's history: a
seasonal autoregressive integrated moving-average model (SARIMA), the gradient-boosted
tree model LightGBM and a long short-term memory network (LSTM). Four are pretrained
time-series foundation models: Chronos-2 (zero-shot), TimesFM-2.5 and TiRex, used without
any training on the station ("zero-shot"), and Chronos-2 (fine-tuned), which is trained
further on each station. The *NWP setting* adds archived numerical weather prediction
(NWP) forecasts to the clear-sky template and calendar features; the *no-weather setting*
does not. Every model is scored on the same quantity, power, and on the same hours.

The design is built to answer four research questions:

- **RQ1.** How accurate are zero-shot foundation models for day-ahead hourly
  photovoltaic (PV) power, compared with purpose-trained models and reference forecasts?
- **RQ2.** Does fine-tuning a foundation model on a station's own history improve on its
  zero-shot forecasts enough to justify the extra compute?
- **RQ3.** What do weather forecasts add as inputs, and which models can use them?
- **RQ4.** How well calibrated are the probabilistic forecasts, and how much time and
  memory does each model need on a laptop?

<!-- TODO(Lorenzo): finalise the wording of RQ1-RQ4 together with the Introduction and keep the two chapters identical. -->

The first full run, panel v1 (1--2 October 2026), covered ten of the twelve models;
Climatology and CLIPER were added afterwards as references. Each of its 120 combinations
of station, weather setting and model was one run of nine folds.

## Forecasting task {#sec:method-task}

A forecast is issued once a day at local midnight, 00:00 Hong Kong Time (HKT, UTC+8,
without daylight saving time), and consists of the 24 hourly mean powers $\hat{P}_t$ of
the local day that begins then. Hours are labelled by the start of their interval
(@sec:data-time), so the targets are labelled 00:00 to 23:00. In the test windows the
daytime mask (@sec:data-clearsky) keeps only the hours labelled 06:00 to 18:00, so the
scored hours lie 6 to 19 hours after the issue time.

At the issue time a model receives the *history*, every observation strictly before
local midnight (power $P_t$, clear-sky index $k_t$ and covariates), and the *future
frame*, the 24 target timestamps with the covariates known in advance for them
(@sec:method-covariates). The runner deletes the power and clear-sky-index columns from
the future frame before passing it on, so a model cannot read the values it must predict,
even by mistake (`pvfc.runner`).

Two simplifications should be kept in mind. First, the last hour before midnight is
assumed to be available at the issue time; an operator who submits a schedule earlier in
the day forecasts further ahead, and 24-hour persistence then becomes 48-hour
persistence [@yang2020verification]. Second, the archived weather forecasts are
compiled from the first hours of successive model runs (@sec:data-weather), so they are
closer to an analysis, the weather model's best estimate of the current state, than to a
forecast frozen at midnight. The NWP setting is therefore optimistic for the models that
use the weather forecasts.

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

with $\varepsilon = 10^{-6}$ kW (`pvfc.normalize.clearsky_index`). The template
estimates the power the station would produce under a clear sky, so $k_t$ is close to 1
in clear hours and falls towards 0 under thick cloud. Dividing by it removes most of the
deterministic daily and seasonal cycle and leaves the weather-driven part to the models
[@yang2020verification]. The clip at 1.3 bounds the ratio in the few hours where it is
unstable, mostly at sunrise and sunset, when the template is small (@sec:data-clearsky).
The index is 0 at night and missing wherever daytime power is missing, because a data gap
is not zero generation.

A forecast $\hat{k}_t$ is converted back to power with

$$
\hat{P}_t = \max\left\{\hat{k}_t,\ 0\right\} P^{\mathrm{cs}}_t ,
$$ {#eq:reconstruct}

and each quantile forecast $\hat{q}_{\tau,t}$ in the same way
(`pvfc.normalize.reconstruct_power`). For these models the forecast is therefore 0 at
night whatever they predict there; no upper clip is applied, and the conversion preserves
the order of the quantiles. Because night values are 0, the index series still switches
on and off every day, but its daytime level no longer follows the sun.

Before the index is passed to the non-reference models, all built with the darts library
[@herzen2022darts], missing daytime values are filled forward from the last valid hour
(`_target_series` in `pvfc.models.darts_wrapper`). Only the two stations with hourly meters
have such gaps, mostly on outage days (@sec:data-qc). There the repeated value is a zero:
the short-gap interpolation of @sec:data-qc writes zeros into the first two hours of each
outage, and the forward fill carries them on. Hours that are still missing after
processing are never scored, but the interpolated hours are. As a result, 38 daytime test
hours without a raw reading, 15 at CYT (Station 2) and 23 at SQ15, count as observations:
about 0.2% of all scored hours.

<!-- TODO(Lorenzo): if the ETL is fixed (ETL v3, see the TODO in sec:data-qc), drop the last two sentences of this paragraph and the matching clause in sec:method-metrics. -->

<!-- TODO(Lorenzo): refer to the raw-power vs clear-sky-index ablation here once it has run (planned for 6-12 Oct). -->

## Covariates and weather settings {#sec:method-covariates}

All covariates are *future* covariates: their values for the 24 target hours are treated
as known at the issue time, so models may use them over the horizon as well as over the
context (@tbl:method-covariates). For the template and the calendar this is exact; for
the archived weather forecasts it is the optimistic approximation of @sec:method-task.

: Covariates known at the issue time, supplied for the context window and the 24 target
hours, and whether each weather setting includes them. Code names are those of the
repository. {#tbl:method-covariates}

| Covariate | Code name | Source | NWP | No weather |
|:--------------------------------------|:--------------------|:------------------------|:--------:|:-----------:|
| Clear-sky power template $P^{\mathrm{cs}}_t$ (kW) | `p_clear` | Template (@sec:data-clearsky) | yes | yes |
| Hour of day, sine and cosine | `hour_sin`, `hour_cos` | Timestamp | yes | yes |
| Day of year, sine and cosine | `doy_sin`, `doy_cos` | Timestamp | yes | yes |
| Global horizontal irradiance (W/m²) | `ghi` | Archived NWP forecast | yes | no |
| Air temperature at 2 m (°C) | `temp_air` | Archived NWP forecast | yes | no |
| Wind speed at 10 m (km/h) | `wind_speed` | Archived NWP forecast | yes | no |
| Relative humidity at 2 m (%) | `rh` | Archived NWP forecast | yes | no |

The calendar features have periods of 24 hours and 365.25 days. They are computed from
UTC timestamps, which shifts their phase by a constant 8 hours but carries the same
information. The weather variables are four of those in the Open-Meteo archive of
historical forecasts for the campus [@openmeteo], with the irradiance shifted to the
hour-start convention (@sec:data-weather; `pvfc.data.WEATHER_COLS`). Covariate gaps are
filled forward, then backward. In the NWP setting (configuration value `nwp`) the models
receive all nine covariates, in the no-weather setting (`none`) only the template and the
calendar features. The no-weather setting isolates what the weather forecasts add (RQ3)
and does not depend on their quality.

SARIMA, LightGBM, the LSTM and both Chronos-2 variants receive all the covariates of the
setting. Smart persistence, Climatology and CLIPER use only the template; Persistence and
Seasonal naive use no covariates. TimesFM-2.5 and TiRex receive none, because their darts
wrappers do not accept covariates (`supports_future_covariates` is `False`). They forecast
from the history of the index alone, so their forecasts are identical in the two settings
by construction; the saved panel forecasts confirm it (maximum absolute difference 0). In
this pipeline, only Chronos-2 receives the weather forecasts. For TimesFM-2.5 the limit
lies in the software: its own package offers covariates through a separate regression
step (XReg), which darts 0.47.0 does not expose.^[TimesFM repository, README, update of
29 October 2025; <https://github.com/google-research/timesfm>, accessed 2 October 2026.]
The TiRex version used here (tirex-ts 1.4.2) takes no covariates.

## Models {#sec:method-models}

@tbl:method-models summarises the twelve models. The settings of the ten models in panel
v1 were fixed in `configs/experiment.yaml` before that run; Climatology and CLIPER, added
afterwards, have no settings to tune. Two decisions drew on a test fold. One is
the fine-tuning recipe of Chronos-2 (fine-tuned), discussed below; the other is a fix to
the lag inputs of LightGBM, a wiring bug that a check on a real fold exposed and that a
regression test now guards against (`tests/test_models_darts.py`). No other setting was
chosen or tuned on the test windows.

### Reference forecasts

A reference forecast is a simple rule that any useful model should beat. *Persistence*
repeats the power of the same hour on the previous day, $\hat{P}_t =
P_{t-24\,\mathrm{h}}$, and *Seasonal naive* that of one week earlier, $\hat{P}_t =
P_{t-168\,\mathrm{h}}$; a daily season would duplicate Persistence
(`configs/experiment.yaml`). *Smart persistence* repeats the clear-sky index instead,
$\hat{P}_t = k_{t-24\,\mathrm{h}}\, P^{\mathrm{cs}}_t$, which corrects for the change in
solar geometry. @yang2020verification call this rule clear-sky persistence and count it
among the most popular references for intra-hour and intra-day forecasts. For day-ahead
forecasts they name two popular references: 24-hour persistence of the clear-sky index,
which is what Smart persistence is here, and climatology.^[@yang2020verification reserve the name "smart
persistence" for persistence of a cloudiness index, which accounts for every major
atmospheric effect except clouds. The shorter name is kept here for persistence of the
clear-sky index.] When the lagged value is missing, these three rules step back one day
at a time, up to seven days, to an earlier value at the same hour, and use 0 if none is
found. They never use the last value of the history, which at midnight is a night-time
zero (`pvfc.models.baselines._lookup_same_hour`).

<!-- TODO(Lorenzo): say why a weekly Seasonal naive is kept at all (e.g. as a second, deliberately weak reference); the config only records that a 24-h season would duplicate Persistence. -->

At a 24-hour lag, Smart persistence and Persistence are almost identical on these data,
because the template changes little from one day to the next: over the panel their skill
scores differ by less than 0.001. The clear-sky correction thus adds almost nothing at this
lag, and two further references were added. *Climatology* forecasts the mean
daytime clear-sky index of the training window, $\bar{k}$, at every hour:
$\hat{P}_t = \bar{k}\, P^{\mathrm{cs}}_t$. *CLIPER* (climatology--persistence) combines
it with Smart persistence [@yang2019cliper]:

$$
\hat{k}_t = \rho\, k_{t-24\,\mathrm{h}} + (1-\rho)\, \bar{k}, \qquad
\hat{P}_t = \hat{k}_t\, P^{\mathrm{cs}}_t ,
$$ {#eq:cliper}

where $\rho$ is the correlation between $k_t$ and $k_{t-24\,\mathrm{h}}$ over the pairs
of daytime hours in the training window, clipped to $[0, 1]$, and $k_{t-24\,\mathrm{h}}$
is looked up as for Smart persistence (`pvfc.models.baselines.CLIPER`). Only daytime
pairs enter, matched by timestamp, because the zeros that @eq:k assigns to night hours
would inflate $\rho$.

If the clear-sky index is treated as a stationary series with variance $\sigma^2$,
climatology has a mean squared error (MSE) of $\sigma^2$, persistence
$2(1-\rho)\,\sigma^2$ and the combination $(1-\rho^2)\,\sigma^2$, never more than either
[@yang2019cliper]. Persistence is thus worse than climatology whenever $\rho < 0.5$. In
the training windows $\rho$ lies between 0.45 and 0.52: from one day to the next the
clear-sky index is only weakly correlated, so Smart persistence is a weak bar.
@yang2020verification strongly recommend the skill score in root mean squared error
(RMSE; @sec:method-metrics) over this combination, with a single climatological mean as
here, in all solar forecasting studies. They estimate the mean and $\rho$ on the verified
sample; here both come from each fold's training window, so the guarantee holds only
approximately.

<!-- TODO(Lorenzo): the text and eq:cliper now follow the code (src/pvfc/models/baselines.py, tests/test_baselines.py): one mean for all hours (Yang's single-valued climatology) and rho = lag-24 h correlation of the daytime index. The protocol spec had one mean per hour of day and rho of the anomalies k - kbar_h; the code docstring reports that the hourly version moves CLIPER's mean skill by 0.0001, and Yang et al. (2020, Sec. 4.2) recommend only the single-valued form. If you prefer the hourly version, change eq:cliper and tbl:method-models (13 means, 14 values: daytime hours exist only at 06-18 local) and add a fallback in the code for the low-sun hours 05:00 and 19:00. -->

### Purpose-trained models

*SARIMA* models the index as a linear regression on the covariates whose errors follow a
seasonal autoregressive integrated moving-average process. Its orders are chosen by the
stepwise procedure of @hyndman2008automatic, in the statsforecast implementation called
through darts [@herzen2022darts]. The search uses a seasonal period of 24 hours and at most
three non-seasonal and two seasonal autoregressive and moving-average terms, ranks the
candidates by the corrected Akaike information criterion (AICc) and ends with a
maximum-likelihood fit. SARIMA is fitted once per fold, on the whole training window. At
each issue time statsforecast's `forward()` applies the fitted model to the last 60 days
of history, which updates its state but not its orders or coefficients, so SARIMA is not
re-fitted daily. Its quantiles come from its Gaussian prediction intervals.

*LightGBM* [@ke2017lightgbm] is a gradient-boosted tree model trained as a direct
multi-horizon regressor: one booster per target hour and quantile level, 216 per fold.
Each booster has LightGBM's default settings (100 trees of at most 31 leaves, learning rate
0.1) and the pinball loss of its quantile (@sec:method-metrics) as objective. Every booster
sees the index over the previous local day and every covariate at each of the 24 target
hours (in darts, target lags $-1$ to $-24$ and future-covariate lags 0 to 23, counted
from the first forecast step). Training samples start at every hour of the training
window, whereas forecasts always start at midnight.

The *LSTM* is a recurrent neural network with long short-term memory cells
[@hochreiter1997lstm], built with darts' `RNNModel`: one layer of 32 units and a linear
output, 5,665 parameters with weather inputs and 5,153 without. At each step it reads the
previous hour's index and the covariates of the hour it predicts. It is trained on the
squared error over 192-hour sequences for 30 epochs, with batches of 32 and no early
stopping, and its inputs are scaled to $[0, 1]$ on the training window. The optimiser is
Adam [@kingma2015adam] at PyTorch's default learning rate of $10^{-3}$, which darts uses
unless told otherwise. At the issue time it reads the last 168 hours and produces the 24 hours
one step at a time, feeding back its own predictions, as point forecasts only. Its
configuration was set once and not tuned: it is a common deep-learning baseline, not an
optimised competitor.

<!-- TODO(Lorenzo): confirm where the LSTM settings (168 h input, 32 units, 30 epochs) come from, so the text can say so. -->

### Foundation models

The foundation models were pretrained on large collections of time series and forecast a
new series from its context window without training on it. *Chronos-2*
[@ansari2025chronos2] is an encoder-only transformer that reads the target together with
its covariates and relates them through group attention, so it can exploit covariates
without task-specific training. We use its small checkpoint
(`autogluon/chronos-2-small`, 28 M parameters) with a 336-hour (14-day) context. Its
authors report that this model trails their 120 M-parameter base model by as little as
one percentage point of skill on the GIFT-Eval benchmark, with nearly twice as fast
inference [@ansari2025chronos2].

<!-- TODO(Lorenzo): confirm that speed and memory on the laptop were the reasons for the small Chronos-2 checkpoint; otherwise state the actual reason. -->

*TimesFM-2.5*, version 2.5 (September 2025) of TimesFM [@das2024timesfm], is a
decoder-only transformer (`google/timesfm-2.5-200m-pytorch`). It uses about 203 M
parameters here, 197 M of them in its 20 transformer layers, in line with the 200 M of its
documentation; its checkpoint also stores an optional 28 M output head for horizons of up
to 1,024 steps, which darts does not use here.^[TimesFM repository, README, update of
15 September 2025 ("200M parameters", "an optional 30M quantile head");
<https://github.com/google-research/timesfm>, accessed 2 October 2026. The counts are
taken from the tensors of the checkpoint.] *TiRex* [@auer2025tirex] is a recurrent model
based on xLSTM, an extended LSTM architecture [@beck2024xlstm] (`NX-AI/TiRex`, 35 M
parameters). TimesFM-2.5 and TiRex read a 512-hour context and no covariates. All three
output the nine quantiles $\tau = 0.1, \dots, 0.9$ of the 24 target hours in one pass.
The contexts are far shorter than the models allow (8,192 hours for Chronos-2-small,
16,384 for TimesFM-2.5); their effect will be examined in a context-length sweep.

<!-- TODO(Lorenzo): state why 336 h (Chronos-2) and 512 h (TimesFM-2.5, TiRex) were chosen, and point to the context-length sweep once it has run. -->

Chronos-2 and TimesFM-2.5 are released under the Apache-2.0 licence, TiRex under the
NXAI Community License, which darts makes the user accept explicitly. The pretraining
corpora named on the model cards do not include the HKUST dataset, but they were not
inspected series by series, so an overlap cannot be ruled out entirely. Newer versions of
two of the models appeared while the project was under way: TiRex-2 in July 2026, which
accepts past and future covariates [@podest2026tirex2], and TimesFM 3.0 in August 2026,
also with covariate support.^[TimesFM repository, README, "Update — August 2026";
<https://github.com/google-research/timesfm>, accessed 2 October 2026.] Neither is
wrapped by darts 0.47.0, the version used here, so the comparison keeps the versions
above.

<!-- TODO(Lorenzo): confirm when and why the model set was frozen (date and reason), and consider TiRex-2 and TimesFM 3.0, both with covariates, as future work in the Discussion. -->

*Chronos-2 (fine-tuned)* starts from the same checkpoint (in the first fold; later folds
continue from the previous fold's weights, @sec:method-protocol) and updates all its
weights on the station's training window. Its recipe was fixed before the panel run, on
one development station-fold (below): Adam at a learning rate of $10^{-5}$, five epochs
and batches of 32. There is one training window per day, a 336-hour context ending at
local midnight followed by the next day's 24 hours, so that every training example has
the shape of the task itself. A fold provides 354 to 933 windows, 12 to 30 optimiser steps
per epoch. The loss is the pinball loss summed over the checkpoint's 13 quantile levels
and averaged over hours and windows, as darts implements fine-tuning; there is no
validation split and no early stopping.

The learning rate of $10^{-5}$ is the default of AutoGluon's Chronos-2 model, which by
default trains low-rank adapters (LoRA) rather than all the weights. For full
fine-tuning, as here, both AutoGluon and the original chronos-forecasting library, which
AutoGluon wraps, suggest $10^{-6}$, and both default to 1,000 optimiser steps, against 60
to 150 per fold here.^[AutoGluon 1.6.1, forecasting model zoo, `Chronos2` model,
hyperparameters `fine_tune_mode`, `fine_tune_lr` and `fine_tune_steps`;
<https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-model-zoo.html>, accessed
2 October 2026. chronos-forecasting 2.2.2, `Chronos2Pipeline.fit`, arguments
`finetune_mode`, `learning_rate` and `num_steps`.] The recipe was chosen in a smoke test
on the first fold of the development station, CYT (Station 2), against two runs at
$10^{-3}$, PyTorch's default for Adam. In skill over Smart persistence (scored on the
median, the rule at the time), the chosen recipe reached +0.35 in 122 s, against +0.24
zero-shot. At $10^{-3}$, one epoch over all hourly windows reached +0.25 in 574 s, and
five epochs over the daily windows +0.08 (`configs/experiment.yaml`). A rate of
$10^{-6}$ was not tried. That station-fold is one of the 54 in the evaluation; no other fold
influenced the recipe.

<!-- TODO(Lorenzo): add lr 1e-6 (the full fine-tuning value suggested by AutoGluon and chronos-forecasting) to the fine-tuning sensitivity ablation of 6-12 Oct, and correct the comment above chronos2_ft in configs/experiment.yaml (1e-5 is AutoGluon's Chronos2 default for LoRA, not for full fine-tuning). -->

: The twelve models. *Size*: number of parameters, counted from the checkpoints for the
foundation models; TimesFM-2.5 uses 203 M of the 231 M parameters it stores; the LSTM
has 5,665 parameters in the NWP setting and 5,153 in the no-weather setting; SARIMA has
at most ten autoregressive and moving-average (ARMA) terms, chosen per fold, plus the
coefficients of its regressors. *Context*: input window at the issue time. *All*: every
covariate of the weather setting (@tbl:method-covariates). *Quantiles*:
$\tau = 0.1, \dots, 0.9$. \*Training continues from the previous fold
(@sec:method-protocol). The licence is that of the pretrained weights.
{#tbl:method-models}

| Model | Family | Size | Context | Covariates | Output | Training per fold | Licence |
|:-------------------|:--------------|:-----------|:---------|:--------------|:------------|:--------------|:----------------|
| Persistence | Reference | -- | $P_{t-24\,\mathrm{h}}$ | -- | Point | -- | -- |
| Seasonal naive | Reference | -- | $P_{t-168\,\mathrm{h}}$ | -- | Point | -- | -- |
| Smart persistence | Reference | -- | $k_{t-24\,\mathrm{h}}$ | $P^{\mathrm{cs}}$ | Point | -- | -- |
| Climatology | Reference | 1 | -- | $P^{\mathrm{cs}}$ | Point | $\bar{k}$ | -- |
| CLIPER | Reference | 2 | $k_{t-24\,\mathrm{h}}$ | $P^{\mathrm{cs}}$ | Point | $\bar{k}$ and $\rho$ | -- |
| SARIMA | Statistical | ≤ 10 ARMA | 60 days | All | Quantiles | Order search and fit | -- |
| LightGBM | Boosted trees | 216 × 100 trees | 24 h | All | Quantiles | 216 boosters | -- |
| LSTM | Recurrent network | 5,665 | 168 h | All | Point | 30 epochs\* | -- |
| Chronos-2 (zero-shot) | Foundation (encoder) | 28 M | 336 h | All | Quantiles | None | Apache-2.0 |
| Chronos-2 (fine-tuned) | Foundation (encoder) | 28 M | 336 h | All | Quantiles | 5 epochs\* | Apache-2.0 |
| TimesFM-2.5 | Foundation (decoder) | 203 M | 512 h | None | Quantiles | None | Apache-2.0 |
| TiRex | Foundation (xLSTM) | 35 M | 512 h | None | Quantiles | None | NXAI Community |

## Evaluation protocol {#sec:method-protocol}

### Walk-forward folds

The models are evaluated by walk-forward validation on whole local days, a form of
rolling-origin evaluation [@tashman2000out] in which the training window expands. The
folds are anchored to the calendar: the first test day is 5 July 2022 for every station,
whatever the start of its record, so all stations are scored on the same days and
averages over stations compare like with like (`pvfc.splits.ExpandingWindowSplitter`).
There are nine 30-day test windows starting 60 days apart. Each is preceded by a one-day
*embargo*, a day that belongs neither to the training window nor to the test window.
Every training window starts at the beginning of the station's record (between 23 March
and 1 July 2021) and must hold at least 365 days, the period on which the clear-sky
template is fitted (@sec:data-clearsky); the windows hold 368 to 948 days. The folds
(@tbl:method-folds) contain 270 issue days and 6,480 target hours per station, 3,041 of
them in the daytime mask, between July 2022 and November 2023. They cover all four
seasons, but unevenly: four windows fall in autumn, one in winter.

: Test windows of the nine folds, in local dates (HKT). Training starts at the beginning
of each station's record, so the number of training days varies across stations. Seasons
follow the meteorological convention, by the majority of a window's days.
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
forecasts of the test window without being re-fitted. At each issue time it receives the
full history up to local midnight, including the embargo day and the earlier test days.
The embargo therefore leaves a one-day gap between the last training target and the first
test day. It does not withhold data from the inputs, which always run up to the issue
time, as for a forecaster that is re-trained now and then but run every day.

The LSTM and Chronos-2 (fine-tuned) differ in one respect. A run builds the model once
and fits it fold after fold, and darts keeps the network between calls to `fit()`. Each
fold's training therefore continues from the weights of the previous fold instead of a
fresh initialisation (LSTM) or the pretrained checkpoint (Chronos-2); only the LSTM's
input scaling is re-fitted. By the last fold Chronos-2 (fine-tuned) has been trained for 45
epochs in total and the LSTM for 270. No test information leaks, because every training
window contains the previous one, but the training time of a fold (@sec:method-compute)
counts only that fold's epochs. All other models are estimated from scratch in every
fold.

<!-- TODO(Lorenzo): decide whether to keep this warm start (and present it as continual training) or to re-run the LSTM and Chronos-2 (fine-tuned) with a fresh start in every fold. -->

### Comparability

Every model is scored on exactly the same timestamps. One splitter produces the folds for
all models, and the runner raises an error if a model's predictions miss any hour of a
test window, so no model can be scored on a convenient subset (`pvfc.runner.run_model`).
The reference forecast is computed in the same run, on the same hours. Each combination
of station, weather setting and model runs in its own operating-system process
(`experiments/run_panel.py`), which isolates crashes and memory measurements.

## Metrics {#sec:method-metrics}

### Scored hours and point forecasts

The headline scores use the daytime mask of @sec:data-clearsky, the hours in which the
sun stands more than 5° above the horizon at mid-hour (apparent solar zenith angle below
85°), as @yang2020verification recommend. Within the mask they use every hour with a
power value after processing (@sec:data-qc), including the 38 interpolated hours of
@sec:method-target. Hours outside the mask are night hours, where all models forecast
zero or almost zero, and about 500 low-sun hours per station at the edges of the day,
which carry 0.2--0.3% of the energy measured in the test windows. Scores over all hours
are kept as a
sensitivity check. $\mathcal{H}_s$ denotes the scored hours of station $s$, pooled over
the nine test windows.

For the deterministic models, the references and the LSTM, the point forecast is the
model's own output. For the probabilistic models it is the mean of the nine quantiles,

$$
\hat{P}_t = \frac{1}{9} \sum_{\tau \in \mathcal{T}} \hat{q}_{\tau,t}, \qquad
\mathcal{T} = \{0.1, 0.2, \dots, 0.9\},
$$ {#eq:point}

an estimate of the predictive mean that ignores the tails beyond the outer quantiles. The
expected squared error is minimised by the predictive mean, not the median
[@gneiting2011making], so scoring the probabilistic models on their median would
handicap them against the LSTM, which is trained on the squared error. The mean suits
the RMSE, the main measure. The absolute error is minimised by the median instead
[@gneiting2011making], so for the probabilistic models, scored on the same mean forecast,
the mean absolute error is only a secondary measure. The quantiles are kept for the
probabilistic scores.

### Point metrics and skill

The point metrics are the RMSE, the mean absolute error (MAE) and the mean bias error
(MBE, positive for over-forecasts), each pooled over $\mathcal{H}_s$; for example,

$$
\mathrm{RMSE}_s = \sqrt{\frac{1}{|\mathcal{H}_s|} \sum_{t \in \mathcal{H}_s}
\left(\hat{P}_{s,t} - P_{s,t}\right)^2 } .
$$

Dividing by the station's nameplate capacity $C_s$ gives the normalised errors nRMSE and
nMAE, which make stations of different sizes comparable; errors are never divided by the
observed power, which is close to zero at dawn and dusk. The main measure is the RMSE
skill score over a reference,

$$
\mathrm{SS}_s = 1 - \frac{\mathrm{RMSE}_s^{\,\mathrm{model}}}{\mathrm{RMSE}_s^{\,\mathrm{ref}}},
$$ {#eq:skill}

computed per station on the same hours for model and reference: a positive score means
the model beats the reference, and $\mathrm{SS}_s = 0.2$ means a 20% lower RMSE. Skill is
reported over two references. Smart persistence gives the headline numbers, because
24-hour persistence of the clear-sky index is among the most common day-ahead references
[@yang2020verification], which keeps the scores comparable with other studies. CLIPER,
the training-window version of the reference that @yang2020verification recommend as a
standard, is the stricter second reference. The reported value is the mean of
$\mathrm{SS}_s$ over the six stations with its range across stations; the other metrics
are aggregated in the same way (@tbl:master-skill).

### Probabilistic metrics

The quantile forecasts are scored with the pinball loss
$\ell_\tau(y, q) = \max\{\tau\,(y - q),\ (\tau - 1)\,(y - q)\}$, whose expectation is
minimised by the true $\tau$-quantile [@gneiting2011making]. They are also scored with
the continuous ranked probability score (CRPS), a proper scoring rule for the whole
predictive distribution [@gneiting2007strictly]. With nine quantiles the CRPS is approximated as
twice the mean pinball loss,

$$
\mathrm{CRPS}_s \approx \frac{1}{|\mathcal{H}_s|} \sum_{t \in \mathcal{H}_s}
\frac{2}{9} \sum_{\tau \in \mathcal{T}} \ell_\tau\!\left(P_{s,t},\ \hat{q}_{\tau,s,t}\right),
$$ {#eq:crps}

a discrete version of $\mathrm{CRPS} = 2\int_0^1 \ell_\tau\,\mathrm{d}\tau$
[@gneiting2011comparing], divided by $C_s$ like the point errors. If all nine quantiles
equal the point forecast, @eq:crps reduces to the absolute error, because the levels
$\tau$ are symmetric about 0.5; the CRPS of a deterministic forecast is therefore its
MAE. This puts the references on the same scale and gives a CRPS skill score over Smart
persistence. Calibration is measured by the prediction interval coverage probability
(PICP) of the central 80% interval $[\hat{q}_{0.1,t}, \hat{q}_{0.9,t}]$, the share of
observations inside it (ideally 0.8), reported with the interval's mean width divided by
$C_s$. It is also measured by the share of observations below each $\hat{q}_{\tau,t}$
(ideally $\tau$), the data of a reliability diagram. Crossing quantiles are sorted before scoring
[@chernozhukov2010quantile]. The LSTM produces no quantiles and is left out of this
comparison (@tbl:probabilistic).

### Rescoring from saved forecasts

The runner saves every scored forecast, one file per run: the observation, the model's
raw output (native point forecast and quantiles), the Smart persistence forecast, the
template and the daytime flag. `experiments/rescore.py` recomputes all metrics from these
files, so a change of scoring rule never requires re-running a model: panel v1 was first
scored on the median (scoring version 1), the rules above are version 2, and every
results row records its version.

## Statistical testing {#sec:method-tests}

The six stations share one campus and one weather regime: their hourly absolute Smart
persistence errors, divided by capacity, have a mean pairwise correlation of 0.91.
Treating station-hours as independent would overstate the evidence, so the tests work
with daily losses. For a model $A$ and a test day $d$, the daily loss averages the
capacity-normalised squared error over the scored hours $\mathcal{H}_{s,d}$ of each
station-day, then over the six stations with equal weight (every station has scored
hours on every test day):

$$
L^{A}_{d} = \frac{1}{6} \sum_{s=1}^{6} \frac{1}{|\mathcal{H}_{s,d}|}
\sum_{t \in \mathcal{H}_{s,d}} \left( \frac{\hat{P}^{A}_{s,t} - P_{s,t}}{C_s} \right)^{2} .
$$ {#eq:daily-loss}

The Diebold--Mariano (DM) test [@diebold1995comparing] compares models $A$ and $B$
through the loss differential $D_d = L^{A}_{d} - L^{B}_{d}$ over the $n = 270$ test days,
under the null hypothesis that its expectation is zero. Consecutive days share weather,
so the variance of the mean $\bar{D}$ is estimated with the heteroskedasticity- and
autocorrelation-consistent (HAC) estimator of @newey1987simple. It uses Bartlett weights
and $m = \lfloor n^{1/3} \rfloor = 6$ lags, a rule of thumb that follows the $n^{1/3}$
growth rate that is optimal for the Bartlett kernel [@andrews1991heteroskedasticity]:

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
with $n - 1$ degrees of freedom. With $n = 270$ both corrections are small: the factor is
$\sqrt{(n-1)/n} \approx 0.998$, and the t distribution with 269 degrees of freedom is
close to the normal distribution. As a check that relies neither on the t approximation
nor on the choice of HAC lags, a second two-sided p-value comes from a moving-block
bootstrap of the differentials [@kunsch1989jackknife], with blocks of seven consecutive
days, which keep the short-range dependence, and 10,000 resamples.

The tests form one family of eleven comparisons, each tied to a research question. The
family was fixed after panel v1 had run, not before:

<!-- TODO(Lorenzo): say when and how the family of 11 was chosen, and state it plainly here. Record: an exploratory set of 10 day-level DM tests was run on the panel v1 forecasts on 2 Oct (scratchpad signif.py, 13:20) before COMPARISONS was written into experiments/report.py (uncommitted); 8 of those pairs are in the family, 2 were dropped (Chronos-2 vs LSTM with NWP; Chronos-2 vs LightGBM without weather) and 3 added (items 1, 10, 11). Also fix the report.py comment "fixed before the tests were run". -->

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

The p-values are adjusted with the Holm--Bonferroni procedure [@holm1979simple], which
keeps the probability of at least one false rejection in the family at or below 5%
(@tbl:significance).

## Compute measurement and environment {#sec:method-compute}

Compute cost is measured per fold: the training time is the wall-clock time of the
model's `fit()` call and the prediction time that of the fold's 30 daily forecasts, both
taken around the model alone. The reference forecast, computed afterwards, is not
charged to the model (`pvfc.runner.run_model`). For the zero-shot models, training time
is the loading of the checkpoint in the first fold; for the LSTM and Chronos-2
(fine-tuned) it covers only that fold's epochs (@sec:method-protocol).

Peak memory is the process's peak resident set size as reported by the operating system
(`peak_rss_mb` in `pvfc.utils`), a high-water mark over the life of the process; this is one
reason why each run has its own process. On this machine it leaves out the memory that
PyTorch allocates for the graphics processor (GPU) through Apple's Metal framework, even
though the GPU and the central processor (CPU) share one physical memory. In a check, a
1 GiB tensor on the GPU raised the peak by about 40 MB, the same tensor on the CPU by
about 1 GiB. For the five neural models, the LSTM and the four foundation-model variants,
the reported peak is therefore a lower bound (@tbl:compute).

All runs used one laptop, an Apple MacBook Air (13-inch, M4, 2025; model identifier
Mac16,12) with an M4 chip (10 CPU cores, 4 of them performance cores; 8 GPU cores), 16 GB
of unified memory shared by CPU and GPU, and macOS 26.6. It has no fan, so long runs may
be slowed by thermal throttling; the timings are not corrected for this. The five neural
models ran on the GPU through PyTorch's Metal Performance Shaders (MPS) backend, with
unsupported operations falling back to the CPU; the other models ran on the CPU. The runs
were executed one at a time, with the machine kept awake (`caffeinate`). The 120 runs of
panel v1 took 19.3 hours of wall-clock time, from 17:16 on 1 October to 12:33 on 2
October 2026 (`results/logs/panel_ledger.csv`).

The software was Python 3.12.13 with NumPy 2.4.6, pandas 2.3.3, scikit-learn 1.7.2,
SciPy 1.15.2, statsmodels 0.14.6, statsforecast 2.0.3, LightGBM 4.6.0, PyTorch 2.10.0,
PyTorch Lightning 2.5.2, darts 0.47.0, tirex-ts 1.4.2, Transformers 4.57.6 and pvlib
0.15.1, from conda-forge and PyPI (`environment.yml`, `requirements-lock.txt`).

## Reproducibility {#sec:method-repro}

Each run is identified by a configuration hash, the SHA-1 digest of the parts of its
configuration that define the run (data, folds, model and seed, as JSON with sorted keys)
truncated to 12 hexadecimal characters (`pvfc.utils.config_hash`). The hash keys the rows
of the results store and names the file of saved forecasts; re-running a configuration
replaces its rows, and any change to these parts yields a new hash. The evaluation
settings and the code are not hashed; a change of scoring rule is recorded by the scoring
version instead (@sec:method-metrics). Every row also records an environment
fingerprint: the Python version, the platform and the versions of the main libraries
(`pvfc.utils.env_fingerprint`). The checkpoints are downloaded once from Hugging Face
(`make checkpoints`); panel v1 used revisions `ddec013` of Chronos-2-small, `1d95242` of
TimesFM-2.5 (pinned by darts) and `63c7409` of TiRex.

A fixed seed (0) initialises Python, NumPy and PyTorch, and every darts model receives
the random state 0. Floating-point results on the GPU are not bit-identical to those on
the CPU, so the neural models are reproducible across devices only up to small numerical
differences, and with a single seed their sensitivity to initialisation was not measured.
Regression tests (`tests/`) check the protocol properties described in this chapter:
local-day folds, calendar anchoring, the training-only clear-sky template, the same-hour
fallback of the references, the removal of the target from the future frame, the
LightGBM lag wiring and the scoring rules.

<!-- TODO(Lorenzo): decide whether to run extra seeds for the LSTM and Chronos-2 (fine-tuned); if not, keep the single-seed caveat. Before any multi-seed run, pass the configuration's seed to the models: darts_wrapper takes random_state from the model params (default 0), and run.py passes only m["params"], so changing `seed` in experiment.yaml would not change the LSTM or fine-tuning initialisation. -->

`make env` and `make install` build the environment, `make checkpoints` fetches the
weights, `make panel` runs the experiment (resumably, skipping finished runs by their
hash), `make report` builds the tables and figures, and `make reproduce` chains the last
three. The raw data must first be downloaded by hand from Dryad (@sec:data-dataset).
