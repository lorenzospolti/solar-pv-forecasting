# Method {#sec:method}

<!-- TODO(Lorenzo): on 13 Oct 2026 Claude revised, after your review (~/Thesis/Thesis Review
2026-10-13 - Assessment ...md) and with the 12 Oct must-fixes: the opening sentence; 4.1 the double
dagger and the first run; 4.2 "covariates" and the two simplifications; 4.3 the quantile gloss; 4.4
the first paragraph; 4.5 which model choices drew on test results and the covariates of the two
references in Table 4.2; 4.8 the purpose and reading of the bootstrap intervals and their scope.
Rewrite in your own words before submission (UniPV AI guidelines). -->

This chapter describes how the forecasts were produced and scored; all models run
through the same code.

## Design overview and research questions {#sec:method-overview}

The experiment crosses four factors: the six stations of the panel (@sec:data-panel), two
weather settings, twelve models and nine test windows, called folds. Five models are
simple reference forecasts (@sec:background-concepts). Three are purpose-trained on each
station's history: a
seasonal autoregressive integrated moving-average model (SARIMA), the gradient-boosted
tree model LightGBM and a long short-term memory network (LSTM). Four are pretrained
time-series foundation models: Chronos-2, TimesFM and TiRex, used without any
training on the station ("zero-shot"), and Chronos-2 fine-tuned on each station. The
*NWP setting* adds archived numerical weather prediction (NWP) forecasts to the clear-sky
power $P^{\mathrm{cs}}_t$ (@sec:data-clearsky) and calendar features; the *no-weather
setting* does not. Every model
is scored on the same quantity, power, and on the same hours.

Each research question of @sec:intro-questions is a contrast within this design: RQ1
compares the zero-shot foundation models with the purpose-trained models and the
references, RQ2 fine-tuned with zero-shot Chronos-2, RQ3 the two weather settings, and
RQ4 the calibration and the cost of every model.

The results come from one complete run of this design, the *main run*: 144 runs, one per
station, weather setting and model, each over the nine folds. Thirteen further
configurations of Chronos-2, TimesFM and TiRex, which vary context length, checkpoint size
and seed or add a weather regression, were run on the same folds after the main results had
been seen; they are post hoc and marked with a double dagger (‡) in every table and
figure, as is *Chronos-sel*, which takes, for each fold, one of eight zero-shot Chronos-2
configurations selected on earlier days (@sec:method-selection). An earlier complete run on the same test days, the *first run*, is not reported, but
some choices were made after it had been seen: @tbl:decisions in Appendix
[-@sec:appendix-repro] dates every choice and states what had been seen when it was made.

## Forecasting task {#sec:method-task}

A forecast is issued once a day at local midnight, 00:00 Hong Kong Time (HKT, UTC+8, no
daylight saving time), and consists of the 24 hourly mean powers $\hat{P}_t$ of the local
day that begins then, labelled by the start of their interval (@sec:data-time) from 00:00
to 23:00. The daytime mask (@sec:data-clearsky) keeps only the hours labelled 06:00 to
18:00 in the test windows, so the scored hours lie 6 to 19 hours after the issue time.

At the issue time a model receives the *history*, every observation strictly before
local midnight (power $P_t$, clear-sky index $k_t$ and, for the models that take them,
covariates), and the *future
frame*, the 24 target timestamps with the covariates known in advance
(@sec:method-covariates). The runner deletes power and clear-sky index from the future
frame, so a model cannot read the values it must predict.

Two simplifications make the task easier than in operation. First, every measurement up to
midnight is available at the issue time, including the previous afternoon's, whereas an
operator who must submit a schedule earlier in the day forecasts further ahead without them
[@yang2020verification]. Second, the weather forecasts for the target day come from
weather-model runs that were not yet available at the issue time, most of them started eight
hours after it, so they are fresher than any forecast that could be used at midnight
(@sec:data-weather, @fig:data-leadtime); the NWP setting therefore gives an upper bound on
what weather forecasts can add.

## Forecasting in clear-sky-index space {#sec:method-target}

Every model is scored on power, but all except Persistence and Seasonal naive forecast
the clear-sky index $k_t$ and convert the forecast back to power with @eq:reconstruct.
With the clear-sky power $P^{\mathrm{cs}}_t$ of @sec:data-clearsky,

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
missing. The probabilistic models forecast not one value per hour but nine quantiles: the
$\tau$-quantile is the value that the outcome should fall below with probability $\tau$, for
$\tau = 0.1, 0.2, \dots, 0.9$ (@sec:method-metrics). A forecast $\hat{k}_t$ of the index, like
each of its quantiles, is converted back to power with

$$
\hat{P}_t = \max\left\{\hat{k}_t,\ 0\right\} P^{\mathrm{cs}}_t ,
$$ {#eq:reconstruct}

which is 0 at night, applies no upper clip and preserves the order of the quantiles.

The models built with the darts library [@herzen2022darts], all except the references,
need a series without gaps. A missing hour of the index (a gap in the data or an outage
hour, @sec:data-qc) therefore takes the value of the same hour on the most recent earlier day
that has one, at most seven days back, the rule of the references (@sec:method-models);
an hour still missing, on the first day of a series or in a longer gap, takes the
series' mean daytime index, or 0 outside the daytime mask. Filled hours are inputs
only, never scored.

## Covariates and weather settings {#sec:method-covariates}

All covariates are *future* covariates: their values for the 24 target hours are treated
as known at the issue time, so models may use them over the horizon as well as over the
context (@tbl:method-covariates). This is exactly true for the template and the calendar,
which follow from the date and the sun's position. For the weather forecasts it is the
optimistic assumption of @sec:method-task: in operation, the values for the target hours
would come from an older weather-model run.

: Covariates supplied for the context window and the 24 target hours, by weather
setting, with their names in the code. {#tbl:method-covariates}

| Covariate | Code name | Source | NWP | No weather |
|:--------------------------------------|:--------------------|:------------------------|:--------:|:-----------:|
| Clear-sky power $P^{\mathrm{cs}}_t$ (kW) | `p_clear` | Template (@sec:data-clearsky) | yes | yes |
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

@Tbl:method-models lists the covariates that each model receives. TimesFM and TiRex
receive none, because their darts wrappers do not accept covariates; their forecasts are
the same in both settings and are reported once, under the no-weather setting. Among the
foundation models, only Chronos-2 thus reads the weather forecasts natively. For TimesFM this
is a limit of the darts version used (Appendix [-@sec:appendix-repro]), which does not
expose the separate covariate regression of the model's own package;^[TimesFM repository,
README, update of 29 October 2025; <https://github.com/google-research/timesfm>, accessed 2
October 2026.] the TiRex version used here takes none. After the main run, this regression
("XReg") was re-implemented for both models (*TiRex-XReg* ‡, *TimesFM-XReg* ‡). Before
each forecast, a ridge regression (penalty 1, intercept not penalised) of the clear-sky
index on the nine covariates of the NWP setting and the NWP clear-sky index, the
forecast irradiance divided by $G^{\mathrm{cs}}_t$ (@sec:data-clearsky), is fitted on the
daytime hours of the last 8,192 hours; the foundation model, still univariate,
forecasts the residual, and the regression's forecast, 0 at night, is added to every
quantile. Both run in the NWP setting only: without weather the regression would read
only the template and the calendar. The NWP clear-sky index adds no data: it transforms
inputs of that setting. The regression's settings were fixed on development days before
the first test day.

## Models {#sec:method-models}

@tbl:method-models summarises the models and defines the short names used from here on.
The settings of the main-run models were fixed in one configuration file and were not
tuned. Three model choices drew on test results (@tbl:decisions): Climatology and CLIPER
were added after the first run, and the fine-tuning recipe, discussed below, and the lag
inputs of LightGBM, corrected for a coding error, were checked on a test fold.

### Reference forecasts

A reference forecast is a cheap rule that any useful model must beat
(@sec:background-concepts). *Smart persistence* repeats the clear-sky index of the same
hour on the previous day,
$\hat{P}_t = k_{t-24\,\mathrm{h}}\, P^{\mathrm{cs}}_t$, which corrects for the change
in solar geometry; with climatology, it is one of the two popular day-ahead references
named by @yang2020verification.^[They call it clear-sky persistence and reserve "smart
persistence" for persistence of a cloudiness index; the shorter name is kept here.] When
the lagged value is missing, the persistence rules step back one day at a time, up to
seven days, to the same hour (0 if none is found), never to the last value of the
history, which at midnight is a night-time zero. *Climatology* forecasts the mean
daytime clear-sky index of the training window, $\bar{k}$, at every hour: $\hat{P}_t =
\bar{k}\, P^{\mathrm{cs}}_t$.

*CLIPER* (CLImatology--PERsistence) combines the two
[@yang2019cliper]. When consecutive days are alike, yesterday is the better guide to
today; when they are not, the long-term mean is. CLIPER therefore weights the two by how
well yesterday has predicted today:

$$
\hat{k}_t = \rho\, k_{t-24\,\mathrm{h}} + (1-\rho)\, \bar{k}, \qquad
\hat{P}_t = \hat{k}_t\, P^{\mathrm{cs}}_t ,
$$ {#eq:cliper}

where the weight $\rho$ is the correlation between $k_t$ and $k_{t-24\,\mathrm{h}}$ over
the pairs of daytime hours in the training window, matched by timestamp and clipped to
$[0, 1]$; $\rho = 1$ gives Smart persistence and $\rho = 0$ Climatology. Night hours are
left out because their zeros would inflate $\rho$. For a stationary
index with variance $\sigma^2$, climatology has a mean squared error (MSE) of $\sigma^2$,
persistence $2(1-\rho)\,\sigma^2$ and the combination $(1-\rho^2)\,\sigma^2$, never more
than either [@yang2019cliper], so persistence is worse than climatology whenever
$\rho < 0.5$. In the training windows $\rho$ lies between 0.44 and 0.51: Smart
persistence is a weak bar. @yang2020verification strongly recommend skill (the relative
reduction in RMSE, @eq:skill) over this combination, with a single climatological mean as
here, for all solar forecasting
studies. They estimate the mean and $\rho$ on the verified sample; here both come from
the training window, so the guarantee holds only approximately.

Three further references were run and appear only in @tbl:master-skill-full: Climatology
on its own, *Persistence* of power, $\hat{P}_t = P_{t-24\,\mathrm{h}}$, which equals Smart
persistence within 0.001 skill at this horizon, and *Seasonal naive*, $\hat{P}_t =
P_{t-168\,\mathrm{h}}$.

<!-- TODO(Lorenzo): eq:cliper follows the code (src/pvfc/models/baselines.py): one mean for all hours, Yang's single-valued climatology. The protocol spec had one mean per hour of day, which the code docstring says moves CLIPER's mean skill by 0.0001 (panel v1). Confirm the single-valued form; otherwise change eq:cliper, tbl:method-models and the code. -->

### Purpose-trained models

*SARIMA* models the index as a linear regression on the covariates with seasonal
autoregressive integrated moving-average errors. Its orders are chosen by the stepwise
procedure of @hyndman2008automatic, in its statsforecast implementation: seasonal
period 24 hours, at most three non-seasonal and two seasonal autoregressive and
moving-average terms, selection by the corrected Akaike information criterion,
final maximum-likelihood fit. It is fitted once per fold
on the whole training window; at each issue time it is applied to the last 60 days of
history, updating its state but not its orders or coefficients.
Its quantiles come from its Gaussian prediction intervals.

*LightGBM* [@ke2017lightgbm] is a direct multi-horizon regressor: one gradient-boosted
tree ensemble (booster) per target hour and quantile level, 216 per fold, each with
LightGBM's default settings (100 trees of at most 31 leaves, learning rate 0.1) and the
pinball loss of its quantile (@sec:method-metrics) as objective. Each sees the index over the
previous local day and every covariate at each of the 24 target hours, and is trained on
every hourly origin of the training window. LightGBM is untuned by design, so its scores
probably understate what tuned gradient boosting can reach.

The *LSTM* [@hochreiter1997lstm], built with darts, has one layer of 32
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
covariates, so it can use covariates without task-specific training. It comes in a small
and a base checkpoint (28 M and 120 M parameters). The main run uses the small one,
*Chronos-S*, chosen because the base checkpoint might not fit in the laptop's memory, especially for fine-tuning, with a 336-hour context, zero-shot and
fine-tuned (*Chronos-S-FT*, below). *TimesFM* (version 2.5, September 2025)
[@das2024timesfm] is a decoder-only transformer with about 200 M parameters, and *TiRex*
[@auer2025tirex] a recurrent model with 35 M parameters based on xLSTM, an extended LSTM
[@beck2024xlstm]; both read a 512-hour context. All three output the nine quantiles
$\tau = 0.1, \dots, 0.9$ of the 24 target hours in one pass. These contexts are far
shorter than the models allow; they were fixed before the first run without testing
alternatives. After the main run had been seen, zero-shot Chronos-2 was also run with
longer contexts of 1,024 and 2,048 hours (suffixes *-mid* and *-long*), with the base
checkpoint (*Chronos-B*, *Chronos-B-long*) and TiRex with 2,048 hours (*TiRex-long*);
the longer contexts were given 90 days of history instead of 60
(@tbl:chronos2-configs). Later, the base checkpoint and TimesFM were run at their maximum
contexts, 8,192 hours (*Chronos-B-max* ‡) and all history up to 16,360 hours
(*TimesFM-max* ‡, no-weather setting only); TiRex-XReg ‡ and
TimesFM-XReg ‡ (@sec:method-covariates) use the contexts of TiRex-long ‡ and
TimesFM-max ‡. Two last runs complete a grid of both checkpoints at four contexts for the
selection of @sec:method-selection: the small checkpoint at 8,192 hours (*Chronos-S-max* ‡)
and the base checkpoint at 1,024 hours (*Chronos-B-mid* ‡). With weather forecasts,
Chronos-B-long ‡ is the most accurate zero-shot configuration on the test days, a choice
made on those days; *Chronos-sel* ‡ is the one selected without them. The comparisons that
involve these runs are post hoc (@sec:method-tests).

The dataset has been public since 2 April 2024 [@lin2024hkustdata], before any of the
checkpoints used here was released; no pretraining corpus named on their model cards
includes it, but an overlap cannot be excluded and would favour the zero-shot models.
Since April 2026 the dataset has also been used in a public foundation-model benchmark, FETS
[@obermeier2026fets], released after all the checkpoints.
Newer versions with covariate support,
TiRex-2 (July 2026) [@podest2026tirex2] and TimesFM 3.0 (August 2026),^[TimesFM
repository, README, "Update — August 2026"; <https://github.com/google-research/timesfm>,
accessed 2 October 2026.] are not wrapped by the darts version used here.

<!-- TODO(Lorenzo): confirm when and why the model set was frozen (date and reason), and consider TiRex-2 and TimesFM 3.0, both with covariates, as future work in the Discussion. -->

*Chronos-S-FT* starts in every fold from a fresh copy of the pretrained small
checkpoint and updates all its weights on the station's training window, with Adam at a
learning rate of $10^{-5}$, five epochs and batches of 32. Each training example is one
day, a 336-hour context ending at local midnight followed by the next day's 24 hours, so
a fold provides 60 to 150 optimiser steps. The loss is
the pinball loss summed over the checkpoint's 13 quantile levels, as darts implements
fine-tuning, with no validation split and no early stopping.

The official Chronos-2 quickstart also fine-tunes all weights at a learning rate of
$10^{-5}$ with batches of 32, but for 1,000 optimiser steps.^[chronos-forecasting
repository, notebook `chronos-2-quickstart.ipynb`, section "Fine-Tuning";
<https://github.com/amazon-science/chronos-forecasting>, accessed 4 October 2026. The
default rate of `Chronos2Pipeline.fit` (chronos-forecasting 2.2.2), which fine-tunes all
weights by default, is $10^{-6}$; AutoGluon 1.6.1 uses $10^{-5}$ for its default low-rank
adapters and recommends a lower rate, such as $10^{-6}$, for fine-tuning all weights
(<https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-model-zoo.html>, accessed 13
October 2026).] The recipe was chosen in a smoke test on one test fold of one station,
where it beat the zero-shot model and the default learning rate of darts, $10^{-3}$
(@tbl:decisions); no other learning rate was tried, including $10^{-6}$, the default of
chronos-forecasting's own fine-tuning routine. No other station-fold influenced the recipe, and excluding that fold
(@tbl:p2-robustness) or that station leaves the comparison with LightGBM unchanged.

: The models of the main tables and their short names. Chronos-S and Chronos-B are the
small and base checkpoints of Chronos-2; *-long* marks a 2,048-hour context and *-FT* fine-tuning;
Chronos-sel is, in each fold, one of eight zero-shot configurations selected on earlier days
(@sec:method-selection); the other configurations are in @tbl:chronos2-configs. ‡ Post hoc: added
after the main run had been seen. *Size*: parameters, counted from the checkpoints for the foundation models
(TimesFM uses 203 M of the 231 M stored); the LSTM has 5,153 without weather inputs; SARIMA has at
most ten autoregressive and moving-average terms plus regressor coefficients. *Context*: input
window at the issue time. *All*: every covariate of the setting. *Training per fold*: what is
estimated in each fold, from scratch. *Licence*: of the pretrained weights. Persistence, Seasonal
naive and Climatology are defined in the text. {#tbl:method-models}

| Model | Family | Size | Context | Covariates | Output | Training per fold | Licence |
|:-------------------|:--------------|:-----------|:---------|:--------------|:------------|:--------------|:----------------|
| Smart persistence | Reference | -- | $k_{t-24\,\mathrm{h}}$ | None | Point | -- | -- |
| CLIPER | Reference | 2 | $k_{t-24\,\mathrm{h}}$ | None | Point | $\bar{k}$ and $\rho$ | -- |
| SARIMA | Statistical | ≤ 10 terms | 60 days | All | Quantiles | Order search and fit | -- |
| LightGBM | Boosted trees | 216 × 100 trees | 24 h | All | Quantiles | 216 boosters | -- |
| LSTM | Recurrent network | 5,665 | 168 h | All | Point | 30 epochs | -- |
| Chronos-S | Foundation (encoder) | 28 M | 336 h | All | Quantiles | None | Apache-2.0 |
| Chronos-sel ‡ | Foundation (encoder) | 28 or 120 M | 336–8,192 h | All | Quantiles | None | Apache-2.0 |
| Chronos-B-long ‡ | Foundation (encoder) | 120 M | 2,048 h | All | Quantiles | None | Apache-2.0 |
| Chronos-S-FT | Foundation (encoder) | 28 M | 336 h | All | Quantiles | 5 epochs | Apache-2.0 |
| TimesFM | Foundation (decoder) | 203 M | 512 h | None | Quantiles | None | Apache-2.0 |
| TiRex | Foundation (xLSTM) | 35 M | 512 h | None | Quantiles | None | NXAI Community |

## Evaluation protocol {#sec:method-protocol}

### Walk-forward folds

The models are tested walk-forward on whole local days, a form of
rolling-origin evaluation with an expanding training window [@tashman2000out]. The folds
are anchored to the calendar: the first test day is 5 July 2022 for every station,
whatever the start of its record, so all stations are scored on the same days. Nine
30-day test windows start 60 days apart, each preceded by a one-day *embargo* that
belongs neither to the training nor to the test window. Every training window starts at
the beginning of the station's record and holds at least 365 days, the fit period of the
template (@sec:data-clearsky). The folds (@tbl:method-folds) contain 270 test days per
station, about 3,000 of their hours in the daytime mask; they cover all four seasons,
unevenly (four windows in autumn, one in winter), and February and April have no test
day.

: Test windows of the nine folds (local dates, HKT) and training days, which vary
across stations; meteorological seasons (three-month blocks, winter being December to
February), by the majority of a window's days.
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
forecasts of the test window without being re-fitted (the XReg regression excepted,
@sec:method-covariates). Its inputs at each issue time run
up to local midnight, including the embargo day and the earlier test days: the embargo
separates the last training target from the first test day but withholds no input data,
as for a forecaster that is re-trained now and then but run every day.

Every fit starts from scratch, so no fold inherits anything estimated in an earlier one:
the references, SARIMA and LightGBM are estimated anew, the LSTM starts from the same
seeded initialisation and Chronos-S-FT from a fresh copy of the pretrained
checkpoint. The zero-shot foundation models are never trained; their checkpoint is loaded
in the first fold and kept.

### Comparability

One splitter produces the folds for all models, the runner stops if a model's
predictions miss any hour of a test window, and the reporting script excludes any run
whose scored hours differ from those of the station's other runs, so every model is
scored on exactly the same timestamps. Each run has its own operating-system process,
which isolates crashes and memory measurements.

### Selecting the zero-shot configuration {#sec:method-selection}

Choosing the best of several configurations by their skill on the test days overstates
the skill of the one chosen [@cawley2010overfitting]; settings should be chosen in advance,
on days that the test never touches [@lago2021forecasting]. The longer contexts and the
base checkpoint were tried only after the main run had been seen, so a zero-shot
configuration of Chronos-2 is also selected in real time, once per fold, from a grid of
eight: the small and the base checkpoint, each with 336, 1,024, 2,048 and 8,192 hours of
context. The 30 days before each test window hold the fold's embargo day and, before it, a
29-day *inner window* (5 June to 3 July 2022 for the first fold); from the second fold on
they lie between two test windows. Every candidate and CLIPER forecast the inner windows
exactly as test windows. Before each fold, the candidates are scored on all earlier inner
and test windows, 29 days for the first fold and 501 for the last, all of them past at the
fold's first issue time, as in recursive model selection [@pesaran1995predictability]; a
fold's own test window never enters its choice. The score is the skill over CLIPER of
@sec:method-metrics on those days. A candidate counts as tied with the best if their skills
differ by at most one bootstrap standard error of their paired difference (seven-day
blocks over the scored days, as in @sec:method-tests), and the first tied candidate is taken
in a fixed order: the small checkpoint before the base one, then the shorter context. The
order leans away from the configurations that did best on the test days. The forecasts of
each fold's choice form the row *Chronos-sel* ‡, separately in each weather setting. Its
weights are never trained on a station; only its inference settings are chosen, on earlier
days. The rule was recorded before any inner-window forecast was made (@tbl:decisions), but
the candidates were chosen after the main run, so the row stays post hoc and untested.

With weather forecasts, Chronos-B-long ‡ is the most accurate zero-shot configuration on the
test days, but it was picked on them, which flatters its score. Three estimates bracket that
optimism; none is exact. The first, after @tibshirani2009bias, averages over the nine test
windows the best window skill among the eight candidates minus the window skill of the
configuration best over all test days; it is about twice too large when the candidates are
close, and it also counts real differences between windows. The second,
$\mathrm{sd}(D)/\sqrt{2\pi}$, is the expected excess of the better of two equally skilful
configurations whose skill difference $D$ over the test days has the bootstrap standard
deviation $\mathrm{sd}(D)$; it applies to a choice between two. The third is the mean excess
of the winner over its own skill across the paired bootstrap resamples of the test days.

## Metrics {#sec:method-metrics}

### Scored hours and point forecasts

The headline scores use the daytime mask of @sec:data-clearsky, the hours whose apparent
solar zenith angle at mid-hour is below 85°, as @yang2020verification recommend. Within
the mask they use every hour with a valid power value after quality control
(@sec:data-qc); outage hours and other gaps are missing data and never scored. Outside
the mask lie the night, where all models forecast zero or almost zero, and low-sun hours
that hold under 0.3% of the energy measured in the test windows; skill scores over all
hours agree with the daytime ones to four decimals and are not reported.
$\mathcal{H}_s$ denotes the scored hours of
station $s$, pooled over the nine test windows.

<!-- TODO(Lorenzo): ETL v4 (like v3) flags outage hours only where P_cs >= 0.1 C, so zero readings at the low-sun edges of outage days and feed gaps escape the rule and are scored: 30 hours in the test windows (SQ1 9, all at 06:00 on 8-16 May 2023; CYT (Station 2) 11 and SQ15 10, mostly the 17:00 hour after their 07:00-17:00 feed gaps), 0.17% of the 17,919 scored hours. Appendix B (single events) now states this, with its own TODO; decide there whether to extend the outage rule to whole outage days in a later ETL. Evidence: scratchpad ch4/v4_stats.py. -->

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
minimised by the median, is secondary for them. The mean rule was adopted after the first
run had been examined (@tbl:decisions). Under the median rule the ranking in the NWP
setting is unchanged, and in the no-weather setting every model except SARIMA scores
slightly below CLIPER (@tbl:point-rule).

All metrics are computed from the saved forecasts, which keep each model's raw output, so
a change of scoring rule needs no re-run.

### Point metrics and skill

The point metrics are the root mean squared error (RMSE), the mean absolute error (MAE)
and the mean bias error (MBE, positive for over-forecasts), each pooled over
$\mathcal{H}_s$; for the RMSE,

$$
\mathrm{RMSE}_s = \sqrt{\frac{1}{|\mathcal{H}_s|} \sum_{t \in \mathcal{H}_s}
\left(\hat{P}_{s,t} - P_{s,t}\right)^2 } .
$$

Divided by the station's nameplate capacity $C_s$, never by the observed power, which is
close to zero at dawn and dusk, they give the normalised errors nRMSE and nMAE,
comparable across station sizes, though only approximately, because $C_s$ is not the same
kind of rating at every station (@sec:data-dataset). Raw errors depend on the station and
on the weather of the test period, so the main measure relates a model's error to that of
a reference forecast; its value depends on the reference, and the weaker the reference,
the higher the score. It is the RMSE skill score over a reference,

$$
\mathrm{SS}_s = 1 - \frac{\mathrm{RMSE}_s^{\,\mathrm{model}}}{\mathrm{RMSE}_s^{\,\mathrm{ref}}},
$$ {#eq:skill}

computed per station on the same hours for model and reference: a positive score means
that the model beats the reference, and $\mathrm{SS}_s = 0.2$ a 20% lower RMSE; a
negative score means that the model does worse than the reference, and
$\mathrm{SS}_s = -0.1$ a 10% higher RMSE. CLIPER, the training-window version of the
reference recommended by
@yang2020verification, gives the headline numbers. Smart persistence, among the most
common day-ahead references [@yang2020verification], appears as a row of the main
tables, and skill scores over it are in @tbl:master-skill-full. CLIPER became the headline
reference after the results of the main run had been seen (@tbl:decisions); the ranking of
the models of the main run is the same under either reference. The reported value is the mean of
$\mathrm{SS}_s$ over the six stations with a 95% interval (@sec:method-tests); the other
metrics are averaged over the stations in the same way (@tbl:master-skill).

### Probabilistic metrics

The quantile forecasts are scored with the pinball loss
$\ell_\tau(y, q) = \max\{\tau\,(y - q),\ (\tau - 1)\,(y - q)\}$, whose expectation is
minimised by the true $\tau$-quantile [@gneiting2011making]. Its mean over the nine
levels and the scored hours is the *quantile score* (QS), in kW, reported divided by
capacity as QS / $C_s$ (@tbl:probabilistic):

$$
\mathrm{QS}_s = \frac{1}{|\mathcal{H}_s|} \sum_{t \in \mathcal{H}_s}
\frac{1}{9} \sum_{\tau \in \mathcal{T}} \ell_\tau\!\left(P_{s,t},\ \hat{q}_{\tau,s,t}\right) .
$$ {#eq:qs}

Twice the QS equals the weighted interval score of the four central intervals and the
median [@bracher2021evaluating] and is a discrete version of the continuous ranked
probability score, $\mathrm{CRPS} = 2\int_0^1 \ell_\tau\,\mathrm{d}\tau$
[@gneiting2011comparing], a proper scoring rule for the whole predictive distribution
[@gneiting2007strictly]; for a calibrated Gaussian forecast twice the QS lies about 9%
above the exact CRPS. For a point forecast, twice the QS reduces to the absolute error, because the
levels are symmetric about 0.5. Calibration is measured by the prediction interval
coverage probability (PICP) of the central 80% interval $[\hat{q}_{0.1,t},
\hat{q}_{0.9,t}]$, the share of observations inside it (ideally 0.8), reported with the
interval's mean width divided by $C_s$ (@tbl:probabilistic), and by the share of
observations at or below each $\hat{q}_{\tau,t}$ (ideally $\tau$), shown as a reliability
diagram (@fig:reliability). Crossing quantiles are sorted before scoring
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
taken in date order, under the null hypothesis that its expectation is zero. Because
consecutive days share weather, the variance of the mean differential is estimated with
the autocorrelation-consistent estimator of @newey1987simple (six lags), with the
small-sample correction of @harvey1997testing and a p-value from Student's t
distribution; a moving-block bootstrap gives a second p-value. The details are in
@sec:appendix-results-tests.

A score measured on 270 particular days would come out differently on other days; a
bootstrap estimates by how much, without new data, by recomputing the score on many
resamples of the test days. Every skill score and skill difference over the 270 test days
in the Results, and the PICP-80 of @tbl:probabilistic, carries a 95% interval, not adjusted
for multiplicity, from a moving-block bootstrap over the test days [@kunsch1989jackknife].
Blocks of seven consecutive test days, in date order, are drawn with replacement, so that
neighbouring days, which share weather, stay together, to form each of 10,000 resamples of
the 270 days (seed 0), and the same resampled days are used for every station and model, so
differences between models are paired. In each resample the skill is recomputed per station
on the hours of the resampled days and averaged over the stations; the interval runs from
the 2.5th to the 97.5th percentile of the resampled values. An interval of a difference that
contains zero means that the two models cannot be told apart on these test days, not that
they are equal. The stations are not resampled, and the range across stations, which is not a
measure of uncertainty, is given only in @tbl:master-skill-full.

The tests form one family of eleven comparisons, each tied to a research question:

1. LightGBM vs Chronos-S, NWP setting (RQ1);
2. LightGBM vs Chronos-S-FT, NWP setting (RQ1, RQ2);
3. Chronos-S-FT vs SARIMA, NWP setting (RQ1);
4. Chronos-S-FT vs Chronos-S, NWP setting (RQ2);
5. Chronos-S-FT vs Chronos-S, no-weather setting (RQ2);
6. SARIMA vs Chronos-S, no-weather setting (RQ1);
7. Chronos-S vs TiRex, no-weather setting (RQ1);
8. TiRex vs TimesFM, no-weather setting (RQ1);
9. LightGBM, NWP vs no-weather setting (RQ3);
10. Chronos-S, NWP vs no-weather setting (RQ3);
11. CLIPER vs Smart persistence (choice of reference).

The family was chosen after an exploratory look at day-level tests on the forecasts of
the first run (eight of the ten pairs tested there kept, items 1, 10 and 11 added) and
fixed before the main run (@tbl:decisions). The first run covered the same test days, so
the family was fixed before the main run but after the first run had been examined. The p-values are adjusted with the Holm--Bonferroni procedure
[@holm1979simple], which keeps the probability of at least one false rejection in the
family at or below 5% (@tbl:significance).

Comparisons that involve the post hoc configurations are themselves post hoc: E1–E3
were chosen after their results had been seen, E4 and E5 before Chronos-B-long ‡ was run,
E6–E15 before the XReg and maximum-context configurations were run, and V1–V4 with the
selection rule, before any inner-window forecast was made (@tbl:decisions). They are
reported as paired skill differences $A - B$ over CLIPER with 95% intervals, without tests
(@tbl:significance, second block), in the NWP setting unless stated:

- **E1.** Chronos-S-long ‡ vs Chronos-S;
- **E2.** LightGBM vs Chronos-S-long ‡;
- **E3.** Chronos-S-FT vs Chronos-S-long ‡;
- **E4.** LightGBM vs Chronos-B-long ‡;
- **E5.** Chronos-S-FT vs Chronos-B-long ‡;
- **E6–E15.** the XReg configurations, TimesFM-max ‡ and Chronos-B-max ‡ each vs
  Chronos-B-long ‡ and LightGBM; each XReg configuration vs its univariate counterpart;
- **V1.** LightGBM vs Chronos-sel ‡;
- **V2.** Chronos-S-FT vs Chronos-sel ‡;
- **V3.** Chronos-sel ‡ vs Chronos-B-long ‡: what choosing without the test days costs or
  gains against the choice made on them, which is not the same as the optimism of that
  choice (@sec:method-selection);
- **V4.** Chronos-sel ‡ vs Chronos-S.

RQ1 is answered twice. Chronos-S, fixed before the main run, is the configuration that the
tests of the family examine. Chronos-sel ‡ is the best-practice zero-shot result: its
configuration is chosen without the test window it is scored on, but among candidates drawn
up after the main run. Chronos-B-long ‡, the most accurate on the test days, is reported
beside them as a choice made on those days.

## Compute measurement {#sec:method-compute}

Compute cost is measured per fold as wall-clock time around the model alone, without the
reference forecast. The training time is that of the model's `fit()` call, which for the
LSTM and Chronos-S-FT includes building the model, for the latter reloading
the pretrained checkpoint; the zero-shot foundation models load their checkpoint only in
the first fold, in a few seconds, so their median training time is close to zero. The
prediction time covers the fold's 30 daily forecasts. @Tbl:compute reports the median
times over the 54 station-folds of each model and setting.

Peak memory is the process's peak resident set size reported by the operating system,
a high-water mark over the life of the process;
@tbl:compute reports its maximum over the station-folds. It leaves
out the memory that PyTorch allocates for the graphics processing unit (GPU) through Apple's
Metal framework, although the GPU and the central processing unit (CPU) share one physical
memory (@sec:appendix-repro), so for the neural models (the LSTM and the foundation
models) the reported peak is a lower bound.

All runs used one fanless laptop, an Apple MacBook Air with an M4 chip and 16 GB of
memory shared by CPU and GPU; timings are not corrected for thermal throttling. The
neural models ran on the GPU through PyTorch, with a fallback to the CPU for unsupported
operations, the others on the CPU. The 144 runs of the main run were executed one at a
time and took 18.5 hours (Appendix [-@sec:appendix-repro]).

## Reproducibility {#sec:method-repro}

Every run is identified by a hash of its configuration: the data, including the version
of the data processing, the folds, the model and the seed. The hash keys the rows of the
results store and names the file of saved forecasts, so runs on differently processed
data never mix. All random number generators are seeded with 0 by default, so the LSTM
starts from the same weights in every fold; Chronos-S-FT was run with three seeds
(@tbl:chronos2-configs), the LSTM with one. Regression tests check the protocol
properties of this chapter. Appendix [-@sec:appendix-repro] gives the full checklist,
the dated record of choices (@tbl:decisions) and where each step is implemented
(@tbl:code-map).
