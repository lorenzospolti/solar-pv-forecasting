# Background and Related Work {#sec:background}

<!-- TODO(Lorenzo): on 13 Oct 2026 Claude revised in Section 2.1, after your review
(~/Thesis/Thesis Review 2026-10-13 - Assessment ...md): the template sentence (a physical clear-sky
model combined with a ratio learned from the clearest hours, and what "template" means), "overcast",
"most of" the cycle and "(panels)". Rewrite in your own words before submission (UniPV AI
guidelines). -->

*Scope note: (a) probabilistic scoring in solar forecasting [yang2019ropes;
lauret2017probabilistic]; (b) purpose-trained models — SARIMA, gradient boosting
[ke2017lightgbm], LSTM [hochreiter1997lstm]; (c) time-series foundation models
[ansari2024chronos; ansari2025chronos2; das2024timesfm; woo2024moirai;
ekambaram2024ttm] and the zero-shot/fine-tuning paradigm.*

## Concepts used in this thesis {#sec:background-concepts}

This section explains the concepts from solar energy and forecast verification on which
the later chapters rely; how they are computed for this dataset is described in Chapters
[-@sec:data] and [-@sec:method].

The power of a photovoltaic (PV) station has a predictable part and an unpredictable
part. The predictable part follows the sun's path across the sky, which astronomy gives
for any future hour: a daily cycle, from zero at night to a peak near solar noon, when
the sun is highest, and a seasonal cycle, as the sun's height at noon changes over the
year. The unpredictable part comes mainly from clouds, which can cut the output to a
small fraction of what the sun's path allows. The power in hour $t$ can therefore be
written as a product [@bacher2009online],

$$
P_t = k_t\, P^{\mathrm{cs}}_t ,
$$

in which $P^{\mathrm{cs}}_t$ carries the sun's path and $k_t$ the effect of the weather.

The *clear-sky power* $P^{\mathrm{cs}}_t$ (kW) is the power that a given station would
deliver in hour $t$ under a cloudless sky. It combines the clear-sky irradiance, the
power of sunlight per square metre (W/m²) that would reach the ground without clouds,
with the characteristics of the station's array of PV modules: its size, tilt,
orientation and shading. It can be simulated with a physical model of the array
[@engerer2014kpv] or learned statistically from the measured power alone
[@bacher2009online]. This thesis combines the two (@sec:data-clearsky): a physical model
of the cloudless atmosphere gives the clear-sky irradiance, and a ratio learned from each
station's clearest hours turns it into power. The result is called the *template* for
short: the station's cloudless-day power curve, which the weather then scales down.

The *clear-sky index* is the share of its clear-sky power that a station delivers,
$k_t = P_t / P^{\mathrm{cs}}_t$. It is about 1 in a clear hour and low under a fully
clouded (overcast) sky, and it can exceed 1 under *cloud enhancement*, when broken clouds briefly push
irradiance above its clear-sky value (@sec:data-clearsky). Solar papers also define a
clear-sky index for irradiance, written $\kappa$, and $k_t$ usually denotes another
quantity, the *clearness index*, the ratio of the irradiance at the ground to that at
the top of the atmosphere [@yang2020verification]; in this thesis $k_t$ always denotes
the clear-sky index of power. Most models here forecast $k_t$ rather than power:
dividing by $P^{\mathrm{cs}}_t$ removes most of the known daily and seasonal cycle and leaves a
series that is closer to stationary, which classical time-series models assume
[@bacher2009online]. The forecast index is multiplied back by $P^{\mathrm{cs}}_t$, and
every model is scored on power (@sec:method-target).

A *reference forecast* is a cheap rule that any useful model must beat. The two most
common in solar forecasting are *persistence*, which assumes that tomorrow will be like
today, and *climatology*, which forecasts a long-term average [@yang2020verification]. A
model's error alone says little, because raw errors depend on the site and on the
weather of the test period: the variability of sunlight governs how predictable it is,
so the same model looks better on a run of clear days than on changeable weather. A
reference scored on the same hours faces the same difficulty, and the *skill score*, or
*skill* for short, measures a model against it,
$\mathrm{SS} = 1 - \mathrm{RMSE}_{\mathrm{model}} / \mathrm{RMSE}_{\mathrm{ref}}$
(@eq:skill), which allows, with caveats, comparisons across sites and periods
[@yang2020verification]. A
score of 0 means that the model is no better than the reference, 0.2 that its RMSE is
20% lower, and a negative score that the model is worse than the reference. The value
depends on the reference: the same forecasts score higher against a weak reference than
against a strong one, so skill scores can be compared only when they share a reference.

Each standard reference has a weakness: persistence does well only when consecutive days
are alike, and climatology ignores the latest weather. Their optimal blend weights
yesterday's value by $\rho$, the correlation between the values on consecutive days,
which measures how well yesterday has predicted today, and the long-term average by
$1 - \rho$; in theory it is never worse than either [@murphy1992climatology;
@yang2019cliper]. This blend, called *CLIPER* (CLImatology–PERsistence),^[The name is
older than its use in solar forecasting. The US National Hurricane Center describes
CLIPER5, a statistical model of tropical cyclone tracks based on climatology and
persistence, as "originally developed in 1972" and "now used primarily as a benchmark
for evaluating the forecast skill of other models"; National Hurricane Center,
*Technical Summary of the National Hurricane Center Track and Intensity Models*, July
2009; <https://www.nhc.noaa.gov/pdf/model_summary_20090724.pdf>, accessed 4 October
2026.] is the reference that @yang2020verification recommend for all solar forecasting
studies. CLIPER is the headline reference of this thesis and Smart persistence,
persistence of the clear-sky index, the secondary one (@sec:method-models).
