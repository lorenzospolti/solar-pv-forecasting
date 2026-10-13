# Data {#sec:data}

<!-- TODO(Lorenzo): on 13 Oct 2026 Claude revised, after your review (~/Thesis/Thesis Review
2026-10-13 - Assessment ...md) and with the 12 Oct must-fixes: 3.1 the nameplate paragraph (AC and DC
ratings explained) and "(panels)"; 3.2 "panel" defined, the count, criteria (i) and (iii) worded,
the size criterion deleted (Zone A2 now fails criterion (iii)); the Table 3.1 caption; 3.3 the
power and energy columns in one sentence; 3.4 the meter-error rules and the outage paragraphs; 3.5
the ratio PR, the reason for the 1.3 ceiling, "slow"; 3.6 Open-Meteo, why the 12 UTC run is
unusable, the upper-bound sentence. Rewrite in your own words before submission (UniPV AI
guidelines). -->

This chapter describes the dataset and the six stations used ([@sec:data-dataset; @sec:data-panel]),
how the raw files become hourly series ([@sec:data-time; @sec:data-qc]), the clear-sky power and
clear-sky index that most models forecast (@sec:data-clearsky) and the weather forecasts used as
inputs (@sec:data-weather). The evidence behind each data decision is collected in Appendix
[-@sec:appendix-data].

## The HKUST rooftop PV dataset {#sec:data-dataset}

The data come from the open dataset of @lin2025hkust, released on Dryad under a CC0 dedication
[@lin2024hkustdata]: three years (2021–2023) of measurements from 60 grid-connected rooftop
photovoltaic (PV) stations on the campus of the Hong Kong University of Science and Technology
(HKUST). Each station feeds the grid through one or more inverters, which convert the direct
current (DC) of its PV modules into alternating current (AC). Thirty-seven stations report
every 15 minutes and the other 23 hourly; each file holds a time stamp, the energy of the interval
and the mean power. The authors marked missing values but, on purpose, did no gap filling or
outlier detection, so the quality control of @sec:data-qc is part of this work. A weather tower on
the campus also records irradiance, the power of sunlight per square metre [@lin2025hkust]; it is
not used as an input (@sec:data-weather).

The metadata give each station's rated power, called here its nameplate capacity $C$ (kW). $C$
normalises the error metrics (@sec:method-metrics), caps the template, the station's clear-sky
power curve (@sec:data-clearsky), and scales the quality-control thresholds. It is not the same
kind of rating at every station. The modules produce direct current, and the inverters that convert
it have their own maximum output. At four of the six stations used here, $C$ is the AC rating of
the inverters; their modules are rated 21–24% higher, so on clear days the output stops at this
limit and reaches $C$. At SQ567 and SQ15, $C$ is the total DC rating of the modules, which the
output never reaches: it peaks at 0.80–0.85 $C$ (Appendix [-@sec:appendix-data]). Skill scores,
which compare two errors at the same station, do not depend on $C$, but errors divided by $C$
compare stations only approximately.

The time stamps carry no time zone. Read as Hong Kong Time (HKT, UTC+8, no daylight saving), they
put the peak of mean power around solar noon, as they should (@tbl:data-checks).

## Station panel {#sec:data-panel}

The benchmark runs on six of the 60 stations, called the *panel*, because one run of every model
on six stations takes about 18.5 hours on the laptop and all 60 would take about eight days
(@sec:method-compute). The six were chosen from a count of the missing daytime hours in all 60
files, with four criteria:

(i) at least a year of data before the first test day, 5 July 2022, because the template is
fitted on the first year of each record (@sec:data-clearsky);
(ii) a complete or near-complete daytime record;
(iii) stable output, that is, a clear-sky index at its ceiling of 1.3 in fewer than 5% of daytime
hours in every year (@sec:data-clearsky);
(iv) both reporting intervals: four 15-minute and two hourly stations.

The panel takes four 15-minute stations of different sizes and buildings, and the two hourly
stations that meet (i) with the most complete records (@tbl:data-panel; the choice within the
criteria is in Appendix [-@sec:appendix-data]). The count and criteria (ii) and (iii) used each station's whole
record, test period included; they measure the completeness and stability of the data, not the
accuracy of any forecast.

: The six panel stations after processing. First hour: first interval-start label (HKT), one hour
before the first raw stamp at the hourly stations (@sec:data-time); all records end on 31 December
2023. Daytime hours: sun more than 5° above the horizon at mid-hour (@sec:data-clearsky). Missing:
daytime hours without a power value (outage hours included). Outage: daytime hours of near-zero
output that the outage rule set missing (@sec:data-qc). $k_t>1$: daytime hours with power above the
template, the clear-sky power, $P_t > P^{\mathrm{cs}}_t$ (@sec:data-clearsky). {#tbl:data-panel}

| Station                    |     $C$ (kW) | Step       | First hour (HKT)            | Daytime hours | Missing (%) |  Outage (h) | $k_t>1$ (%) |
|:---------------------------|-------------:|:-----------|:----------------------------|--------------:|------------:|------------:|------------:|
| LSK North                  |        55.00 | 15 min     | 2021-06-01 00:00            |        10,717 |        0.00 |           0 |        6.29 |
| SQ1                        |        27.60 | 15 min     | 2021-06-01 00:00            |        10,717 |        0.98 |         105 |        7.12 |
| SQ567                      |        47.97 | 15 min     | 2021-07-01 00:00            |        10,327 |        0.00 |           0 |        7.60 |
| UG Hall6                   |        25.00 | 15 min     | 2021-06-01 00:00            |        10,717 |        0.00 |           0 |        7.90 |
| CYT (Station 2)            |        25.00 | 1 h        | 2021-05-22 23:00            |        10,834 |        2.70 |          17 |        7.56 |
| SQ15                       |        33.29 | 1 h        | 2021-03-23 23:00            |        11,535 |        4.07 |          73 |       11.72 |

Zone A2 (249.5 kW) fails criterion (iii): its output changes regime in May 2023, when its grid
connection is dated (Appendix [-@sec:appendix-data]). Zone A2 was the only larger station with a
year of data before the first test day, so the effect of station size cannot be tested: the panel
represents stations of 25–55 kW, around the dataset's median capacity (32 kW), and, through
criteria (ii) and (iii), well-kept ones, so the results may be optimistic for the rest of the
dataset.

## Time stamps and resolution {#sec:data-time}

Every hourly value $P_t$ is the mean power over the hour that starts at its label: the value labelled
12:00 covers 12:00 to 13:00. All series are stored in UTC, and the clear-sky quantities of
@sec:data-clearsky are computed for the same intervals.

Mean power comes from the power column of each file, not from the energy column, which in the
15-minute files reads about 3% above the inverters' own records and drops to zero in many dim
quarter hours (@tbl:data-energy).

The hourly files label the end of the hour, a convention that the dataset paper does not state.
Correlating each hourly file with the 15-minute stations at a range of time shifts shows it
(@tbl:data-labels), so every hourly file was moved back by one hour. The same test puts CYT
(Station 1) about half an hour off, which no whole-hour shift can repair; this is why it was
excluded.

## Quality control {#sec:data-qc}

Quality control was deliberately light, because each rule changes the target on which every model
is scored.

Impossible readings were removed: raw values above 1.1 $C$, more power than any station can
deliver (on the panel, legitimate readings reach at most 1.005 $C$), or below −0.05 $C$, before
hourly averaging, so that one bad reading cannot contaminate an hourly mean; and hourly values above
1% of $C$ while the sun is more than 5° below the horizon. These limits apply to power; the ceiling
of 1.3 in @sec:data-clearsky applies to the clear-sky index, a ratio. On the panel these rules
removed three values (Appendix [-@sec:appendix-data]).

Gaps stay missing, because a missing hour is not an hour without generation: its clear-sky index is
missing too, and it is not scored (@sec:method-metrics). Only gaps of at most two hours with a
measured hour on both sides were bridged, by linear interpolation; on the panel this filled 18
hours.

Some stations report exactly zero for hours or days while the sun is up: SQ1, for example, from 7
to 16 May 2023. Such runs are outages or shutdowns of the equipment, which the files record as
measured zeros rather than as missing values. Left in, they would teach the models that a sunny day
can produce nothing, and they would count every forecast of the sunshine that did fall as an error.
Cloud dims a PV array but seldom stops it: even under heavy overcast, the diffuse light that passes
through keeps it producing. An *outage hour* is therefore a daytime hour that meets three
conditions: the station reports at most 0.1% of $C$; a cloudless sky would give at least 10% of $C$
($P^{\mathrm{cs}}_t \ge 0.1\,C$, @sec:data-clearsky), so the sun is well up; and the hour lies in
a run of at least three consecutive such hours, so that a single zero reading is not enough. Outage
hours are set to missing, like any other gap. Because the rule needs the template, the template was
fitted once, the outage hours were set to missing, and the template was fitted again without them.

The rule flagged hours at three stations (@tbl:data-panel; events in @tbl:data-outages). Eighteen
of them belong to three very dark storms in 2023, which these stations read as almost zero: they are
weather, not outages, and the other three stations keep these hours as measured. A few zero readings
at the low-sun edges of outages and of gaps in the data escape the rule and are scored (Appendix
[-@sec:appendix-data]). The missing hours of SQ1 and of the hourly stations fall mostly on days with
outage hours and on days whose daytime values are missing from the file altogether
(@fig:data-coverage).

![Share of daytime hours without a power value, per panel station and local day. Each bar spans the
station's record; triangles mark days with outage hours set missing. The nine 30-day test windows
(top) start on 5 July 2022.](figures/data_coverage.pdf){#fig:data-coverage width=100%}

Before any model runs, each station is checked against a list of invariants, from the time zone of
the index to the stability of the template (@tbl:data-checks), and the batch stops if one fails.
All six stations passed.

## Clear-sky power and clear-sky index {#sec:data-clearsky}

Most models forecast the clear-sky index $k_t = P_t / P^{\mathrm{cs}}_t$ rather than power
(@sec:background-concepts), so the clear-sky power $P^{\mathrm{cs}}_t$, the template, sets the scale
of every forecast. A physical model of each array was impractical: the metadata list flat arrays,
arrays split between two directions and, for SQ15, curved roofs [@lin2025hkust], and shading is not
recorded. The template was therefore learned from each station's own clearest hours.

It combines two parts. The clear-sky irradiance $G^{\mathrm{cs}}_t$ (W/m²), the same for every
station, comes from the Ineichen–Perez model of a cloudless atmosphere [@ineichen2002linke] in the
PV modelling library pvlib [@holmgren2018pvlib; @anderson2023pvlib]. A ratio
$\widehat{\mathrm{PR}}$, specific to each station, converts this irradiance into a fraction of the
station's capacity: it is the capacity factor $P_t/C$ divided by the clear-sky irradiance in units
of $G_0 = 1{,}000$ W/m², that is, the share of its capacity that the station delivers for every
1,000 W/m² of clear-sky irradiance. The ratio was fitted on the first year of each record: the
observed ratios were binned by the sun's position (the solar declination $\delta_t$ and the hour
angle $\omega_t$), each bin took its 95th percentile, so that the ratio follows the clearest hours,
and the grid was smoothed (Appendix [-@sec:appendix-data]). As a function of the sun's position, the ratio absorbs each
array's tilt, orientation, efficiency and fixed shading. The template is

$$
P^{\mathrm{cs}}_t = \min\left\{ C,\; \widehat{\mathrm{PR}}(\delta_t, \omega_t)\, C\, \frac{G^{\mathrm{cs}}_t}{G_0} \right\},
$$ {#eq:data-pcs}

which is zero when the sun is below the horizon for the whole hour and never reaches the cap $C$ on
the panel.

Fitted before the first test day and applied unchanged to the whole record, the template carries no
test-period information, but it does not follow slow changes such as soiling (dirt on the modules)
or degradation (the loss of module output with age).

The scored hours are the daytime hours, those in which the sun stands more than 5° above the
horizon at mid-hour [@reda2004spa], about 11 per day (@sec:method-metrics).

The clear-sky index is $k_t = P_t / P^{\mathrm{cs}}_t$, clipped to $[0, 1.3]$ (@eq:k); it is zero
when the template is zero and missing where $P_t$ is missing. The ceiling bounds the ratio near
sunrise and sunset, when the template is small and a small difference in power would give a very
large ratio. It applies only to the index from which the models learn: every score uses the
measured power. @Fig:data-clearsky shows one week at LSK North.

![Measured power and template $P^{\mathrm{cs}}_t$ at LSK North (55 kW), 5–11 July 2022, the first
week of the first test window (top), and the daytime clear-sky index (bottom). Power stays far below the
template on the overcast first days and follows it on the clear days, where $k_t$ is close to
1.](figures/data_clearsky.pdf){#fig:data-clearsky width=100%}

Values above 1 come from cloud enhancement, when broken clouds briefly push irradiance above its
clear-sky value [@gueymard2017cloud], and from clear hours above a template set at the 95th
percentile (@tbl:data-panel, @fig:data-k). The ceiling of 1.3 lies above all but 0.3–0.5% of
daytime values and binds mostly in the first or last daytime hour of a day (Appendix
[-@sec:appendix-data]). It also serves as the stability check of criterion (iii): it detects a rise in output after the fit year, as at Zone A2, but not a loss,
such as new shading.

![Distribution of the daytime clear-sky index at each panel station over its whole record, as a
share of all daytime hours, in bins of 0.05. The vertical line marks $k_t = 1$; the orange bar is
the share clipped at 1.3.](figures/data_k.pdf){#fig:data-k width=100%}

## Weather covariates {#sec:data-weather}

In the NWP setting the models also receive archived numerical weather prediction (NWP) forecasts,
the output of physical models of the atmosphere, for the campus: irradiance, temperature, wind speed
and humidity (@tbl:data-weather). They come from the Historical Forecast archive of Open-Meteo, an open
weather-data service [@openmeteo], whose series for this location match the high-resolution forecast of the Integrated
Forecasting System (IFS) of the European Centre for Medium-Range Weather Forecasts (ECMWF)
(Appendix [-@sec:appendix-data]). Open-Meteo stamps a radiation value at the end of the hour it
averages [@openmeteo_forecast_docs], so the radiation columns were moved back by one hour to match
the labels of the power.

The archive is not a set of day-ahead forecasts (@fig:data-leadtime). Open-Meteo builds it by
stitching together the first hours of successive model runs, the forecasts that the weather model
starts at fixed times of day [@openmeteo]; for this location the series jumps at 00 and 12 UTC, as
if stitched from two runs a day. Most scored hours therefore come from a run that starts eight hours
after the issue time, at lead times (the time from the start of a run to the hour forecast) of about
0–12 hours. A forecast issued at midnight could use at most the run started the afternoon before, because a
run reaches users several hours after it starts [@ecmwf_dissemination]: even the 12 UTC run (20:00
HKT) is not yet available at midnight, and the 06 UTC run reaches the scored hours at lead times of
16–28 hours. Open-Meteo's archive of forecasts at fixed
lead times starts mostly in 2024 [@openmeteo_previous_runs], too late for this test period.

![The weather forecasts behind one forecast day (schematic). The forecast is issued at 00:00 HKT and
scored on the daytime hours (shaded). The archive used in this thesis fills these hours with the
first hours of the 12 and 00 UTC runs, mostly from a run that starts eight hours after the issue
time; at the issue time, the newest run available is the 06 UTC run of the day
before.](figures/data_leadtime.pdf){#fig:data-leadtime width=100%}

Because these runs are fresher than any that a forecast issued at midnight could use, the NWP
setting gives an upper bound on what weather forecasts add, and the no-weather setting a lower
bound; together they approximately bracket a real day-ahead forecast (@sec:threats).

Measured weather from the campus tower is not used as an input, because its irradiance record is
inconsistent over time while the PV output follows the archive (Appendix [-@sec:appendix-data]).

## Summary {#sec:data-summary}

Each panel station ends up as one hourly series of power $P_t$, template $P^{\mathrm{cs}}_t$,
clear-sky index $k_t$ and a daytime flag, plus the weather forecasts in the NWP setting. All six are
scored on the same 270 test days, from July 2022 to November 2023, after at least a year of training
data (@sec:method-protocol). The data cover one subtropical campus over three years, and the
stations were selected for complete records and stable output, so the results describe well-kept
rooftop systems in one climate.
