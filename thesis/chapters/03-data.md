# Data {#sec:data}

This chapter describes the dataset and the six stations used ([@sec:data-dataset; @sec:data-panel]),
the processing of the raw files into hourly series ([@sec:data-time; @sec:data-qc]), the clear-sky
template behind the forecasting target (@sec:data-clearsky) and the weather covariates
(@sec:data-weather).

## The HKUST rooftop PV dataset {#sec:data-dataset}

The data come from the open dataset of @lin2025hkust, released on Dryad under a CC0 dedication
[@lin2024hkustdata]: three years (2021–2023) of measurements from 60 grid-connected rooftop
photovoltaic (PV) stations on the campus of the Hong Kong University of Science and Technology
(HKUST) in Sai Kung (22.34° N, 114.26° E), with 2,230.8 kW of nameplate capacity in total. Of these,
23 stations have a single inverter and report hourly; the other 37 have panel-level optimizers,
DC–DC converters that keep each pair of modules at its maximum power point, and report every 15
minutes, with inverter-level data every 5 minutes. Each station-level file holds a time stamp, the
energy of the interval (kWh) and the power (W). The authors aligned all streams to regular stamps
and marked missing values but, on purpose, did no gap filling or outlier detection. A weather tower
on the campus records global horizontal irradiance (GHI) and other variables every minute
[@lin2025hkust].

The metadata, a Brick model [@balaji2016brick], give each station's rated power, grid-connection
date, tilt and azimuth. The nameplate capacities $C$ (kW) read from it
(`pvfc.data.load_nameplate_capacities`) sum to the paper's total; $C$ normalises the error metrics
(@sec:method-metrics), caps the clear-sky template and scales the quality-control thresholds. The
time stamps carry no time zone and were read as Hong Kong Time (HKT, UTC+8, no daylight saving),
which puts the peak of mean power in the hour starting at 12:00 (11:00 at SQ1), around solar noon
(12:07–12:37 HKT).

## Station panel {#sec:data-panel}

The first full run of the benchmark took 19.3 hours for six stations on the laptop; all 60 stations
would have needed about eight days, so the benchmark runs on a panel of six stations, chosen from a
coverage sweep of the 60 station-level files (daytime: hours labelled 07:00 to 17:00 local). The
median share of missing daytime hours is 0.2% at the 15-minute stations and 4.3% at the hourly ones,
with a worst case of 23.2%; ten stations, all with optimizers, have no missing daytime hour and at
least two years of data.
<!-- TODO(Lorenzo): the 28 Sep sweep left only its output (station_gap_report.csv, in
~/Thesis/thesis-review-2026-09-28/), not its script. A re-implementation that reproduces every
number of this paragraph and the next is in the scratchpad (revise-data/coverage_counts.py, which
reads the power column like ETL v4); add it to the repository (e.g.
experiments/coverage_sweep.py, output under results/) so that the selection can be reproduced. -->

The selection criteria, recorded in `configs/experiment.yaml`, were (i) at least a year of data
before the first test day, 5 July 2022, which is also the fit window of the clear-sky template; (ii)
a complete or near-complete daytime record; (iii) stable output, that is, a clear-sky index at its
ceiling in fewer than 5% of daytime hours in every year (@sec:data-clearsky); (iv) a nameplate
capacity of 25–55 kW; and (v) both groups, as four 15-minute and two hourly stations. Nine 15-minute
stations meet (i) and (ii) with at most one missing daytime hour. The panel takes the largest of
them (LSK North, 55 kW), a second size class (SQ567, 48 kW), one of four similar staff-quarter
towers (SQ1) and UG Hall6, whose only missing hour in the sweep was bridged (@sec:data-qc). CYT
(Station 2) and SQ15 are the hourly stations that meet (i) with the smallest shares of missing
daytime hours in the sweep (2.7% and 3.8%). @Tbl:data-panel lists the six stations.
<!-- TODO(Lorenzo): confirm the within-criteria choices: SQ1 rather than its 33.3 kW siblings
SQ2-SQ4, and UG Hall7 and UG Hall4 not taken (they peak at 77-79% of nameplate). -->

: The six panel stations after processing (`data/processed/qc_log.csv`). First hour: first
interval-start label (HKT), one hour before the first raw stamp at the hourly stations
(@sec:data-time); all records end on 31 December 2023. Daytime hours: sun more than 5° above the
horizon at mid-hour (@sec:data-clearsky). Missing: daytime hours without a power value. Outage:
hours set missing by the outage rule (@sec:data-qc). {#tbl:data-panel}

| Station                    |     $C$ (kW) | Step       | First hour (HKT)            | Daytime hours | Missing (%) |  Outage (h) | $k_t>1$ (%) |
|:---------------------------|-------------:|:-----------|:----------------------------|--------------:|------------:|------------:|------------:|
| LSK North                  |        55.00 | 15 min     | 2021-06-01 00:00            |        10,717 |        0.00 |           0 |        6.29 |
| SQ1                        |        27.60 | 15 min     | 2021-06-01 00:00            |        10,717 |        0.98 |         105 |        7.12 |
| SQ567                      |        47.97 | 15 min     | 2021-07-01 00:00            |        10,327 |        0.00 |           0 |        7.60 |
| UG Hall6                   |        25.00 | 15 min     | 2021-06-01 00:00            |        10,717 |        0.00 |           0 |        7.90 |
| CYT (Station 2)            |        25.00 | 1 h        | 2021-05-22 23:00            |        10,834 |        2.70 |          17 |        7.56 |
| SQ15                       |        33.29 | 1 h        | 2021-03-23 23:00            |        11,535 |        4.07 |          73 |       11.72 |

Two candidates were excluded on evidence (`pvfc.data.EXCLUDED_STATIONS`). CYT (Station 1) ranks 59th
of 60 for coverage (8.8% of daytime hours missing), and its stamps are offset by about half an hour
(@sec:data-time). Zone A2 (249.5 kW), the only larger station with a year of data before the first
test day, changes regime in May 2023: the daily maximum of its hourly power rises from below 0.27
$C$ before 18 May to 0.66–0.73 $C$ on every day from 27 May to 4 June, and the Brick model dates its
grid connection to 31 May 2023. With a template fitted on its first year, its clear-sky index sits
at the 1.3 ceiling in 43% of the measured daytime hours of 2023, against at most 0.62% for any panel
station in any year, so UG Hall6 took its place. Without Zone A2 the effect of station size cannot
be tested: the panel represents stations of 25–55 kW, around the dataset's median capacity (32 kW),
and, through criteria (ii) and (iii), well-kept ones, so the results may be optimistic for the rest
of the dataset.

## Time stamps and resolution {#sec:data-time}

Every processed frame is indexed in UTC (`pvfc.data._read_ts_csv`) and uses interval-start labels:
an hourly value labelled HH:00 is the mean over the hour from HH:00 to HH+1:00, and the clear-sky
quantities of @sec:data-clearsky were computed for the same intervals.

Mean power was taken from the power column, in W (`pvfc.data._read_power`); in the hourly files it
equals the energy column, in kWh per hour, to within rounding. In the 15-minute files the two
columns differ by about 3%, and the power column is the better record: it integrates to within 0.3%
of the AC power that the inverters log every five minutes, and the dataset paper's validation
example, 642.09 kWh at the Library on 31 August 2022 against 641.55 kWh from its inverters
[@lin2025hkust], is its integral. The energy column lies 3.1–3.3% above the inverters and reads zero
in 2–23% of the quarter hours in which the power column reports 20 W to 1 kW, although it resolves 1
Wh. Both columns follow the tower's irradiance over the quarter hour after their stamp more closely
than over the neighbouring ones (@tbl:data-energy).

The hourly files label the end of the hour, a convention that the dataset paper does not state. Each
hourly file was correlated in daytime with the mean capacity factor ($P_t/C$) of the four 15-minute
panel stations over the window $[t+s,\ t+s+60\ \mathrm{min})$, where $t$ is the raw stamp:
interval-start labels would peak at $s = 0$, end-of-hour labels at $s = -60$ minutes. CYT (Station
2) peaks at $s = -60$ minutes (0.985, against 0.837 at $s = 0$), SQ15 at $-75$ and $-60$ minutes
(0.973 and 0.969, against 0.790; @tbl:data-labels). Every hourly file was therefore moved back by
one hour (`pvfc.data._read_power`). CYT (Station 1) peaks between −15 and −45 minutes (0.91–0.92),
so no whole-hour relabelling can align it.

## Quality control {#sec:data-qc}

Quality control was deliberately light, because each rule changes the target on which every model is
scored.

Meter errors were removed at native resolution, so that one bad reading cannot contaminate an hourly
mean (`pvfc.data._qc_power`). Readings above 1.1 $C$ were set to missing; none occur on the panel,
where legitimate readings reach 1.005 $C$. Readings below −0.05 $C$ were set to missing too: SQ15
holds two, of −111 $C$ and −148 $C$, in the hours from 09:00 on 3 April and from 14:00 on 29 July
2022 (smaller ones would become zero; there are none). Finally, an hourly value above 1% of $C$ in
an hour whose midpoint has the sun more than 5° below the horizon was set to missing: 8.9 kW at SQ15
at 02:00 on 29 August 2022, the only such value on the panel.

Gaps stay missing, because a missing hour is not an hour without generation; its clear-sky index is
missing too, and it is not scored (@sec:method-metrics). Only gaps of at most two hours with a
measured hour on both sides were bridged by linear interpolation (`pvfc.data._bridge_short_gaps`):
17 hours at SQ15, including the three values removed above, and one hour at UG Hall6; five of them
are daytime hours in test windows and are scored. At the hourly stations the longer gaps are mostly
days on which the feed holds no value from 07:00 to 17:00 (28 days at CYT (Station 2), 37 at SQ15).

Some stations report exactly zero for hours or days while the sun is up. Since cloud dims a PV array
but seldom stops it, an outage hour was defined as a daytime hour in a run of at least three
consecutive hours with power at most 0.1% of $C$ while the clear-sky power $P^{\mathrm{cs}}_t$
(@sec:data-clearsky) is at least 10% of $C$ (`pvfc.qc`); outage hours were set to missing before the
clear-sky template was fitted again. The rule flagged 105 hours at SQ1, 17 at CYT (Station 2), 73 at
SQ15 and none elsewhere (@tbl:data-outages), including outages of SQ1 from 7 to 16 May 2023 and of
SQ15 on 9–10 November 2022, both inside test windows. Three very dark storm spells in 2023 (7 May,
17 June and 8 September; tower irradiance about 30–50 W/m²) cut output to about 0.3% of $C$ or less;
they stay as measured at LSK North, SQ567 and UG Hall6, and the rule removes them only where a
station read at most 0.1% of $C$ for three hours, 18 hours at SQ1 and the hourly stations. ETL v3
had also removed them at the other 15-minute stations, because the energy column it read is zero in
many such quarter hours. Zero readings at the edges of gaps and outages, where the sun is low or the
run short, escape the rule: 113 daytime hours at three stations, 30 of them in test windows.
<!-- TODO(Lorenzo): edge zeros remain in ETL v4 (unchanged from v3). Daytime hours that read
exactly 0 next to a missing hour or on an outage day: SQ1 9 (all 9 in test windows, 06:00 on 8-16
May 2023), CYT (Station 2) 42 (11), SQ15 62 (10); 30 in test windows = 0.17% of the 17,919 scored
hours (same count as the TODO in sec:method-metrics). Mostly the 17:00 hour after the 07:00-17:00
feed gaps, where P_cs reaches 26% of C at CYT (Station 2). They are scored as observations. Decide:
leave them (this sentence) or extend the outage rule to whole outage days in a later ETL. Evidence:
scratchpad ch3/v4_edge_defs.py, ch3/v4_edge_zeros_list.py. -->

The missing hours of SQ1 and the hourly stations fall mostly on outage days, here days with at least
half of the daytime hours missing: 9 at SQ1, 29 at CYT (Station 2) and 46 at SQ15, of which 9, 10
and 13 lie in test windows (@tbl:data-panel, @fig:data-coverage).

![Share of daytime hours without a power value, per panel station and local day. Each bar spans the
station's record; triangles mark days with outage hours set missing. The nine 30-day test windows
(top) start on 5 July 2022.](figures/data_coverage.pdf){#fig:data-coverage width=100%}

These rules define the fourth version of the processing (ETL v4). ETL v3 fixed four defects found in
an audit of the second version, the basis of an earlier full run (panel v1): negative meter errors
kept as zeros, zeros interpolated into long gaps, an unwrapped hour angle and a weather cache that
began after SQ15's first 993 hours. ETL v4 returned from the energy column, adopted in v3, to the
power column. A first panel v2 run on ETL v3 was stopped after 96 of 144 runs, for this change and
for a fix to how the models fill gaps in their inputs (@sec:method-target), and repeated on ETL v4,
the run reported here; the data version is part of every run's hashed configuration, so results on
different data cannot be mixed.

Before any model runs, `experiments/check_data.py` checks each station against 17 invariants
(@tbl:data-checks in Appendix [-@sec:appendix-data]), and the overnight batch stops if one fails
(`experiments/run_panel.py`). All six stations passed.

## Clear-sky template and clear-sky index {#sec:data-clearsky}

Most models forecast the clear-sky index, the measured power divided by a clear-sky power template
$P^{\mathrm{cs}}_t$, which removes the known daily and seasonal path of the sun
(@sec:method-target).

A physical model of each array was impractical: the metadata list flat arrays, arrays split between
two directions and, for SQ15, a "Mixed" orientation, the label the dataset uses for curved roofs
[@lin2025hkust], and shading is not recorded. The template was therefore learned from the data
through a performance ratio (PR), the capacity factor $P_t / C$ divided by the horizontal irradiance
in units of $G_0 = 1{,}000$ W/m². As a function of the sun's position, the PR absorbs an array's
tilt, orientation, efficiency and fixed shading. It was built in four steps
(`pvfc.clearsky.empirical_clearsky_power`):

1. The clear-sky GHI $G^{\mathrm{cs}}_t$ comes from the Ineichen–Perez model
   [@ineichen2002linke] in pvlib [@holmgren2018pvlib; @anderson2023pvlib] for one campus
   coordinate (22.337° N, 114.263° E, 60 m), averaged over twelve instants in each hour, with the
   Linke turbidity (the dimming of clear-sky sunlight by water vapour and aerosols) from pvlib's
   monthly climatology [@remund2003linke].
2. The sun's position at the interval midpoint is expressed by the solar declination $\delta_t$,
   the latitude at which the sun stands overhead at noon, and the hour angle $\omega_t$, the sun's
   angular distance from solar noon, which grows by 15° per hour, is negative in the morning and
   is kept between −180° and 180°.
3. The ratio $\mathrm{PR}_t = (P_t / C) / (G^{\mathrm{cs}}_t / G_0)$ is collected over the first 365
   days of each record (ending between 23 March and 30 June 2022), for hours with measured power and
   $G^{\mathrm{cs}}_t \ge 50$ W/m², and binned on a grid of 4° of declination by 15° of hour angle
   (12 × 16 cells). Each cell with at least eight samples takes the 95th percentile of the ratio, so
   that the template follows the clearest hours. The other cells are filled from their neighbours,
   and the grid is smoothed with a 3 × 3 mean and interpolated bilinearly, giving
   $\widehat{\mathrm{PR}}(\delta_t, \omega_t)$.
4. The template is

   $$
   P^{\mathrm{cs}}_t = \min\left\{ C,\; \widehat{\mathrm{PR}}(\delta_t, \omega_t)\, C\, \frac{G^{\mathrm{cs}}_t}{G_0} \right\},
   $$ {#eq:data-pcs}

   which is zero when the sun is below the horizon for the whole hour. Its peak lies between
   0.75 $C$ (SQ15) and 0.99 $C$ (SQ1), so the cap at $C$ never binds on the panel.

Fitted before the first test day and applied unchanged to the whole record, the template carries no
test-period information, but it does not follow slow changes such as soiling or degradation.

Binning by sun position rather than by calendar month lets a month missing from the fit year borrow
from dates with the same declination on the other side of the solstice; an earlier, month-binned
version was zero throughout February at CYT (Station 1).

The daytime mask keeps the hours in which the sun stands more than 5° above the horizon at mid-hour,
that is, whose apparent solar zenith at the interval midpoint, computed with pvlib's implementation
of the solar position algorithm of @reda2004spa, is below 85°; it defines the scored hours
(@sec:method-metrics), about 11 per day.

The clear-sky index is $k_t = P_t / P^{\mathrm{cs}}_t$, clipped to $[0, 1.3]$; it is zero when the
template is zero ($P^{\mathrm{cs}}_t \le 10^{-6}$ kW) and missing where $P_t$ is missing (@eq:k).
@Fig:data-clearsky shows one week at LSK North. In the twilight hours just outside the daytime mask,
$k_t$ is erratic, at the ceiling in 16–26% of them; they are not scored but remain in the models'
inputs.

![Measured power and clear-sky template at LSK North (55 kW), 5–11 July 2022, the first week of the
first test window (top), and the daytime clear-sky index (bottom). Power stays far below the
template on the overcast first days and follows it on the clear days, where $k_t$ is close to
1.](figures/data_clearsky.pdf){#fig:data-clearsky width=100%}

Daytime values above 1 come from cloud enhancement, when broken clouds briefly push irradiance above
its clear-sky value [@gueymard2017cloud], and from clear hours above a template set at the 95th
percentile. They occur in 6.3–11.7% of daytime hours (@tbl:data-panel, @fig:data-k), most often at
SQ15. The ceiling binds in only 0.3–0.5% of daytime hours, 76–96% of them the first or last daytime
hour of a day, when $P^{\mathrm{cs}}_t$ is small. It also serves as the stability check of criterion
(iii), which detects an increase in output after the fit year, as at Zone A2, but not a loss, such
as new shading.

![Distribution of the daytime clear-sky index at each panel station over its whole record, as a
share of all daytime hours, in bins of 0.05. The vertical line marks $k_t = 1$; the orange bar is
the share clipped at 1.3.](figures/data_k.pdf){#fig:data-k width=100%}

## Weather covariates {#sec:data-weather}

In the no-weather setting the models see only the clear-sky template and calendar features
(@sec:method-covariates). The NWP setting adds archived numerical weather prediction (NWP) forecasts
from the Historical Forecast service of Open-Meteo [@openmeteo], fetched for the campus coordinate
(`pvfc.weather`; @tbl:data-weather). The request leaves the choice of weather model to Open-Meteo
("best match"); on six sample weeks between March 2021 and November 2023 the cached series equal,
value for value, those of the high-resolution forecast (HRES) of the Integrated Forecasting System
(IFS) of the European Centre for Medium-Range Weather Forecasts (checked on 2 October 2026),
archived on a 9 km grid [@openmeteo] whose cell for the campus lies about 7 km away
[@openmeteo_forecast_docs].

Open-Meteo stamps a radiation value at the end of the hour it averages [@openmeteo_forecast_docs],
so the radiation columns were moved back by one hour, which raises the daytime correlation between
forecast GHI and the mean capacity factor of the four 15-minute stations from 0.73 to 0.85;
instantaneous variables keep their stamp. The cache holds complete calendar years, so every station
has weather values from its first hour.

The archive is not a set of day-ahead forecasts: Open-Meteo stitches the first hours of successive
model runs, and IFS starts a run every six hours [@openmeteo], so each archived value comes from a
run at most a few hours old, whereas a forecast issued at local midnight would see the midday hours
(11:00–14:00) at lead times of 11–14 hours or more. The NWP setting is therefore optimistic, by an
amount that cannot be measured for the panel period: Open-Meteo's Previous Runs archive, which keeps
forecasts at fixed lead times, starts mostly in January 2024 [@openmeteo_previous_runs] and held no
one-day-ahead irradiance for July 2022 (checked on 2 October 2026). The two settings thus
approximately bracket what a real day-ahead forecast can add, the NWP setting from above and the
no-weather setting from below.

The campus tower could have supplied an "oracle" setting with measured weather, but its irradiance
record is inconsistent: its mean midday GHI lies 29% below the archive in 2022 and 8% above it in
2023, with hourly means far above the clear-sky maximum, while the PV output follows the archive
(@fig:data-weather in Appendix [-@sec:appendix-data]). The oracle setting is therefore kept for
diagnostics only.

## Summary {#sec:data-summary}

Each panel station ends up as one hourly UTC series of $P_t$, $P^{\mathrm{cs}}_t$, $k_t$ and a
daytime flag, plus the forecast covariates in the NWP setting, spanning 21,936 hours (SQ567) to
24,300 hours (SQ15). All stations are scored on the same nine test windows of 30 local days, from 5
July 2022 to 26 November 2023: 270 test days and 3,041 daytime hours per station, after a first
training window of at least 365 days (368 at SQ567; @sec:method-protocol). The data cover one
subtropical campus over three years [@lin2025hkust], and the stations were selected for complete
records and stable output.
