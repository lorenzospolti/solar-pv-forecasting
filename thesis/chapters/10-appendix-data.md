# Data quality checks {#sec:appendix-data}

This appendix collects the evidence behind the data decisions of Chapter [-@sec:data]: the
nameplate ratings, the choice of the panel and the excluded stations, the pre-flight checks run
before every batch (@tbl:data-checks), the hours removed by the outage rule (@tbl:data-outages)
and single events, the power and energy columns of the 15-minute files (@tbl:data-energy), the
label convention of the hourly files (@tbl:data-labels), the construction of the template, the
weather covariates (@tbl:data-weather) and the campus tower (@fig:data-weather). All numbers refer
to the data of the final processing (Appendix [-@sec:appendix-repro]).

## Nameplate capacity

The metadata are a Brick model [@balaji2016brick], a machine-readable description of building
equipment: a text file of subject–property–value statements that link each station to its inverter
and module types and their ratings and give its rated power, grid-connection date, tilt and azimuth
(the compass direction that the modules face). The pipeline reads only the rated power, the
nameplate capacity $C$; the 60 values sum to the paper's total of 2,230.8 kW. At LSK North, SQ1, UG
Hall6 and CYT (Station 2), $C$ is the AC rating of the inverter, which caps the output: their
highest hourly power is 0.99–1.00 $C$, although their modules total 1.21–1.24 $C$ in DC. At SQ567
and SQ15, $C$ is the modules' total DC rating, and their power peaks at 0.85 $C$ and 0.80 $C$.
Across all 60 stations, $C$ equals the modules' DC total (to within 0.02 kW) in 26, the inverters'
AC rating in 28 and neither in 6.

## Station selection

The coverage sweep counted, in each of the 60 station files, the daytime hours (labelled 07:00 to
17:00 local) without a power value. The median share of missing daytime hours is 0.2% at the
15-minute stations and 4.3% at the hourly ones, with a worst case of 23.2%; ten stations, all
reporting every 15 minutes, have no missing daytime hour and at least two years of data. Nine
15-minute stations meet criteria (i) and (ii) of @sec:data-panel with at most one missing daytime
hour. The panel takes the largest of them (LSK North, 55 kW), a second size class (SQ567, 48 kW),
one of four similar staff-quarter towers (SQ1) and UG Hall6, whose only missing hour in the sweep was
bridged (@sec:data-qc). CYT (Station 2) and SQ15 are the hourly stations that meet (i) with the
smallest shares of missing daytime hours in the sweep (2.7% and 3.8%).
<!-- TODO(Lorenzo): the 28 Sep sweep left only its output (station_gap_report.csv, in
~/Thesis/thesis-review-2026-09-28/), not its script. A re-implementation that reproduces every
number of this paragraph is in the scratchpad of 5 Oct (revise-data/coverage_counts.py, which
reads the power column like ETL v4); add it to the repository (e.g.
experiments/coverage_sweep.py, output under results/) so that the selection can be reproduced. -->
<!-- TODO(Lorenzo): confirm the within-criteria choices: SQ1 rather than its 33.3 kW siblings
SQ2-SQ4, and UG Hall7 and UG Hall4 not taken (they peak at 77-79% of nameplate). Evidence (Brick
model; raw 15-minute site files, power column): their C is the modules' DC total (UG Hall4
32.76 kW = 84 x 390 W; UG Hall7 31.98 kW = 82 x 390 W), but each has a single 25 kW SolarEdge
SE25K inverter, and their 15-minute power caps at 25.25 kW = 0.771 C and 25.15 kW = 0.787 C. The
77-79% is therefore the inverter limit, not a fault; SQ567 and SQ15, also rated in DC, peak at
0.85 C and 0.80 C. -->

Two candidates were excluded. CYT (Station 1) ranks 59th of 60 for coverage (8.8% of daytime hours
missing), and its stamps are offset by about half an hour (@tbl:data-labels). Zone A2 (249.5 kW),
the only larger station with a year of data before the first test day, changes regime in May 2023:
the daily maximum of its hourly power rises from below 0.27 $C$ before 18 May to 0.66–0.73 $C$ on
every day from 27 May to 4 June, and the Brick model dates its grid connection to 31 May 2023. With
a template fitted on its first year, its clear-sky index sits at the 1.3 ceiling in 43% of the
measured daytime hours of 2023, against at most 0.62% for any panel station in any year, so UG Hall6
took its place.

## Pre-flight checks

: Pre-flight checks of `experiments/check_data.py` on the six panel stations
(`results/logs/check_data__*.log`). Night: outside the daytime mask. Measured: with a power value.
Template: the clear-sky power $P^{\mathrm{cs}}_t$ (@sec:data-clearsky). The last row is
informational. {#tbl:data-checks}

| Check                        | Criterion                                                                | Panel result                         |
|:-----------------------------|:-------------------------------------------------------------------------|:-------------------------------------|
| Time index                   | time-zone-aware UTC                                                      | pass at all six                      |
| Peak hour                    | mean power peaks in the 11:00, 12:00 or 13:00 interval (local)           | 12:00 at five stations, 11:00 at SQ1 |
| Night power                  | mean power at night below 1% of $C$                                      | at most 0.07% of $C$                 |
| Daytime coverage             | less than 20% of daytime hours missing                                   | 0.00–4.07%                           |
| Quality bound                | maximum hourly power at most 1.1 $C$                                     | 0.80–1.00 $C$                        |
| Finite target                | $P^{\mathrm{cs}}_t$ finite in daytime; $k_t$ finite where $P_t$ is measured | pass at all six                      |
| Template holes               | every daily maximum of $P^{\mathrm{cs}}_t$ above 40% of its peak         | lowest 56–70%                        |
| Template steps               | day-to-day change of the daily maximum below 5%                          | at most 1.0%                         |
| Output stability             | $k_t$ at 1.3 in less than 5% of measured daytime hours, every year       | at most 0.62%                        |
| Typical index                | daytime median of $k_t$ between 0.2 and 1.0                              | 0.624–0.672                          |
| Forecast irradiance          | daytime correlation of NWP GHI with $P^{\mathrm{cs}}_t$ above 0.3        | 0.76–0.78                            |
| No negative power            | no negative value of $P_t$                                               | pass at all six                      |
| No outage left               | the outage rule flags no measured hour                                   | pass at all six                      |
| Morning template             | mean daytime $P^{\mathrm{cs}}_t$ rises from the 06:00 to the 10:00 interval | pass at all six                      |
| Weather from the start       | NWP covariates present from the first hour                               | pass at all six, no missing value    |
| No-weather index             | same index as the NWP frame                                              | pass at all six                      |
| No-weather columns           | no weather columns                                                       | pass at all six                      |
| Hour labels (info)           | ratio of mean power in the first and last daytime hour of a day          | 0.98–1.26; SQ15 0.68                 |

## Outage hours

: Hours set missing by the outage rule of @sec:data-qc, per event and station (local days). The
three storm spells of 2023 (tower irradiance about 30–50 W/m²) count only where a station read at
most 0.1% of $C$ for three consecutive hours; ETL v3, which read the energy column, had also removed
them at LSK North (9 hours), SQ567 (10) and UG Hall6 (9). On 8 September two of the three hours at
SQ15 had first been bridged as a short gap. {#tbl:data-outages}

| Event                           | LSK North | SQ1 | SQ567 | UG Hall6 | CYT (Station 2) | SQ15 |
|:--------------------------------|----------:|----:|------:|---------:|----------------:|-----:|
| 11–12 June 2021                 |           |     |       |          |              14 |      |
| 10 November 2021                |           |     |       |          |                 |    3 |
| 23–28 March 2022                |           |     |       |          |                 |   45 |
| 9–10 November 2022              |           |     |       |          |                 |   16 |
| 7–16 May 2023                   |           |  99 |       |          |                 |      |
| Storm, 7 May 2023               |           |     |       |          |                 |    3 |
| Storm, 17 June 2023             |           |   3 |       |          |                 |    3 |
| Storm, 8 September 2023         |           |   3 |       |          |               3 |    3 |
| Total                           |         0 | 105 |     0 |        0 |              17 |   73 |

## Single events and gaps

The meter-error rules of @sec:data-qc removed three values on the panel, all at SQ15. No reading
exceeds 1.1 $C$; legitimate readings reach 1.005 $C$. Two readings lie below −0.05 $C$, of −111 $C$
and −148 $C$, in the hours from 09:00 on 3 April and from 14:00 on 29 July 2022; there are no smaller
negative readings, which would have become zero. One hourly value, 8.9 kW at 02:00 on 29 August 2022,
exceeds 1% of $C$ with the sun more than 5° below the horizon.

Three very dark storm spells in 2023 (7 May, 17 June and 8 September; tower irradiance about 30–50
W/m²) cut output to about 0.3% of $C$ or less. They stay as measured at LSK North, SQ567 and UG
Hall6; the outage rule removes them only where a station read at most 0.1% of $C$ for three hours,
18 hours at SQ1 and the hourly stations (@tbl:data-outages). ETL v3 had also removed them at the
other 15-minute stations, because the energy column it read is zero in many such quarter hours.

At the hourly stations the gaps longer than two hours are mostly days on which the feed holds no
value from 07:00 to 17:00: 28 days at CYT (Station 2) and 37 at SQ15.

Short gaps were bridged in 18 hours: 17 at SQ15, including the three values removed above, and one
at UG Hall6; five of them are daytime hours in test windows and are scored. Zero readings at the
edges of gaps and outages, where the sun is low or the run short, escape the outage rule: 113
daytime hours at three stations, 30 of them in test windows (0.17% of the scored hours), where they
are scored as observations.
<!-- TODO(Lorenzo): edge zeros remain in ETL v4 (unchanged from v3). Daytime hours that read
exactly 0 next to a missing hour or on an outage day: SQ1 9 (all 9 in test windows, 06:00 on 8-16
May 2023), CYT (Station 2) 42 (11), SQ15 62 (10); 30 in test windows = 0.17% of the 17,919 scored
hours (same count as the TODO in sec:method-metrics). Mostly the 17:00 hour after the 07:00-17:00
feed gaps, where P_cs reaches 26% of C at CYT (Station 2). They are scored as observations. Decide:
leave them (this sentence) or extend the outage rule to whole outage days in a later ETL. Evidence:
scratchpad ch3/v4_edge_defs.py, ch3/v4_edge_zeros_list.py. -->

## Power and energy columns

In the hourly files the power column equals the energy column, in kWh per hour, to within
rounding. In the 15-minute files the two columns differ by about 3%, and the power column is the
better record (@tbl:data-energy): it integrates to within 0.3% of the AC power that the inverters log
every five minutes, and the dataset paper's validation example, 642.09 kWh at the Library on 31
August 2022 against 641.55 kWh from its inverters [@lin2025hkust], is its integral. The energy
column lies 3.1–3.3% above the inverters and reads zero in 2–23% of the quarter hours in which the
power column reports 20 W to 1 kW, although it resolves 1 Wh. Both columns follow the tower's
irradiance over the quarter hour after their stamp more closely than over the neighbouring ones.

: Power and energy columns of the four 15-minute panel stations. Energy / power: daytime energy of the
energy column relative to the hourly means of the power column. Inverter ratio: energy of each
column relative to the AC power of the station's inverters (5-minute values), on the 872–895 days
with complete site files and at least ten hours of inverter data. Energy zeros: share of the quarter
hours with a power reading of 20 W to 1 kW (5–250 Wh) in which the energy column reads zero; it
resolves 1 Wh. Correlation: daytime correlation of each column, stamped $t$, with the campus tower's
one-minute GHI averaged over $[t, t+15)$ minutes, and in brackets the larger of the correlations for
the two neighbouring quarter hours; the tower's calibration problems do not affect this test of
timing. {#tbl:data-energy}

| Station        | Energy / power | Inverter ratio, power | Inverter ratio, energy | Energy zeros (%) | Correlation, power | Correlation, energy |
|:---------------|-----------:|-----------:|-----------:|-----------:|------------------:|------------------:|
| LSK North      |      1.029 |      1.003 |      1.031 |       22.8 |     0.897 (0.869) |     0.910 (0.867) |
| SQ1            |      1.030 |      1.003 |      1.033 |        2.2 |     0.864 (0.837) |     0.872 (0.842) |
| SQ567          |      1.029 |      1.002 |      1.031 |       16.1 |     0.887 (0.853) |     0.893 (0.863) |
| UG Hall6       |      1.028 |      1.002 |      1.031 |        4.9 |     0.894 (0.862) |     0.904 (0.867) |

## Labels of the hourly files

Each hourly file was correlated in daytime with the mean capacity factor ($P_t/C$) of the four
15-minute panel stations over the window $[t+s,\ t+s+60\ \mathrm{min})$, where $t$ is the raw
stamp. CYT (Station 2) peaks at $s = -60$ minutes and SQ15 at $-75$ to $-60$ minutes, so both label
the end of the hour; CYT (Station 1) peaks between −15 and −45 minutes, which no whole-hour shift can
align (@tbl:data-labels).

: Daytime correlation of each hourly file, with raw stamps $t$, with the mean capacity factor of the
four 15-minute panel stations over $[t+s, t+s+60\ \mathrm{min})$. Interval-start labels would peak
at $s = 0$, end-of-hour labels at $s = -60$ minutes. With the raw stamps, the first daytime hour of
a day carries 2% of the mean power of the last at CYT (Station 2) and at SQ15; after the shift by
one hour, 98% and 68%. {#tbl:data-labels}

| $s$ (min)       |   −90 |   −75 |   −60 |   −45 |   −30 |   −15 |     0 |   +15 |
|:----------------|------:|------:|------:|------:|------:|------:|------:|------:|
| CYT (Station 2) | 0.946 | 0.975 | 0.985 | 0.969 | 0.936 | 0.890 | 0.837 | 0.781 |
| SQ15            | 0.956 | 0.973 | 0.969 | 0.941 | 0.898 | 0.847 | 0.790 | 0.728 |
| CYT (Station 1) | 0.796 | 0.854 | 0.897 | 0.917 | 0.921 | 0.912 | 0.893 | 0.866 |

## Clear-sky template

The template of @eq:data-pcs was built in four steps:

1. The clear-sky global horizontal irradiance (GHI) $G^{\mathrm{cs}}_t$ comes from the
   Ineichen–Perez model
   [@ineichen2002linke] in pvlib for one campus coordinate (22.337° N, 114.263° E, 60 m),
   averaged over twelve instants in each hour, with the Linke turbidity (the dimming of clear-sky
   sunlight by water vapour and aerosols) from pvlib's monthly climatology, a table of long-term
   monthly averages [@remund2003linke].
2. The sun's position at the interval midpoint is expressed by the solar declination $\delta_t$,
   the latitude at which the sun stands overhead at noon, and the hour angle $\omega_t$, the sun's
   angular distance from solar noon, which grows by 15° per hour, is negative in the morning and
   is kept between −180° and 180°.
3. The ratio $\mathrm{PR}_t = (P_t / C) / (G^{\mathrm{cs}}_t / G_0)$ is collected over the first 365
   days of each record (ending between 23 March and 30 June 2022), for hours with measured power and
   $G^{\mathrm{cs}}_t \ge 50$ W/m², and binned on a grid of 4° of declination by 15° of hour angle
   (12 × 16 cells). Each cell with at least eight samples takes the 95th percentile of the ratio.
   The other cells are filled from their neighbours, and the grid is smoothed with a 3 × 3 mean and
   interpolated bilinearly, giving $\widehat{\mathrm{PR}}(\delta_t, \omega_t)$.
4. The template is @eq:data-pcs. Its peak lies between 0.75 $C$ (SQ15) and 0.99 $C$ (SQ1), so the
   cap at $C$ never binds on the panel.

Binning by the sun's position rather than by calendar month lets a month missing from the fit year
borrow from dates with the same declination on the other side of the solstice; an earlier,
month-binned version was zero throughout February at CYT (Station 1). The 1.3 ceiling of the
clear-sky index binds in 0.3–0.5% of daytime hours, 76–96% of them the first or last daytime hour of
a day. In the twilight hours just outside the daytime mask, $k_t$ is erratic, at the ceiling in
16–26% of them; they are not scored but remain in the models' inputs.

## Weather covariates

The request leaves the choice of weather model to Open-Meteo ("best match"); on six sample weeks
between March 2021 and November 2023 the cached series equal, value for value, those of the
high-resolution forecast of the ECMWF IFS (checked on 2 October 2026), archived on a 9 km grid
[@openmeteo] whose cell for the campus lies about 7 km away [@openmeteo_forecast_docs]. Moving the
radiation columns back by one hour raises the daytime correlation between forecast GHI and the mean
capacity factor of the four 15-minute stations from 0.73 to 0.85; instantaneous variables keep their
stamp. The cache holds complete calendar years, so every station has weather values from its first
hour.

For 2022–2023 the cached series jumps at 00 and 12 UTC. Of the scored hours, 06:00 to 18:00 HKT
(22:00 to 10:00 UTC), all but the first two therefore come from the 00 UTC run, which starts at 08:00
HKT; the first two come from the 12 UTC run of the day before (@fig:data-leadtime). At the issue
time, 00:00 HKT (16:00 UTC), the newest run actually available is that of 06 UTC (14:00 HKT) on the
previous day [@ecmwf_dissemination]. Open-Meteo's Previous Runs archive held no one-day-ahead
irradiance for July 2022 (checked on 2 October 2026).

: Weather covariates of the NWP setting (`pvfc.weather`). Archived value: convention of the
Open-Meteo variable [@openmeteo_forecast_docs]. Alignment: shift to interval-start labels. The last
three variables are stored but not passed to the models. Open-Meteo data are licensed under Creative
Commons Attribution 4.0 [@zippenfenig2024openmeteo]. {#tbl:data-weather}

| Column        | Open-Meteo variable    | Unit | Archived value          | Alignment | Used |
|:--------------|:-----------------------|:-----|:------------------------|:----------|:-----|
| `ghi`         | `shortwave_radiation`  | W/m² | mean over the past hour | −1 h      | yes  |
| `temp_air`    | `temperature_2m`       | °C   | instantaneous           | none      | yes  |
| `rh`          | `relative_humidity_2m` | %    | instantaneous           | none      | yes  |
| `wind_speed`  | `wind_speed_10m`       | km/h | instantaneous           | none      | yes  |
| `cloud_cover` | `cloud_cover`          | %    | instantaneous           | none      | no   |
| `ghi_direct`  | `direct_radiation`     | W/m² | mean over the past hour | −1 h      | no   |
| `ghi_diffuse` | `diffuse_radiation`    | W/m² | mean over the past hour | −1 h      | no   |

## Campus tower

The campus tower's irradiance record is not consistent over time (@fig:data-weather). Its mean GHI
over the midday hours (10:00–14:00) is 422 W/m² in 2022 and 632 W/m² in 2023, against 591 and 587
W/m² in the Open-Meteo archive, and in March and April 2021 it read almost nothing (63 and 9 W/m²).
In 2023 it reports 84 hourly means above 1,200 W/m², although the largest hourly clear-sky value on
the campus is 967 W/m², and one-minute values up to 3,609 W/m², more than 2.6 times the solar
irradiance at the top of the atmosphere, about 1,361 W/m² [@kopp2011new]. The PV output does not
follow these jumps: from 2022 to 2023 the panel's apparent performance ratio (the PR of
@sec:data-clearsky with measured or forecast GHI in place of $G^{\mathrm{cs}}_t$, over midday hours
on which all six stations report) falls by 5% with the archive but by 37% with the tower, and with
the tower it exceeds 1 in 2021 and 2022 (1.11 and 1.09), as if the arrays delivered more than their
nameplate capacity per 1,000 W/m². The tower data are therefore not used; the code keeps them as a
weather mode (`weather_mode: oracle`) that no reported run uses.

![Monthly mean of hourly GHI between 10:00 and 14:00 local time from the campus tower and from the
Open-Meteo archive (top), and the panel's apparent performance ratio for each irradiance series
(bottom; the PR of @sec:data-clearsky with measured or forecast GHI, over hours on which all six
stations report). The tower reads almost nothing in March and April 2021, 29% less than the archive
over 2022 and 8% more over 2023; the PV output stays in line with the
archive.](figures/data_weather.pdf){#fig:data-weather width=100%}
