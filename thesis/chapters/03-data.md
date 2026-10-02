# Data {#sec:data}

This chapter describes the dataset (@sec:data-dataset), the six stations used (@sec:data-panel), how
the raw files become hourly series ([@sec:data-time; @sec:data-qc]), the clear-sky template behind
the forecasting target (@sec:data-clearsky) and the weather covariates taken from numerical weather
prediction (NWP) models (@sec:data-weather). Several checks of the raw data changed the pipeline,
and a few found defects that remain in the current processing; each is reported where it belongs,
and @sec:data-summary lists the open ones.

## The HKUST rooftop PV dataset {#sec:data-dataset}

The data come from the open dataset of @lin2025hkust, deposited on Dryad under a CC0 public-domain
dedication [@lin2024hkustdata]: three years (2021–2023) of measurements from 60 grid-connected
rooftop photovoltaic (PV) stations on the campus of the Hong Kong University of Science and
Technology (HKUST) in Sai Kung (22.34° N, 114.26° E), with 2,230.8 kW of nameplate capacity in
total. The stations fall into two groups [@lin2025hkust]. In 23 stations the modules feed a single
inverter, which converts their direct current (DC) into alternating current for the grid and reports
the station's power every hour. The other 37 stations have panel-level optimizers, DC–DC converters
on each pair of modules that keep the modules at their maximum power point; these stations report
station-level power every 15 minutes, and their inverters every 5 minutes. Each station-level file
("site level" in the dataset) holds a time stamp, the energy of the interval (kWh) and the power
(W); of the PV files, this thesis uses only these. The authors aligned all streams to regular stamps
and marked missing values, but on purpose did no gap filling or outlier detection; they report
missing rates of 1.85% for the hourly and 0.79% for the 15-minute files. A 10-m weather tower on the
campus records global horizontal irradiance (GHI, with a LI-COR LI-200R pyranometer), temperature,
humidity, pressure, visibility, wind and rainfall every minute [@lin2025hkust].

The metadata are a Brick model [@balaji2016brick], a unified metadata schema for buildings, stored
as a Turtle (.ttl) file; it gives each station's rated power, contractor, grid-connection date, tilt
and azimuth. The pipeline reads the nameplate capacity $C$ (kW) from it
(`pvfc.data.load_nameplate_capacities`); the 60 values sum to the paper's total of 2,230.8 kW. $C$
normalises the error metrics (@sec:method-metrics), caps the clear-sky template and sets the
quality-control bound. The time stamps carry no time zone. They are read as Hong Kong Time (HKT),
eight hours ahead of Coordinated Universal Time (UTC+8) and without daylight saving. Solar noon on
the campus falls between 12:07 and 12:37 HKT, and with this reading the mean power of the processed
series peaks in the hour from 12:00 to 13:00 at five of the six panel stations and one hour earlier
at SQ1 (@tbl:data-checks).

## Station panel {#sec:data-panel}

The first full run of the benchmark, ten models in two weather settings, took 19.3 hours for six
stations on the laptop (@sec:method-compute), about 3.2 hours per station; all 60 stations would
need about eight days. The benchmark therefore runs on a panel of six stations, which keeps a full
run within a day. The panel was chosen from a coverage sweep of the 60 station-level files (hourly
means of the raw stamps, with daytime taken as the eleven hours labelled 07:00 to 17:00 local). The
median share of missing daytime hours is 0.2% at the 15-minute stations and 4.3% at the hourly
ones, with a worst case of 23.2%. Ten stations, all with optimizers, have no missing daytime hour
and at least two years of data, and three more miss at most four hours.
<!-- TODO(Lorenzo): the 28 Sep sweep left only its output (station_gap_report.csv, in
~/Thesis/thesis-review-2026-09-28/), not its script. A re-implementation that reproduces every
number of this paragraph and the next is in the scratchpad (revise-data/coverage_counts.py); add it
to the repository (e.g. experiments/coverage_sweep.py, output under results/) so that the selection
can be reproduced. -->

The selection criteria, recorded in `configs/experiment.yaml`, are (i) at least a year of data
before the first test day, 5 July 2022, so that every station is tested from that day on after a
full training year, which is also the fit window of the clear-sky template; (ii) a complete or
near-complete daytime record; (iii) stable output, meaning a clear-sky index at its ceiling in fewer
than 5% of daytime hours in every year (@sec:data-clearsky); (iv) a nameplate capacity of 25–55 kW,
for stations of comparable size; and (v) both groups, as four 15-minute and two hourly stations.
Nine 15-minute stations meet (i) and (ii) with at most one missing daytime hour: eight miss none,
and UG Hall6 misses the hour from 17:00 to 18:00 on 21 April 2023, which the interpolation of
@sec:data-qc bridges. The panel takes the largest of them (LSK North, 55 kW), a second size class
(SQ567, 48 kW), one of four similar staff-quarter towers (SQ1) and UG Hall6. CYT (Station 2) and
SQ15 are the hourly stations that meet (i) with the smallest shares of missing daytime hours (2.7%
and 3.8%). @Tbl:data-panel lists the six stations.
<!-- TODO(Lorenzo): confirm the within-criteria choices: SQ1 rather than its 33.3 kW siblings
SQ2-SQ4, and UG Hall7 and UG Hall4 not taken (they peak at 77-79% of nameplate). -->

: The six panel stations after processing (`data/processed/qc_log.csv`). Step: native resolution.
First hour: first interval-start label, in HKT; the hourly files were moved back by one hour
(@sec:data-time), so their records start at 23:00. All records end on 31 December 2023 (at 23:00;
at 22:00 for CYT (Station 2) and 10:00 for SQ15). Daytime hours: apparent solar zenith below 85° at
the interval midpoint (@sec:data-clearsky). Missing: daytime hours without a power value.
$k_t>1$: daytime hours with a clear-sky index above 1. {#tbl:data-panel}

| Station              | $C$ (kW)  | Step      | First hour (HKT)        | Daytime hours | Missing (%) | $k_t>1$ (%) |
|:---------------------|----------:|:----------|:------------------------|-------------:|----------:|----------:|
| LSK North            |     55.00 | 15 min    | 2021-06-01 00:00        |       10,717 |      0.00 |      6.97 |
| SQ1                  |     27.60 | 15 min    | 2021-06-01 00:00        |       10,717 |      0.00 |      7.07 |
| SQ567                |     47.97 | 15 min    | 2021-07-01 00:00        |       10,327 |      0.00 |      7.94 |
| UG Hall6             |     25.00 | 15 min    | 2021-06-01 00:00        |       10,717 |      0.00 |      8.49 |
| CYT (Station 2)      |     25.00 | 1 h       | 2021-05-22 23:00        |       10,834 |      2.07 |      7.85 |
| SQ15                 |     33.29 | 1 h       | 2021-03-23 23:00        |       11,535 |      2.75 |     11.92 |

Two candidates were dropped on evidence (`pvfc.data.EXCLUDED_STATIONS`). CYT (Station 1) ranks 59th
of 60 for coverage (8.8% of daytime hours missing), and its stamps are offset by about half an hour
(@sec:data-time). Zone A2 (249.5 kW), the only station above 55 kW with a year of data before the
first test day, changes regime in late May 2023. The daily maximum of its hourly power stays below
0.27 $C$ until 17 May 2023 (below 0.24 $C$ from 1 to 17 May) but lies between 0.65 and 0.73 $C$ on
each day from 27 May to 4 June, and the Brick model dates its grid connection to 31 May 2023,
although its record starts in July 2021. Against a template fitted on its first year, its clear-sky
index sits at the 1.3 ceiling in 42.6% of the measured daytime hours of 2023, against at most 0.74%
for any panel station in any year. UG Hall6 took its place. Without Zone A2 the effect of station
size cannot be tested: the panel represents stations of 25–55 kW, a range that contains the
dataset's median nameplate capacity (32 kW). It also represents well-kept stations, since criteria
(ii) and (iii) select complete records and stable output, so the results may be optimistic for the
rest of the dataset.

## Time stamps and resolution {#sec:data-time}

Every processed frame uses interval-start labels: an hourly value labelled HH:00 is the mean over the
hour from HH:00 to HH+1:00, and the clear-sky quantities of @sec:data-clearsky are computed for the
same intervals. The index is time-zone-aware UTC (`pvfc.data._read_ts_csv`), while the forecasting
task is defined on Hong Kong calendar days: forecasts are issued at local midnight (16:00 UTC on the
previous day) for the 24 hours of the day that begins (@sec:method-task).

The 15-minute files already label the start of each quarter hour. In daytime their power correlates
better with the tower's one-minute irradiance averaged over the quarter hour after the stamp
(0.86–0.90 at the four 15-minute panel stations) than over the quarter hour before it (0.83–0.85);
this test of timing is unaffected by the tower's calibration problems (@sec:data-weather). Their
first and last daytime hours also carry similar mean power (ratio 1.01–1.26, @tbl:data-checks), as
correct labels should: clear-sky irradiance is nearly symmetric about solar noon, while labels off
by an hour push the ratio far from 1 (see below). The pipeline averages the readings available in
each hour, normally four.

In the hourly files the power column equals the energy of the interval (power in W equals energy in
kWh times 1000), so their values are hourly means. In the 15-minute files power and energy agree
only approximately: the hourly means of the power column, which the pipeline uses, hold 2.8–2.9%
less daytime energy than the energy column, with a correlation of 0.997. The gap lies within the ±5%
accuracy the authors give for these calculated quantities [@lin2025hkust]. A constant factor of this
size would leave skill scores unchanged and scale the capacity-normalised errors of the four stations
by about 3%.
<!-- TODO(Lorenzo): keep the power column at the 15-minute stations (current choice, 2.8-2.9% less
energy than the generation column) or switch to generation, which gives the exact hourly energy? -->

The hourly files label the end of the hour. The dataset paper does not state the convention, so it
was inferred from the data: each hourly file, with raw stamps $t$, was correlated in daytime with
the mean capacity factor (power divided by $C$) of the four 15-minute panel stations over the window
$[t+s,\ t+s+60\ \mathrm{min})$, for shifts $s$ in steps of 15 minutes. Interval-start labels would
give the highest correlation at $s = 0$, end-of-hour labels at $s = -60$ minutes. CYT (Station 2)
peaks at $s = -60$ minutes (0.985, against 0.837 at $s = 0$), SQ15 at $-75$ and $-60$ minutes (0.973
and 0.968, against 0.790). With the raw stamps, the first daytime hour of a day also carries only 2%
of the mean power of the last at both stations; after moving the stamps back by one hour the ratio
is 0.98 at CYT (Station 2). The pipeline therefore shifts every hourly file back by one hour
(`pvfc.data._read_power`). At SQ15 the ratio rises only to 0.67, which, with the extra quarter hour
in the correlation, suggests an array that produces relatively more in the afternoon; its metadata
("Mixed", a curved roof) give no orientation to check this against. CYT (Station 1) fits neither
convention: its correlation is flat at 0.91–0.92 for shifts of −15 to −45 minutes and lower at 0 and
−60 minutes (0.89 and 0.90), so no whole-hour relabelling can align it.

## Quality control {#sec:data-qc}

Quality control is deliberately light. The dataset authors did no gap filling or outlier detection
[@lin2025hkust], so the rules below are the only cleaning; they are few because each one changes the
target on which every model is scored, and they apply to all models alike. Raw readings above 1.1
$C$ are treated as meter errors and set to missing before resampling, so that a single spike cannot
contaminate an hourly mean (`pvfc.data._qc_power`); the margin keeps legitimate readings slightly
above nameplate, which reach 1.005 $C$ on the panel. The bound matters elsewhere in the dataset
(Tower A, SQ9 and UG Hall4 Flexible PV reach 788, 42 and 9 times their nameplate) but removed no
reading from the panel. Negative readings are set to zero, and this rule does act on the panel:
SQ15's file holds two meter errors of −3,699 kW and −4,916 kW, more than a hundred times its
nameplate, which become hours of zero power at 09:00 on 3 April 2022 and at 14:00 on 29 July 2022,
between hours of 6–15 kW. The second lies in a test window and is scored as an observation.
<!-- TODO(Lorenzo): pvfc.data._read_power clips negatives to 0 before _qc_power, so SQ15's two gross
meter errors (raw stamps 2022-04-03 10:00 and 2022-07-29 15:00, about -111 C and -148 C) become
0 kW daytime hours; the second is in fold 0 and scored for every model. Decide: treat large negative
readings (e.g. below -0.05 C) as missing in the next ETL version and rebuild SQ15, or keep this
paragraph. The dataset holds 12 negative readings at 9 stations. Evidence: scratchpad
revise-data/negatives_zone_twilight.py, coverage_counts.py. -->

Gaps stay missing: a missing hour is not an hour without generation, and zeros would show the models
dark days that never happened. The clear-sky index of a missing hour is missing too
(`pvfc.normalize.clearsky_index`), and missing hours are not scored (@sec:method-metrics). One step
departs from this rule: after resampling, the pipeline interpolates linearly across up to two
consecutive missing hours (`pvfc.data.load_station`), which at the 15-minute stations bridges a
single hour (at UG Hall6). The gaps of the hourly stations are mostly outage days on which the ten
hours from 07:00 to 17:00 are missing (all 28 gaps at CYT (Station 2), 37 of 55 at SQ15), while the
hours around them, including a few daytime hours at sunrise and sunset, read zero. As implemented in
pandas, the interpolation also fills the first two hours of every longer gap, on outage days with
zeros: 51 daytime hours at CYT (Station 2) and 69 at SQ15, of which 15 and 17 lie in test windows.
At SQ15 it also fills 11 daytime hours at the start of six shorter gaps of 3–8 hours, and it bridges
14 daytime hours in gaps of one or two hours, 6 of them in test windows, as intended.
<!-- TODO(Lorenzo): interpolate(limit=2) in pvfc.data.load_station also fills the first two hours
of every longer gap, so 07:00-09:00 of each outage day at CYT (Station 2) and SQ15 hold 0 kW that
models train on and that are scored as truth (15 + 17 daytime hours in test windows); the zero
readings just outside the gap (06:00, 17:00, 18:00) look like reporting artefacts too (about 1.3
per outage day against <0.2 per day otherwise). Decide: fix in an ETL v3 (interpolate only runs of
at most 2 h, or mask whole outage days) and rerun the two hourly stations, or keep this paragraph.
Evidence: scratchpad revise-data/gap_fills.py; data-chapter/outage_edges.py. -->

After processing, the 15-minute stations have no missing daytime hours; CYT (Station 2) misses 2.07%
and SQ15 2.75% (2.63% and 2.89% within the test windows), mostly on outage days, taken here as days
with at least half of the daytime hours missing: 28 at CYT (Station 2) and 38 at SQ15, 17 of them
shared, and 10 and 11 inside test windows (@fig:data-coverage).

![Share of daytime hours without a power value, per panel station and local day, after
processing. Each bar spans the station's record; white means no record. The dark lines at the two
hourly stations are outage days. The nine 30-day test windows (top) start on 5 July 2022, after
at least a year of data at every station.](figures/data_coverage.pdf){#fig:data-coverage
width=100%}

Before any model runs, `experiments/check_data.py` checks each station against the invariants in
@tbl:data-checks, and the overnight batch stops if one fails (`experiments/run_panel.py`). All six
stations passed.

: Pre-flight checks of `experiments/check_data.py` on the six panel stations
(`results/logs/check_data__*.log`). Night: outside the daytime mask. The last row is informational.
{#tbl:data-checks}

| Check                      | Criterion                                                            | Panel result                       |
|:---------------------------|:---------------------------------------------------------------------|:-----------------------------------|
| Time index                 | time-zone-aware UTC                                                  | pass at all six                    |
| Peak hour                  | mean power peaks in the 11:00, 12:00 or 13:00 interval (local)       | 12:00 at five stations, 11:00 at SQ1 |
| Night power                | mean power outside the daytime mask below 1% of $C$                  | at most 0.07% of $C$               |
| Daytime coverage           | less than 20% of daytime hours missing                               | 0.00–2.75%                         |
| Quality bound              | maximum hourly power at most 1.1 $C$                                 | 0.80–1.00 $C$                      |
| Finite target              | $P^{\mathrm{cs}}_t$ finite in daytime; $k_t$ finite where $P_t$ is measured | pass at all six                    |
| Template holes             | every daily maximum of $P^{\mathrm{cs}}_t$ above 40% of its peak     | lowest 55–70%                      |
| Template steps             | day-to-day change of the daily maximum below 5%                      | at most 1.0%                       |
| Output stability           | $k_t$ at 1.3 in less than 5% of measured daytime hours, every year   | at most 0.74%                      |
| Typical index              | daytime median of $k_t$ between 0.2 and 1.0                          | 0.625–0.665                        |
| Forecast irradiance        | daytime correlation of NWP GHI with $P^{\mathrm{cs}}_t$ above 0.3    | 0.76–0.78                          |
| No-weather frame           | same index as the NWP frame, no weather columns                      | pass at all six                    |
| Hour labels (info)         | ratio of mean power in the first and last daytime hour near 1        | 0.98–1.26; SQ15 0.67               |

## Clear-sky template and clear-sky index {#sec:data-clearsky}

PV output follows the daily and seasonal path of the sun, which is known in advance; a forecast has
to predict how much of the clear-sky output the weather lets through. The models therefore work with
the clear-sky index, the measured power divided by a clear-sky power template $P^{\mathrm{cs}}_t$
(@sec:method-target).

A physical model of each array is impractical. The metadata list flat arrays (SQ1, SQ567, UG Hall6),
modules split between two directions (LSK North, CYT (Station 2)) and, for SQ15, a "Mixed"
orientation, the label the dataset uses for curved roofs [@lin2025hkust]; shading is not recorded.
The template is therefore learned from the data, through a performance ratio (PR): the capacity
factor $P_t / C$ divided by the horizontal irradiance in units of $G_0 = 1{,}000$ W/m². A PR of 1
means that a station delivers its nameplate power at 1,000 W/m² of horizontal irradiance, and
proportionally less at lower irradiance. Taken as a function of the sun's position, the PR absorbs
an array's tilt, orientation, efficiency and fixed shading. The template is built in four steps
(`pvfc.clearsky.empirical_clearsky_power`):

1. The clear-sky GHI $G^{\mathrm{cs}}_t$ comes from the Ineichen–Perez model
   [@ineichen2002linke] in pvlib [@holmgren2018pvlib; @anderson2023pvlib], for one campus
   coordinate (22.337° N, 114.263° E, 60 m), averaged over twelve instants in each hour. The model
   takes the Linke turbidity, which describes how strongly water vapour and aerosols dim clear-sky
   sunlight relative to a clean, dry atmosphere, from pvlib's monthly climatology
   [@remund2003linke].
2. The sun position at the interval midpoint is expressed by two angles: the solar declination
   $\delta_t$, the latitude at which the sun stands overhead at noon, which depends only on the
   date; and the hour angle $\omega_t$, the sun's angular distance from solar noon, which grows by
   15° per hour and is negative in the morning.
3. The performance ratio with respect to the clear-sky GHI,
   $\mathrm{PR}_t = (P_t / C) / (G^{\mathrm{cs}}_t / G_0)$, is collected over the first 365 days of
   each record, for hours with measured power and $G^{\mathrm{cs}}_t \ge 50$ W/m², and binned on a
   grid of 4° of declination by 15° of hour angle (12 × 16 cells). Each cell with at least eight
   samples takes the 95th percentile of the ratio, so that the template follows the clearest hours;
   there is no separate clear-sky detection. Cells with fewer samples are filled from their
   neighbours, and the grid is smoothed with a 3 × 3 mean and interpolated bilinearly, giving $\widehat{\mathrm{PR}}(\delta_t, \omega_t)$. The code stores the
   ratio as $P_t / G^{\mathrm{cs}}_t$, in kW per W/m², which differs from $\mathrm{PR}_t$ only by the
   constant factor $C / G_0$.
4. The template is

   $$
   P^{\mathrm{cs}}_t = \min\left\{ C,\; \widehat{\mathrm{PR}}(\delta_t, \omega_t)\, C\, \frac{G^{\mathrm{cs}}_t}{G_0} \right\},
   $$ {#eq:data-pcs}

   which is zero when the sun is below the horizon for the whole hour. Its peak lies between
   0.75 $C$ (SQ15) and 0.99 $C$ (SQ1), so the cap at $C$ never binds on the panel.

The implementation used for every result in this thesis departs from step 2 in the early morning
(`pvfc.clearsky.sun_coordinates`). The hour angle is computed from UTC time stamps and not wrapped
into the range from −180° to 180°, so for the hours before 08:00 local time it comes out 360° too
high and is clipped into the last column of the grid, [105°, 120°), which lies after sunset. All
fitted samples of the hours from 06:00 to 08:00 land in that column, every hour before 08:00 reads
one ratio per declination band, and the true morning cells are filled from their neighbours; 134 of
the 192 cells are fitted at every panel station, 12 of them in that column, against 135 with a
wrapped hour angle. The template is unaffected from 10:00 to 16:00 and differs by up to about 10% at
06:00 and 18:00 (median per station); the daytime clear-sky index changes by 0.006–0.011 on average
and by more than 0.05 in 3–9% of daytime hours, all before 10:00 or from 17:00 on. The flaw is the
same for all models and leaves the measured power on which they are scored untouched; whether it
changes any result can only be settled by a rerun.
<!-- TODO(Lorenzo): pvfc.clearsky.sun_coordinates passes the UTC index to
pvlib.solarposition.hour_angle, which does not wrap its result, so hours before 08:00 local get
180-294 deg and are clipped into the after-sunset column. Preferred fix: ha = (ha + 180) % 360 - 180
in sun_coordinates, bump ETL_VERSION, rebuild the frames and rerun the panel (135 cells would be
fitted); then replace this paragraph by one sentence. Or keep the code and this paragraph. Evidence:
scratchpad revise-data/hour_angle.py, hour_angle_by_hour.py. -->

The fit windows end between 23 March and 30 June 2022, before the first test day, and the template is
applied unchanged to the whole record. It therefore carries no test-period information, but it does
not follow slow changes after the first year, such as soiling or degradation.

Binning by sun position rather than by calendar month is deliberate. An earlier version of the
pipeline, developed on CYT (Station 1), took the percentile per calendar month and hour and failed
wherever the fit year was unusual. With no data for February 2022, the template of CYT (Station 1)
was zero throughout every February, which set its clear-sky index to zero for the whole month; its
March and June templates sat 30–40% low after cloudy months in the fit year; and the template jumped
at every month boundary. Declination is nearly symmetric about the solstices, so a month missing
from the fit year borrows from dates with a similar declination on the other side of the solstice
(February from November, for example), and the interpolation removes the steps. On the panel, no
day's maximum of $P^{\mathrm{cs}}_t$ falls below 55% of its peak and no day differs from the
previous one by more than 1.0% (@tbl:data-checks).

The daytime mask keeps the hours in which the sun stands more than 5° above the horizon at mid-hour,
that is, whose apparent solar zenith (the angle between the sun and the vertical, corrected for
atmospheric refraction) at the interval midpoint is below 85°. It is computed with pvlib's
implementation of the solar position algorithm of @reda2004spa and defines the scored hours
(@sec:method-metrics), about 11 per day.

The clear-sky index is $k_t = P_t / P^{\mathrm{cs}}_t$, clipped to $[0, 1.3]$; it is zero when the
template is zero, that is, when the sun is below the horizon for the whole hour
($P^{\mathrm{cs}}_t \le 10^{-6}$ kW), and otherwise missing where $P_t$ is missing (@eq:k).
@Fig:data-clearsky shows one week at LSK North. In the twilight hours just outside the daytime mask,
when the sun is up for part of the hour but less than 5° high at mid-hour, $k_t$ is computed as
usual. It is erratic there: each station has 1,526–1,691 such hours, and $k_t$ sits at the 1.3
ceiling in 219–389 of them. These hours do not enter the headline scores, but they are part of the
series that the trained models learn from and that the foundation models read as context.

![Measured power and clear-sky template at LSK North (55 kW) in the first week of the first test
window, 5–11 July 2022 (top), and the clear-sky index of the daytime hours (bottom). Power stays far
below the template on the overcast days at the start and follows it on the clear days, where $k_t$
stays close to 1.](figures/data_clearsky.pdf){#fig:data-clearsky width=100%}

Daytime values above 1 come from cloud enhancement, when broken clouds briefly push irradiance above
its clear-sky value [@gueymard2017cloud], and from clear hours that lie above a template set at the
95th percentile. They occur in 7.0–11.9% of daytime hours (@tbl:data-panel, @fig:data-k), most often
at SQ15. The ceiling binds in only 0.3–0.6% of daytime hours, and 81–97% of these are the first or
last daytime hour of a day, when $P^{\mathrm{cs}}_t$ is small. The ceiling also serves as the
stability check of criterion (iii) in @sec:data-panel. With a template frozen after one year, an
increase in output, such as capacity added after the fit year, shows up as $k_t$ piling up at 1.3,
as at Zone A2; a loss of output, such as new shading, lowers $k_t$ and passes this check unnoticed.

![Distribution of the daytime clear-sky index at each panel station over its whole record, as a
share of all daytime hours, in bins of 0.05. The vertical line marks $k_t = 1$; the orange bar is
the share clipped at 1.3.](figures/data_k.pdf){#fig:data-k width=100%}

## Weather covariates {#sec:data-weather}

In the no-weather setting the models see only the clear-sky template and calendar features
(@sec:method-covariates). The NWP setting adds archived forecasts from the Historical Forecast
service of Open-Meteo [@openmeteo], fetched for the campus coordinate (`pvfc.weather`;
@tbl:data-weather); Open-Meteo licenses its data under Creative Commons Attribution 4.0
[@zippenfenig2024openmeteo]. The request leaves the weather model to Open-Meteo's default "best
match". For the campus, the cached series equal, value for value, Open-Meteo's series from the
high-resolution forecast (HRES) of the Integrated Forecasting System (IFS) of the European Centre
for Medium-Range Weather Forecasts on six sample weeks between May 2021 and November 2023 (checked
on 2 October 2026).
The covariates are therefore in practice IFS HRES forecasts, which Open-Meteo archives from 2017 on
a 9 km grid [@openmeteo]. The grid cell that Open-Meteo selects for the campus, by default a land
cell of similar elevation, lies about 7 km away [@openmeteo_forecast_docs].

: Weather covariates of the NWP setting (`pvfc.weather`). Archived value: convention of the
Open-Meteo variable [@openmeteo_forecast_docs]. Alignment: shift to interval-start labels. The last
three variables are stored but not passed to the models. {#tbl:data-weather}

| Column        | Open-Meteo variable    | Unit | Archived value          | Alignment | Used |
|:--------------|:-----------------------|:-----|:------------------------|:----------|:-----|
| `ghi`         | `shortwave_radiation`  | W/m² | mean over the past hour | −1 h      | yes  |
| `temp_air`    | `temperature_2m`       | °C   | instantaneous           | none      | yes  |
| `rh`          | `relative_humidity_2m` | %    | instantaneous           | none      | yes  |
| `wind_speed`  | `wind_speed_10m`       | km/h | instantaneous           | none      | yes  |
| `cloud_cover` | `cloud_cover`          | %    | instantaneous           | none      | no   |
| `ghi_direct`  | `direct_radiation`     | W/m² | mean over the past hour | −1 h      | no   |
| `ghi_diffuse` | `diffuse_radiation`    | W/m² | mean over the past hour | −1 h      | no   |

Open-Meteo stamps a radiation value at the end of the hour it averages [@openmeteo_forecast_docs], so
the radiation columns are shifted back by one hour. The shift raises the daytime correlation between
forecast GHI and the mean capacity factor of the four 15-minute stations from 0.73 to 0.84.
Instantaneous variables keep their stamp, which is the start of the interval. The cache begins on 4
May 2021, so SQ15 has no forecast covariates for its first 993 hours; the model adapters fill them
with the first available values (`pvfc.models.darts_wrapper`), and these hours lie in training data
only.
<!-- TODO(Lorenzo): the 2021 Open-Meteo cache was fetched starting 4 May 2021 (for CYT (Station 1))
and is reused for every station. The API serves 24 Mar - 3 May 2021 as well (checked 2 Oct 2026).
Re-fetch 2021 from 1 January and rerun SQ15 in the NWP setting, or keep this sentence. -->

The archive is not a set of day-ahead forecasts. Open-Meteo stitches the first few hours of each
successive model run into one series, and IFS starts a new run every six hours [@openmeteo], so each
archived value comes from a run at most a few hours old. A forecast issued at local midnight would
come from an earlier run, and for the midday hours that matter most for PV, 11:00–14:00, its lead
time would be at least 11–14 hours. The NWP setting is therefore optimistic, by an amount that
Open-Meteo cannot measure for the panel period. Its Previous Runs service keeps forecasts at fixed
lead times of one to seven days, mostly from January 2024 only, with exceptions for 2-m temperature
from the US Global Forecast System (GFS, from March 2021) and for the models of the Japan
Meteorological Agency (JMA, from 2018) [@openmeteo_previous_runs]. For July 2022 it returned
one-day-ahead 2-m temperature from best match, GFS and the JMA global model, but no one-day-ahead
irradiance from any of them and no data at all from IFS (checked on 2 October 2026). The two
settings therefore approximately bracket what a real day-ahead weather forecast can add, the NWP
setting from above and the no-weather setting from below. A conclusion that holds in both is
unlikely to hinge on the optimism of the archive, although intermediate forecast quality is not
tested.

![Monthly mean of hourly GHI between 10:00 and 14:00 local time from the campus tower and from the
Open-Meteo archive (top), and the panel's apparent performance ratio for each irradiance series
(bottom; the PR of @sec:data-clearsky with measured or forecast GHI, over hours on which all six
stations report). Over each year the tower reads about 30% below the archive in 2021–2022 and 8%
above it in 2023; the PV output stays in line with the
archive.](figures/data_weather.pdf){#fig:data-weather width=100%}

The campus tower could have supplied an "oracle" setting with measured weather in place of
forecasts, but its irradiance record is not consistent over time (@fig:data-weather). Its mean GHI
over the midday hours (10:00–14:00) is 403, 422 and 632 W/m² in 2021 (from 4 May, when the archive
starts), 2022 and 2023, against 597, 591 and 587 W/m² in the archive: about 30% below the archive
until January 2023 and above it for most of 2023. In 2023 it reports 84 hourly means above 1,200
W/m², although the largest hourly clear-sky value on the campus is 967 W/m², and one-minute values
up to 3,609 W/m², more than 2.6 times the solar irradiance at the top of the atmosphere, about 1,361
W/m² [@kopp2011new]; in March and April 2021 it read almost nothing. The PV output does not follow
these jumps: the panel's apparent performance ratio (the PR of @sec:data-clearsky with measured or
forecast GHI in place of $G^{\mathrm{cs}}_t$) changes by 6% from 2022 to 2023 with the archive but
by 37% with the tower, and with the tower it exceeds 1 in 2021 and 2022, as if the arrays delivered
more than their nameplate power per 1,000 W/m². The oracle setting is therefore kept in the code for
diagnostics only (`weather_mode: oracle`) and is not part of the benchmark.

## Summary {#sec:data-summary}

Each panel station ends up as one hourly UTC series of $P_t$, $P^{\mathrm{cs}}_t$, $k_t$ and a
daytime flag, plus the forecast covariates in the NWP setting, spanning 21,936 hours (SQ567) to
24,300 hours (SQ15). All stations are scored on the same nine test windows of 30 local days, from 5
July 2022 to 26 November 2023: 270 test days and 3,041 daytime hours per station. Every station's
first training window holds at least 365 days (368 at SQ567, the last to start).
@Sec:method-protocol describes the folds (@tbl:method-folds).

Four known defects of the current processing remain, all of them shared by every model: the
unwrapped hour angle of the template (@sec:data-clearsky), the two negative meter readings at SQ15
that became zeros, the zero fills at the start of outage gaps at the hourly stations (both
@sec:data-qc), and the constant weather covariates of SQ15's first 993 hours (@sec:data-weather).
The data cover one subtropical campus over three years, a limit on generalisation that the dataset
authors point out themselves [@lin2025hkust], and the six stations were selected for complete records
and stable output (@sec:data-panel).
