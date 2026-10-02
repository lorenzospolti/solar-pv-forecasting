# Data quality checks {#sec:appendix-data}

This appendix collects the evidence behind the data decisions of Chapter [-@sec:data]: the
pre-flight checks run before every batch (@tbl:data-checks), the hours removed by the outage rule
(@tbl:data-outages), the power and energy columns of the 15-minute files (@tbl:data-energy), the
label convention of the hourly files (@tbl:data-labels), the weather covariates (@tbl:data-weather)
and the campus tower (@fig:data-weather). All numbers refer to the data of the final processing
(ETL v4).

## Pre-flight checks

: Pre-flight checks of `experiments/check_data.py` on the six panel stations
(`results/logs/check_data__*.log`). Night: outside the daytime mask. Measured: with a power value.
The last row is informational. {#tbl:data-checks}

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

## Power and energy columns

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

## Weather covariates

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
nameplate power per 1,000 W/m². The tower data therefore serve only for diagnostics
(`weather_mode: oracle`).

![Monthly mean of hourly GHI between 10:00 and 14:00 local time from the campus tower and from the
Open-Meteo archive (top), and the panel's apparent performance ratio for each irradiance series
(bottom; the PR of @sec:data-clearsky with measured or forecast GHI, over hours on which all six
stations report). The tower reads almost nothing in March and April 2021, 29% less than the archive
over 2022 and 8% more over 2023; the PV output stays in line with the
archive.](figures/data_weather.pdf){#fig:data-weather width=100%}
