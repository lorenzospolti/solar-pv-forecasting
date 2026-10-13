---
title: "Day-Ahead Photovoltaic Power Forecasting on Consumer Hardware: Time-Series Foundation Models versus Purpose-Trained Models"
author: "Lorenzo Spolti"
# The cover of the submitted thesis is printed by the BAI LaTeX class (~/Thesis/BAI_Thesis_Spolti/thesis.tex:
# \subject, \AY{2025/2026}, \supervisor); this pandoc build is only a draft view.
supervisor: "Prof. Claudio Cusano, University of Pavia"
bibliography: references.bib
link-citations: true
# Libertinus (in tectonic's bundle, loaded by file name): covers the symbols Latin Modern lacks (≈ ≥ ⁻²)
mainfont: LibertinusSerif-Regular.otf
mainfontoptions:
  - BoldFont=LibertinusSerif-Bold.otf
  - ItalicFont=LibertinusSerif-Italic.otf
  - BoldItalicFont=LibertinusSerif-BoldItalic.otf
sansfont: LibertinusSans-Regular.otf
mathfont: LibertinusMath-Regular.otf
# wide results tables (up to 11 columns) need a smaller font than the 11 pt body
header-includes:
  - \usepackage{etoolbox}
  - \AtBeginEnvironment{longtable}{\small}
  # figures stay where the text introduces them instead of floating past the long tables
  - \usepackage{float}
  - \floatplacement{figure}{H}
# pandoc-crossref (make manuscript): chapter-relative numbering, "Figure 3.1"
chapters: true
chaptersDepth: 1
figPrefix: ["Figure", "Figures"]
tblPrefix: ["Table", "Tables"]
secPrefix: ["Section", "Sections"]
eqnPrefix: ["Equation", "Equations"]
linkReferences: true
---

<!--
One file per chapter, concatenated in file-name order by `make manuscript`
(pandoc + pandoc-crossref + citeproc + tectonic -> thesis/build/thesis.pdf).
Chapters are level-1 headings; cross-reference with @sec:, @fig:, @tbl:, @eq:
labels; cite with [@key] from references.bib. Image paths are relative to thesis/.
Numbers, tables and figures come from thesis/tables/ and thesis/figures/
(regenerate with `make report`).
-->

# Abstract {.unnumbered}

<!-- TODO(Lorenzo): drafted with Claude on 5 Oct 2026 and revised with Claude on 12 Oct 2026 after the
review (~/Thesis/Thesis Review 2026-10-12 - Cover, Abstract, Chapter 1.md); rewrite it in your own words
before submission (UniPV AI guidelines). Every number is in Tables 5.1-5.5. The LaTeX copy is
~/Thesis/BAI_Thesis_Spolti/content/abstract.tex: keep the two in step. -->

Solar output follows the weather, so the next day's production has to be forecast for grid
scheduling and day-ahead markets. Time-series foundation models promise such forecasts with no
training on the site's own data. This thesis tests that promise on six rooftop photovoltaic stations
in Hong Kong, forecasting their hourly power one day ahead on a 16 GB laptop. Three foundation
models (Chronos-2, TimesFM and TiRex) are used zero-shot, and Chronos-2 is also fine-tuned on each
station's history. They are compared with three purpose-trained models (SARIMA, LightGBM and an
LSTM) and with simple reference forecasts, all scored on the same 270 test days. Accuracy is
measured as RMSE skill, the relative reduction in RMSE against CLIPER, a reference that blends the
long-term average with yesterday's value, with bootstrap confidence intervals. With archived weather
forecasts as inputs, three models lead and cannot be told apart: LightGBM at default settings (skill
0.28), the small Chronos-2 checkpoint fine-tuned on each station (0.26) and zero-shot Chronos-2 with
its larger checkpoint and a long context window (0.27), a configuration chosen after the main
results had been seen. Without weather forecasts, no model beats CLIPER; TimesFM and TiRex, which
take no weather inputs in the software used here, are tested in this setting only. With weather
forecasts, fine-tuning adds 0.14 in skill at the short context fixed in advance; a longer context
alone gives about two thirds of that gain without training (fine-tuning at a long context did not
fit in 16 GB at the batch size used). The best zero-shot configuration forecasts 30 days for one station in 12 seconds, against about three minutes to fit LightGBM or to fine-tune Chronos-2. The archived weather
forecasts come from weather-model runs not yet available when the power forecasts are issued, so the
gains from them are an upper bound. On these data, the weather input matters more than the model
class, and a zero-shot foundation model given weather forecasts and a long context is a competitive
option that needs no forecasting model to be trained.

**Keywords:** Photovoltaic power forecasting, Day-ahead forecasting, Time-series foundation models,
Zero-shot forecasting, Fine-tuning, Numerical weather prediction, Forecast verification

# Notation and abbreviations {.unnumbered}

The subscript $t$ denotes an hour, labelled by the start of its interval
(@sec:data-time), and $s$ a station. Symbols used in one section only are defined where they occur.

| Symbol | Meaning | Unit | Section |
|:-------------------|:-----------------------------------------------------------|:------|:---------|
| $P_t$ | Measured power, mean over hour $t$ | kW | [-@sec:data-time] |
| $\hat{P}_t$ | Forecast of $P_t$; for a probabilistic model, the mean of its nine quantiles | kW | [-@sec:method-metrics] |
| $P^{\mathrm{cs}}_t$ | Clear-sky power (the template): the power a station would deliver under a cloudless sky | kW | [-@sec:data-clearsky] |
| $G^{\mathrm{cs}}_t$ | Clear-sky global horizontal irradiance, the same for every station | W/m² | [-@sec:data-clearsky] |
| $G_0$ | Irradiance of 1,000 W/m², the unit of irradiance in the ratio PR | W/m² | [-@sec:data-clearsky] |
| $C$ | Nameplate capacity: the rated power in the metadata, either the AC rating of the inverter or the total DC rating of the modules | kW | [-@sec:data-dataset] |
| $\widehat{\mathrm{PR}}$ | Ratio of the capacity factor $P_t/C$ to $G^{\mathrm{cs}}_t/G_0$, fitted to the clearest hours as a function of $\delta_t$ and $\omega_t$ | -- | [-@sec:data-clearsky] |
| $k_t$ | Clear-sky index $P_t / P^{\mathrm{cs}}_t$, clipped to $[0, 1.3]$ | -- | [-@sec:data-clearsky] |
| $\hat{k}_t$ | Forecast of $k_t$ | -- | [-@sec:method-target] |
| $\bar{k}$ | Mean daytime clear-sky index of the training window | -- | [-@sec:method-models] |
| $\rho$ | Correlation between $k_t$ and $k_{t-24\,\mathrm{h}}$ over the daytime hours of the training window; the weight of persistence in CLIPER | -- | [-@sec:method-models] |
| $\delta_t$ | Solar declination, the latitude at which the sun stands overhead at noon | ° | [-@sec:data-clearsky] |
| $\omega_t$ | Hour angle, the sun's angular distance from solar noon, negative in the morning | ° | [-@sec:data-clearsky] |
| $\tau$ | Quantile level, $0.1, 0.2, \dots, 0.9$ | -- | [-@sec:method-models] |
| $\hat{q}_{\tau,t}$ | Forecast $\tau$-quantile of power | kW | [-@sec:method-metrics] |
| $\mathrm{SS}$ | RMSE skill score over a reference; $\mathrm{SS}_s$ at station $s$ | -- | [-@sec:method-metrics] |
| $\mathcal{H}_s$ | Scored hours of station $s$, pooled over the nine test windows | -- | [-@sec:method-metrics] |

Abbreviations used only once, or only in an appendix, are spelled out where they occur.

| Abbreviation | Meaning |
|:-------------|:---------------------------------------------------------------|
| AC | Alternating current |
| CLIPER | Climatology–persistence blend, the headline reference forecast |
| CPU | Central processing unit |
| CRPS | Continuous ranked probability score |
| DC | Direct current |
| DM | Diebold–Mariano (test) |
| GPU | Graphics processing unit |
| HKT | Hong Kong Time, UTC+8 |
| HKUST | Hong Kong University of Science and Technology |
| IFS | Integrated Forecasting System of the European Centre for Medium-Range Weather Forecasts |
| LSTM | Long short-term memory (network) |
| nRMSE | RMSE divided by the nameplate capacity |
| NWP | Numerical weather prediction |
| PICP | Prediction interval coverage probability |
| PR | Performance ratio: the capacity factor $P_t/C$ divided by the clear-sky irradiance in units of $G_0$ |
| PV | Photovoltaic |
| QS | Quantile score |
| RMSE | Root mean squared error |
| RQ | Research question |
| SARIMA | Seasonal autoregressive integrated moving average |
| UTC | Coordinated Universal Time |
