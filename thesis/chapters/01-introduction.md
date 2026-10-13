# Introduction {#sec:intro}

<!-- TODO(Lorenzo): drafted with Claude on 5 Oct 2026 and revised with Claude on 12 Oct 2026 after the
review (~/Thesis/Thesis Review 2026-10-12 - Cover, Abstract, Chapter 1.md); rewrite it in your own words
before submission (UniPV AI guidelines: no unrevised AI-written paragraphs). Read the introduction of
Antonanzas et al. (2016) before keeping that citation. Keep RQ1-RQ4 identical to the Results headings.
The LaTeX copy is ~/Thesis/BAI_Thesis_Spolti/content/introduction.tex: keep the two in step.
On 13 Oct 2026 Claude added the glosses of "checkpoint" and "context window" in "In brief" after
your review (~/Thesis/Thesis Review 2026-10-13 - Assessment ...md, item A2). -->

The output of a photovoltaic (PV) system follows the weather. As the share of solar power grows,
forecasting it becomes crucial for the stability of the grid and for deciding in advance which power
plants run and how much each produces. Such forecasts are made for single sites as well as for whole regions [@antonanzas2016review]. Much of this planning is done the day before, in grid
scheduling and in day-ahead electricity markets. This thesis studies a basic form of that task for
rooftop PV systems, called *stations* in the dataset used here: 24 hourly values of power for the
coming day, issued at midnight.

The usual approach trains a model, such as a statistical model, gradient-boosted trees or a neural
network, on each station's own history and feeds it weather forecasts; this thesis calls such models
*purpose-trained*. Time-series foundation models offer another route. Pretrained on large
collections of series from many domains, they forecast a new series from its recent history without
any training on it (*zero-shot*) [@ansari2025chronos2; @auer2025tirex; @das2024timesfm]. If they
were accurate enough, a new station could be forecast with no model to build and maintain.

Whether they are accurate enough is open: two studies from 2026 reach different conclusions, under
different inputs. On 54 energy datasets, zero-shot foundation models given covariates (weather
forecasts and calendar features) perform best, with Chronos-2 ahead of tuned tree models trained on
each dataset's full history with the same covariates [@obermeier2026fets]. On simulated solar power
in Texas, zero-shot models given the history alone have high errors; fine-tuning cuts them sharply,
but two transformers trained from scratch on the same year of data match or beat the fine-tuned
models, and the authors conclude that foundation models are not yet a practical off-the-shelf
solution [@zater2026empirical]. This thesis puts the same question to the metered output of rooftop
PV stations, with archived weather forecasts as inputs and one protocol for every model. All
experiments ran on one 16 GB laptop, with no GPU server, so the time and memory that each model
needs are measured on consumer hardware.

## Research questions {#sec:intro-questions}

The thesis asks whether time-series foundation models, zero-shot or fine-tuned, can match or beat
purpose-trained models and simple reference forecasts for day-ahead hourly PV power on consumer
hardware, and at what compute cost. Three foundation models are tested zero-shot: Chronos-2, TimesFM
and TiRex. Only Chronos-2 is also fine-tuned, because it is the only one of the three that reads weather forecasts natively in the software used here (@sec:method-covariates). Four research questions break this
down:

- **RQ1.** How accurate are zero-shot foundation models for day-ahead hourly PV power, compared
  with purpose-trained models and reference forecasts?
- **RQ2.** How much does fine-tuning a foundation model on a station's own history improve on its
  zero-shot forecasts, and at what compute cost?
- **RQ3.** What do weather forecasts add as inputs, and which models can use them?
- **RQ4.** How well calibrated are the models' forecast intervals (does an 80% interval contain 80%
  of the observations?), and how much time and memory does each model need on a laptop?

## Approach and contributions {#sec:intro-contributions}

Every model runs through one evaluation pipeline. Except for two of the simple reference forecasts,
every model predicts the clear-sky index, the ratio of a station's power to the power it would deliver under a cloudless sky, which removes most of the known daily and seasonal cycle from the series
(@sec:background-concepts). The data are six of the 60 stations of a public dataset from a campus in
Hong Kong [@lin2025hkust] (Chapter [-@sec:data]). The models are scored on the same 270 test days and
the same hours by their skill over CLIPER: the relative reduction in RMSE against this reference
forecast, a blend of the long-term average and yesterday's value, which @yang2020verification
recommend for solar forecasting; Chapter [-@sec:method] gives the details. Two weather settings, with
and without archived weather forecasts as inputs, separate what the models learn from the power
history from what the weather forecasts supply.

The thesis contributes:

1. a benchmark that scores reference forecasts, purpose-trained models and foundation models,
   zero-shot and fine-tuned, with one evaluation pipeline on the same test days and hours, never
   lets a model read power measured after the issue time, compares models with confidence intervals
   and significance tests, and traces every run to its configuration;
2. a documented processing of the dataset for forecasting (Chapter [-@sec:data]): corrected time labels for the hourly files, light quality control, a model of each station's clear-sky power learned from its clearest hours, and an analysis of the lead times of the archived weather forecasts;
3. an account of the time and memory that each model needs on one 16 GB laptop.

In brief, with weather forecasts three models lead and cannot be told apart: LightGBM at default
settings, the fine-tuned Chronos-2 and the best zero-shot configuration of Chronos-2, chosen after
the results had been seen: its larger checkpoint (the larger of the two released sets of pretrained
weights) with a long context window (the stretch of recent history that the model reads). The
zero-shot configuration fixed in advance, the small checkpoint with a short context, trails them.
Fine-tuning adds 0.14 in skill to that configuration, enough to join the leaders; a longer context
alone gives about two thirds of that gain without any training, and fine-tuning at a long context did
not fit in the laptop's memory at the batch size used, so it was not tested. Without weather
forecasts, no model beats CLIPER, so the weather forecasts, not the model class, supply nearly all the
skill of the models that can use them; this gain is an upper bound, because the archived forecasts
come from weather-model runs not yet available at the issue time (@sec:data-weather). With weather forecasts,
the fine-tuned model's forecast intervals are the closest to calibrated, and every model needs at
most about four minutes on the laptop per station and 30 days of test forecasts.

## Outline {#sec:intro-outline}

Chapter [-@sec:background] introduces the concepts that the later chapters rely on and reviews
related work, Chapter [-@sec:data] presents the data and Chapter [-@sec:method] the method. Chapter
[-@sec:results] answers the four questions, Chapter [-@sec:discussion] interprets the answers and
their limits, and Chapter [-@sec:conclusion] concludes.
