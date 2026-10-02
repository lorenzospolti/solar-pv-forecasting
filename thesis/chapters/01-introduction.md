# Introduction {#sec:intro}

*Scope note: motivate PV forecasting (grid integration, day-ahead markets); the rise
of pre-trained time-series foundation models; the open question of whether they pay
off against well-tuned baselines on a single, realistic site at low cost. State the
contribution: a single open benchmark harness that scores every model under
identical splits/masking/metrics, on a real multi-year campus dataset, on a laptop.*

Contributions:

1. A reproducible, config-hashed benchmark harness in which every model is a
   `Forecaster` scored by one shared `Runner` over identical causal folds, masking,
   and metrics.
2. A like-for-like comparison of baselines, classical/ML/deep models, and foundation
   models (zero-shot and fine-tuned) for day-ahead hourly PV power.
3. A compute-cost accounting (wall-clock and memory) framing accuracy *per unit of
   compute* on an Apple M4 (16 GB).
