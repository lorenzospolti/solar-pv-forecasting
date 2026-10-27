"""pvfc — PV day-ahead forecasting benchmark harness.

A single evaluation harness behind which every model (persistence, smart
persistence, SARIMA, LightGBM, LSTM, foundation models) is run on *identical*
time-aware splits, normalization, and metrics. See the thesis plan for the
five core abstractions: Forecaster, Splitter, Normalizer, Evaluator, Runner.
"""

__version__ = "0.0.1"
