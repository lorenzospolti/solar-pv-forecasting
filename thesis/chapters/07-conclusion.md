# Conclusion {#sec:conclusion}

<!-- TODO(Lorenzo): drafted with Claude on 5 Oct 2026; rewrite it in your own words (UniPV AI
guidelines). Keep it consistent with the Discussion once that is written. -->

Time-series foundation models can match purpose-trained models for day-ahead hourly PV power on a
laptop, on two conditions. First, they need weather forecasts as inputs: without them, no model of
either kind beats CLIPER, a blend of climatology and yesterday's value, while with them the leading
models gain about 0.27 in skill; TiRex and TimesFM, given them through a linear regression added
after the main run, reach about 0.25, behind LightGBM. Second, the configuration matters. The small
Chronos-2 checkpoint at the context fixed in advance trails LightGBM, whereas the base checkpoint with a long context,
chosen after the main run, cannot be told apart from LightGBM or from the fine-tuned model and needs
no training, about 12 seconds per 30-day test window. Fine-tuning closes the gap at the short
context for about three minutes per window, the same as fitting LightGBM, and, with weather
forecasts, gives the best calibrated forecast intervals. These answers rest on one campus, on
archived weather forecasts with shorter lead times than in operation, and on 270 test days that also
informed several choices; they are a comparison under one protocol and an upper bound for the
weather inputs, not a final ranking.

Four extensions would test them. Fine-tuning at a long context needs gradient accumulation to fit
in 16 GB. A true day-ahead test needs weather forecasts archived at fixed lead times. Newer
foundation models that take covariates natively, such as TiRex-2 [@podest2026tirex2] and TimesFM 3.0, may
change the zero-shot picture. Finally, tuned purpose-trained baselines and more sites, in other
climates, would show how far the ranking holds.
