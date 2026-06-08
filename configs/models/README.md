# `configs/models/`

Reserved for per-model config fragments if/when the experiment grows beyond a
single composed file. **The active model ladder is currently inline** in
[`../experiment.yaml`](../experiment.yaml) under the `models:` key — that is the
single source of truth the runner reads.

Keep model hyper-parameters there (so the whole config hashes into one
reproducible `config_hash`) until there is a concrete need to split them out.
