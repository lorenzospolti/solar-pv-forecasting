# Convenience targets. Assumes the `pv` conda env is active.
.PHONY: help env install check test smoke lint clean reproduce

help:
	@grep -E '^[a-zA-Z_-]+:.*?# .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?# "}{printf "  %-12s %s\n", $$1, $$2}'

env: # create the conda env from environment.yml
	mamba env create -f environment.yml || conda env create -f environment.yml

install: # editable install of the pvfc package + extras
	pip install -e ".[forecast,dev]"

check: # toolchain smoke test (versions, MPS, pvlib, lightgbm)
	python experiments/check_toolchain.py

test: # run the pytest harness smoke tests
	pytest -q

smoke: # end-to-end harness run on a synthetic station (no data download)
	python experiments/run.py --config configs/experiment.yaml --synthetic

lint: # ruff
	ruff check src experiments tests

clean: # remove caches and the results store
	rm -rf .pytest_cache **/__pycache__ results/results.parquet

reproduce: # regenerate all results from raw (filled in as the project grows)
	python experiments/run.py --config configs/experiment.yaml
