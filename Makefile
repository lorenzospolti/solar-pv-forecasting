# Convenience targets. Assumes the `pv` conda env is active.
.PHONY: help env install checkpoints check test smoke lint clean reproduce data nwp-cache check-data report lock panel rescore

help:
	@grep -E '^[a-zA-Z_-]+:.*?# .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?# "}{printf "  %-12s %s\n", $$1, $$2}'

env: # create the conda env from environment.yml
	mamba env create -f environment.yml || conda env create -f environment.yml

install: # editable install of the pvfc package + extras (python -m pip: bare pip may be another Python)
	python -m pip install -e ".[forecast,dev]"

checkpoints: # pre-download the foundation-model weights (Chronos-2-small, TimesFM-2.5, TiRex)
	python experiments/fetch_checkpoints.py

lock: # snapshot the exact environment after any install/upgrade
	python -m pip freeze --exclude-editable > requirements-lock.txt

check: # toolchain smoke test (versions, MPS, pvlib, lightgbm)
	python experiments/check_toolchain.py

test: # run the pytest harness smoke tests
	python -m pytest -q

smoke: # end-to-end harness run on a synthetic station (no data download)
	python experiments/run.py --config configs/experiment.yaml --synthetic

data: # extract + inspect the HKUST Dryad dataset (download Dataset.zip into data/raw first)
	python experiments/get_data.py

nwp-cache: # pre-fetch + cache the Open-Meteo archived-forecast weather for HKUST
	python -c "from pvfc.weather import fetch_open_meteo; from pvfc.data import HKUST_LAT, HKUST_LON; fetch_open_meteo(HKUST_LAT, HKUST_LON, '2021-01-01', '2023-12-31')"

check-data: # verify the real-data ETL pipeline (requires `make data`)
	python experiments/check_data.py

lint: # ruff
	ruff check src experiments tests

clean: # remove caches and the results store
	rm -rf .pytest_cache **/__pycache__ results/results.parquet

panel: # overnight batch: every panel station x weather setting x model (resumable)
	python experiments/run_panel.py

reproduce: # regenerate every result from raw data: checkpoints -> panel -> rescore -> tables/figures
	python experiments/fetch_checkpoints.py && python experiments/run_panel.py && python experiments/rescore.py && python experiments/report.py

rescore: # rebuild every metric row of results/results.parquet from results/predictions (current scoring rules); preview: python experiments/rescore.py --dry-run
	python experiments/rescore.py

report: # build thesis tables + figures from results/predictions (timing and memory from results/results.parquet)
	python experiments/report.py
