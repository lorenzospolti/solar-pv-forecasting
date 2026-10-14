# Convenience targets. Assumes the `pv` conda env is active.
.PHONY: help env install checkpoints check test smoke lint clean reproduce data nwp-cache check-data report lock panel manuscript rescore

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

report: # build thesis tables + figures from results/predictions, results/extra/predictions and results/covariates/predictions (timing and memory from the three results stores)
	python experiments/report.py

# One tectonic process per TeX pass, repeated until the .aux settles. Within one process tectonic
# keeps the glyph sizes it cached under each font number, the numbers shift between passes, and on
# the last pass lines with italics came out too tall, so their line spacing was uneven.
manuscript: # build thesis/build/thesis.pdf from thesis/chapters/*.md (pandoc + pandoc-crossref + citeproc -> thesis/build/thesis.tex -> tectonic)
	mkdir -p thesis/build && cd thesis && pandoc chapters/*.md --lua-filter filters/include.lua --filter pandoc-crossref --citeproc \
		--top-level-division=chapter --number-sections -V documentclass=report -V geometry:margin=2.5cm \
		-V fontsize=11pt -V linkcolor=blue --standalone -o build/thesis.tex
	cd thesis && for pass in 1 2 3 4; do \
		cp -f build/thesis.aux build/thesis.aux.prev 2>/dev/null || rm -f build/thesis.aux.prev; \
		tectonic -Z search-path=. --reruns 0 --keep-intermediates build/thesis.tex || exit 1; \
		if cmp -s build/thesis.aux build/thesis.aux.prev; then break; fi; \
	done
