# Notes for Claude

Thesis code: day-ahead PV forecasting, time-series foundation models vs purpose-trained
models. Working branch: scoring-report-oct. The LaTeX thesis, the defense deck and notes
are in the private repo lorenzospolti/thesis-writing.

## Cloud sessions (CLAUDE_CODE_REMOTE=true)
- Dependencies come from the environment setup script (CPU-only torch). Before running
  scripts: uv pip install --system --break-system-packages -e . --no-deps
- 4 vCPUs, 16 GB RAM, no GPU. Hugging Face and Open-Meteo are reachable; Dryad is not.
- Not in the clone (gitignored): data/, results/predictions, results/extra, results/archive.
  make panel / reproduce / report and the deck extractors run on the Mac, not here.
- Fine here: make test, make smoke, make lint, code and chapter edits.
