#!/usr/bin/env bash
# Regenerate everything from the raw MESA runs (strict mode: no embedded fallback tables).
# Approximate wall time on 2 CPU cores: simulation ~2 min, figures ~3 min, tables ~10 min.
set -euo pipefail
cd "$(dirname "$0")"
python3 ztf_j1239_simulations.py --strict --out results/ztf_j1239_simulation_results.json
python3 ztf_j1239_figures.py --strict --outdir figures
python3 make_tables.py
python3 tools/secular_survival_check.py > results/secular_survival.txt
pytest -q tests
(cd paper && pdflatex -interaction=nonstopmode ZTF_J1239_paper_revised.tex >/dev/null && pdflatex -interaction=nonstopmode ZTF_J1239_paper_revised.tex >/dev/null) || echo "pdflatex not available: skipped paper build"
