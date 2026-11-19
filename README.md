# ZTF J1239+8347: non-conservative mass transfer in a brown-dwarf binary

Code, MESA run data, figures, tables and manuscript for
*"Can Non-Conservative Mass Transfer Stabilise the Brown-Dwarf Binary ZTF J1239+8347?"* (K. Paliwal).

## Layout

| Path | Contents |
|------|----------|
| `ztf_j1239_simulations.py` | Main inlist creation and code for MESA|
| `ztf_j1239_figures.py` | The six figures |
| `make_tables.py` | Every table of the paper |
| `tools/` | `secular_survival_check.py`, `regen_fallback_tables.py` |
| `runs/` | 266 MESA run directories |
| `models/` | 175 MESA `.mod` start models |
| `out/` | Cached reductions of the runs (`age_grid_curves.json`, `rate_ladder.json`, `zeta_model.json`, `zeta_variants.json`, `cond_compare.json`) |
| `data/COND03_1Gyr.csv` | Baraffe et al. (2003) COND03 1 Gyr benchmark |
| `results/` | Stored outputs: simulation JSON, tables JSON, secular-survival log |
| `figures/` | Figures 1–6 as generated |
| `tests/` | `pytest` suite |

## Reproduce

```bash
pip install -r requirements.txt
./run_all.sh            # ~15 min on 2 cores: simulations, figures, tables, lifetime check, tests, PDF
# or step by step
python3 ztf_j1239_simulations.py --strict --out results/ztf_j1239_simulation_results.json
python3 ztf_j1239_figures.py --strict --outdir figures
python3 make_tables.py            # add --skip-qmax to skip the slow q_max scan (Appendix Table A2)
pytest -q tests
```

`--strict` makes the code fail if `runs/` is missing instead of silently falling back to embedded tables
(the fallback values are kept in the code only so that the module imports standalone, and a loud warning is printed when used).
Provenance (how many run histories vs. cached JSON were used) is written to the `provenance` field of the results JSON.

Regenerating the MESA runs themselves needs MESA r25.12.1: `python3 ztf_j1239_simulations.py --mode mesa-gen --mesa-runs <dir>` writes the inlists.
