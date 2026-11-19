#!/usr/bin/env python3
"""Regenerate the embedded FALLBACK_* tables of ztf_j1239_simulations.py from the raw MESA runs.

Run from the repository root:  python3 tools/regen_fallback_tables.py --write
Without --write the new blocks are only printed. The fallback tables are used ONLY when ./runs is absent;
the analysis scripts warn (and `--strict` refuses) in that case.
"""
import argparse, re, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ztf_j1239_simulations as sim

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    ds = sim.load_or_build_age_grid_dataset(strict=True)
    snap = ds["snap_table"]
    L = ["FALLBACK_SNAPSHOT_TABLE: Dict[str, Dict[int, Dict[str, float]]] = {"]
    for tag in sim.AGE_TAGS:
        L.append(f'    "{tag}": {{')
        for m in sim.MASSES_MJUP:
            r = snap[tag][m]
            L.append(f'        {m}: {{"R": {r["R"]:.4f}, "rho": {r["rho"]:.4f}, "LH_L": {r["LH_L"]:.4f}, "Teff": {r["Teff"]:.1f}}},')
        L.append("    },")
    L.append("}")
    blk1 = "\n".join(L)
    nodes = sim.FALLBACK_ZETA_NODES_MJUP
    L = ["FALLBACK_AGE_ZETA_AD: Dict[float, List[float]] = {"]
    for ag in sim.AGES_GYR:
        cen, med = ds["age_zeta_ad"][ag][0], ds["age_zeta_ad"][ag][1]
        ok = np.isfinite(med)
        vals = np.interp(nodes, cen[ok], med[ok])
        L.append(f"    {ag}: [" + ", ".join(f"{v:.3f}" for v in vals) + "],")
    L.append("}")
    blk2 = "\n".join(L)
    print(blk1); print(blk2)
    if a.write:
        p = ROOT / "ztf_j1239_simulations.py"; s = p.read_text()
        s = re.sub(r"FALLBACK_SNAPSHOT_TABLE: .*?\n\}\n", blk1 + "\n", s, count=1, flags=re.S)
        s = re.sub(r"FALLBACK_AGE_ZETA_AD: .*?\n\}\n", blk2 + "\n", s, count=1, flags=re.S)
        p.write_text(s); print("patched", p)
if __name__ == "__main__":
    main()
