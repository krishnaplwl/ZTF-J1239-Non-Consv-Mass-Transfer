#!/usr/bin/env python3
"""Secular-survival check of the paper claim that the system survives Gyr at high Mdot. Run from repo root:
   python3 tools/secular_survival_check.py > results/secular_survival.txt"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, ztf_j1239_simulations as sim
ds=sim.load_or_build_age_grid_dataset(strict=True); zfn,rfn,_=sim.make_age_interpolators(ds)
print("candidate  f   channels/eta   -> fate, lifetime[Gyr], final Md, final P[min], Mdot_start, Mdot_at_1Gyr")
for (Md,Ma,fs) in ((55,80,(0.1,0.3)),(50,70,(0.15,0.3)),(52,70,(0.2,0.4))):
    for f in fs:
        for ch,eta in ((("GW",),0),(("GW","MB"),0.05),(("GW","MB"),0.15),(("GW","MB"),1.0)):
            r=sim.integrate_secular_evolution(Md,Ma,f,"isotropic",lambda m: zfn(m,3.0),Ra_J=rfn(Ma,3.0),eta_MB=eta,channels=ch,t_max_gyr=13.8)
            t=r["t_yr"]; i1=np.searchsorted(t,1e9)
            print(f" {Md}/{Ma} f={f} {'GW' if eta==0 else 'GW+MB '+str(eta):10s}-> {r['fate'][:22]:22s} life={r['lifetime_yr']/1e9:6.3f} Gyr Md_end={r['Md'][-1]:5.1f} P_end={r['P_min'][-1]:6.1f} min Mdot0={abs(r['Mdot'][0]):.2e} Mdot(1Gyr)={abs(r['Mdot'][min(i1,len(t)-1)]):.2e}")
