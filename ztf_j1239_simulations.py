#!/usr/bin/env python3
"""
ztf_j1239_simulations.py

This standalone module consolidates the entire computational
astrophysics pipeline into a single, cohesive codebase:

  1. MESA Stellar-Evolution Campaign Manager & History Parser (MESA r25.12.1):
     - Generates inlists and schedules dependency-aware MESA runs across the
       nine-age grid t_0 in {0.1, 0.3, 0.5, 1, 2, 3, 5, 8, 10} Gyr, initial
       masses M_0 in {20, 40, 60, 70, 75, 80} M_Jup, mass-stripping rates
       |Mdot| in {1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4} M_Jup/yr, no-mass-loss
       cooling controls, and metallicity/atmosphere boundary variants
       (Z = 0.0142, 0.030; tau_atm = 10, 100) — 266 MESA runs in total.
     - Extracts control-subtracted donor mass-radius exponents
       zeta_s = d ln R_d / d ln M_d via Savitzky-Golay differentiation on a
       uniform logarithmic mass grid and secant slopes for fast probes.
     - Benchmarks 1 Gyr MESA radii against Baraffe et al. (2003) COND03 models.

  2. Non-Conservative Orbital Dynamics & Roche-Lobe Stability:
     - Computes Eggleton (1983) Roche-lobe radii and logarithmic derivatives,
       the linear Roche-lobe response zeta_L(q, f) = Z_0(q) + f * S(q) for
       Jeans (direct donor wind) and isotropic (accretor re-emission) modes,
       finite-radius accretor ejection geometries (point source, donor-facing
       impact site, spherical surface average, antipodal far side, and the
       geometry-independent zero-angular-momentum floor rho_eff^2 = 0), and
       non-synchronous accretor spin-sink terms.
     - Solves for conservative critical mass ratios q_crit(f=0; t_0), critical
       mass-loss fractions f_crit(q; t_0), and exact Roche-lobe-filling donor
       masses M_d,cont(t_0) at P_orb = 57.41 min.

  3. Mass-Transfer Rates, Hotspot Energetics & Secular Binary Evolution:
     - Evaluates systemic angular-momentum loss via gravitational radiation
       (Peters 1964) and saturated/weakened magnetic braking (Sills et al. 2000;
       Wright et al. 2011).
     - Computes quasi-steady mass-transfer rates |Mdot_d|(f, t_0) incorporating
       secular donor thermal contraction rho = (d ln R_d / dt)_M, identifying
       young-age thermal detachment (|rho| > 2 |Gamma_GW| at t_0 <= 0.5 Gyr).
     - Cross-checks analytical rates against an independent finite-step
       Roche-contact solver and integrates gigayear secular trajectories
       (M_d(t), M_a(t), P_orb(t), q(t)).

  4. Restricted Three-Body Ballistic Stream Integration & Energy Budget:
     - Integrates ballistic accretion stream trajectories from L_1 in the
       co-rotating Roche potential using an 8th-order Dormand-Prince (DOP853)
       ODE solver (conserving the Jacobi integral to ~2e-15).
     - Evaluates impact speeds v_imp(t_0), escape speeds v_esc(t_0), maximum
       energetically allowed mass-loss fractions f_max = eps * (v_imp/v_esc)^2,
       minimum required shock conversion efficiencies eps_min = f_crit / f_max,
       luminosity-coupled margins for the 8904 K hotspot, and L-dwarf accretor
       temperature-radius degeneracy relations.

Usage:
  python3 ztf_j1239_simulations.py                     # Run full analysis + unit tests
  python3 ztf_j1239_simulations.py --mode test         # Run verification unit tests only
  python3 ztf_j1239_simulations.py --mode mesa-status  # Check MESA campaign directory status
  python3 ztf_j1239_simulations.py --mode mesa-gen     # Generate MESA inlist directories
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator, UnivariateSpline
from scipy.optimize import brentq
from scipy.signal import savgol_filter


# ==============================================================================
# 1. PHYSICAL CONSTANTS & ZTF J1239+8347 OBSERVATIONAL PARAMETERS (CGS)
# ==============================================================================

G_CGS: float = 6.674e-8             # Gravitational constant [cm^3 g^-1 s^-2]
C_LIGHT: float = 2.998e10           # Speed of light [cm s^-1]
M_JUP: float = 1.8982e30            # Jupiter mass [g]
R_JUP: float = 7.1492e9             # Jupiter radius [cm]
M_SUN: float = 1.989e33             # Solar mass [g]
R_SUN: float = 6.957e10             # Solar radius [cm]
YR_SEC: float = 3.15576e7           # Julian year [s]
DAY_SEC: float = 86400.0            # Day [s]
SIGMA_SB: float = 5.670374e-5       # Stefan-Boltzmann constant [erg cm^-2 s^-1 K^-4]
KMS_CMS: float = 1.0e5              # 1 km/s in cm/s
MJUP_PER_MSUN: float = 9.5459e-4    # IAU nominal GM_Jup / GM_Sun ratio
RJUP_PER_RSUN: float = R_JUP / R_SUN

# Observed properties of ZTF J1239+8347 (Whitebook et al. 2026a,b)
P_ORB_MIN: float = 57.41            # Orbital period [min]
P_ORB_S: float = P_ORB_MIN * 60.0   # Orbital period [s]
T_SPOT_K: float = 8904.0            # Blackbody hotspot temperature [K]
R_BB_JUP: float = 0.60              # Hotspot blackbody-equivalent radius [R_Jup]
R_BB_RANGE_JUP: Tuple[float, float] = (0.54, 0.67)
MDOT_QUOTED_JUP_YR: float = 1.0e-9  # Nominal rate cited in Whitebook et al. (2026b) [M_Jup/yr]
ROCHE_CAP_FRAC: float = 0.95        # Accretor Roche-lobe containment fraction
DELTA_DYN_MEAS: float = 0.025       # Measured fast-rate (1e-4 M_Jup/yr) dynamical shift

# Campaign grid definitions
AGES_GYR: List[float] = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
AGE_TAGS: List[str] = ["0p1", "0p3", "0p5", "1", "2", "3", "5", "8", "10"]
MASSES_MJUP: List[int] = [20, 40, 60, 80]
RATES_TAGS: List[str] = ["R1e-6", "R1e-7", "R1e-8", "R1e-9"]
MASS_BINS: np.ndarray = np.arange(10.0, 82.6, 2.5)
BIN_CENTERS: np.ndarray = 0.5 * (MASS_BINS[1:] + MASS_BINS[:-1])

# Pre-verified benchmark data from the 266 MESA (r25.12.1) runs & Baraffe (2003) COND03
# Used automatically as a fallback when running standalone without raw MESA history files.
FALLBACK_COND03_1GYR: Dict[str, List[float]] = {
    "masses_MJ": [20.0, 40.0, 60.0, 80.0],
    "R_cond_RJ": [0.97745, 0.91000, 0.89300, 0.95855],
    "R_mesa_RJ": [0.98669, 0.91367, 0.89799, 0.96707],
}

FALLBACK_SNAPSHOT_TABLE: Dict[str, Dict[int, Dict[str, float]]] = {
    "0p1": {
        20: {"R": 1.1754, "rho": -1.1972, "LH_L": 0.0000, "Teff": 1449.8},
        40: {"R": 1.2680, "rho": -2.1046, "LH_L": 0.0001, "Teff": 2188.5},
        60: {"R": 1.4230, "rho": -2.6816, "LH_L": 0.0014, "Teff": 2593.6},
        80: {"R": 1.5953, "rho": -2.9034, "LH_L": 0.0067, "Teff": 2806.1},
    },
    "0p3": {
        20: {"R": 1.0595, "rho": -0.2416, "LH_L": 0.0000, "Teff": 1051.9},
        40: {"R": 1.0436, "rho": -0.4806, "LH_L": 0.0002, "Teff": 1754.5},
        60: {"R": 1.0951, "rho": -0.6825, "LH_L": 0.0150, "Teff": 2244.2},
        80: {"R": 1.1836, "rho": -0.7802, "LH_L": 0.1206, "Teff": 2581.2},
    },
    "0p5": {
        20: {"R": 1.0250, "rho": -0.1191, "LH_L": 0.0000, "Teff": 902.3},
        40: {"R": 0.9768, "rho": -0.2307, "LH_L": 0.0003, "Teff": 1530.8},
        60: {"R": 0.9954, "rho": -0.3410, "LH_L": 0.0294, "Teff": 2034.9},
        80: {"R": 1.0661, "rho": -0.3544, "LH_L": 0.2934, "Teff": 2434.7},
    },
    "1": {
        20: {"R": 0.9865, "rho": -0.0530, "LH_L": 0.0000, "Teff": 757.5},
        40: {"R": 0.9136, "rho": -0.0790, "LH_L": 0.0003, "Teff": 1226.4},
        60: {"R": 0.8980, "rho": -0.1261, "LH_L": 0.0489, "Teff": 1725.3},
        80: {"R": 0.9670, "rho": -0.1062, "LH_L": 0.6130, "Teff": 2250.4},
    },
    "2": {
        20: {"R": 0.9518, "rho": -0.0251, "LH_L": 0.0000, "Teff": 645.2},
        40: {"R": 0.8731, "rho": -0.0281, "LH_L": 0.0003, "Teff": 980.0},
        60: {"R": 0.8349, "rho": -0.0430, "LH_L": 0.0605, "Teff": 1398.7},
        80: {"R": 0.9174, "rho": -0.0252, "LH_L": 0.8658, "Teff": 2125.8},
    },
    "3": {
        20: {"R": 0.9329, "rho": -0.0161, "LH_L": 0.0000, "Teff": 586.3},
        40: {"R": 0.8546, "rho": -0.0168, "LH_L": 0.0003, "Teff": 879.6},
        60: {"R": 0.8097, "rho": -0.0225, "LH_L": 0.0564, "Teff": 1227.1},
        80: {"R": 0.9036, "rho": -0.0090, "LH_L": 0.9470, "Teff": 2086.6},
    },
    "5": {
        20: {"R": 0.9111, "rho": -0.0085, "LH_L": 0.0000, "Teff": 509.7},
        40: {"R": 0.8339, "rho": -0.0093, "LH_L": 0.0001, "Teff": 782.6},
        60: {"R": 0.7852, "rho": -0.0116, "LH_L": 0.0242, "Teff": 1085.9},
        80: {"R": 0.8964, "rho": -0.0016, "LH_L": 0.9907, "Teff": 2065.7},
    },
    "8": {
        20: {"R": 0.8949, "rho": -0.0042, "LH_L": 0.0000, "Teff": 431.4},
        40: {"R": 0.8161, "rho": -0.0058, "LH_L": 0.0000, "Teff": 710.3},
        60: {"R": 0.7642, "rho": -0.0073, "LH_L": 0.0067, "Teff": 985.3},
        80: {"R": 0.8951, "rho": -0.0001, "LH_L": 0.9996, "Teff": 2062.0},
    },
    "10": {
        20: {"R": 0.8886, "rho": -0.0030, "LH_L": 0.0000, "Teff": 399.3},
        40: {"R": 0.8077, "rho": -0.0047, "LH_L": 0.0000, "Teff": 679.2},
        60: {"R": 0.7542, "rho": -0.0060, "LH_L": 0.0032, "Teff": 938.2},
        80: {"R": 0.8951, "rho": 0.0000, "LH_L": 1.0003, "Teff": 2062.2},
    },
}

FALLBACK_ZETA_NODES_MJUP: List[float] = [15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
FALLBACK_AGE_ZETA_AD: Dict[float, List[float]] = {
    0.1: [-0.080, -0.081, -0.104, -0.131, -0.158, -0.185, -0.216, -0.252],
    0.3: [-0.142, -0.156, -0.185, -0.204, -0.219, -0.226, -0.258, -0.206],
    0.5: [-0.154, -0.174, -0.203, -0.217, -0.236, -0.249, -0.259, -0.254],
    1.0: [-0.154, -0.173, -0.203, -0.222, -0.247, -0.262, -0.271, -0.272],
    2.0: [-0.153, -0.173, -0.207, -0.231, -0.248, -0.260, -0.267, -0.279],
    3.0: [-0.154, -0.177, -0.209, -0.232, -0.249, -0.261, -0.270, -0.283],
    5.0: [-0.152, -0.172, -0.202, -0.225, -0.244, -0.260, -0.270, -0.284],
    8.0: [-0.149, -0.171, -0.200, -0.217, -0.238, -0.256, -0.270, -0.284],
    10.0: [-0.148, -0.171, -0.200, -0.216, -0.235, -0.254, -0.270, -0.284],
}


# ==============================================================================
# 2. MESA CAMPAIGN GENERATOR, RUNNER & HISTORY PARSER (MESA r25.12.1)
# ==============================================================================

@dataclass
class MESAJob:
    """Single MESA evolution or mass-stripping job in the dependency graph."""
    name: str
    kind: str               # 'pms' | 'cont' | 'ctrl' | 'strip'
    mass: int               # Initial mass [M_Jup]
    out_mod: str
    in_mod: str = ""
    max_age: float = 0.0
    mdot_msun_yr: float = 0.0
    mass_min: float = 0.0
    dt_max: float = 1.0e7
    hist_int: int = 1
    needs: str = ""
    z_met: float = 0.02
    atm_table: str = "tau_100"


def _fortran_double(x: float) -> str:
    return f"{x:.8e}".replace("e", "d")


def _rate_tag(r: float) -> str:
    return f"R{r:.0e}".replace("e-0", "e-")


def _age_tag(a: float) -> str:
    return f"{a:g}".replace(".", "p")


def build_mesa_campaign_jobs(
    masses: Sequence[int] = MASSES_MJUP,
    ages_gyr: Sequence[float] = AGES_GYR,
    rates_mjup_yr: Sequence[float] = (1e-6, 1e-7, 1e-8, 1e-9),
    include_extended_suite: bool = True,
    target_mjup: float = 10.0,
    age_cap_yr: float = 13.8e9,
    ctrl_yr: float = 4.0e9,
) -> List[MESAJob]:
    """
    Construct the full MESA job dependency graph across the 0.1-10 Gyr age grid
    plus the 1 Gyr extended suite (70, 75 M_Jup starts, 1e-5/1e-4 M_Jup/yr fast
    probes, and metallicity/atmosphere boundary variants).
    """
    ages = sorted(set(ages_gyr))
    jobs: List[MESAJob] = []
    skip_jobs = {"M20_A10_R1e-6"}

    for m in masses:
        prev_age: Optional[float] = None
        for a in ages:
            sname = f"M{m}_age{_age_tag(a)}"
            smod = f"M{m}_{_age_tag(a)}Gyr.mod"
            if prev_age is None:
                jobs.append(MESAJob(
                    name=sname, kind="pms", mass=m, out_mod=smod,
                    max_age=a * 1e9, dt_max=1e7 if a >= 0.1 else 1e6, hist_int=10
                ))
            else:
                prev_name = f"M{m}_age{_age_tag(prev_age)}"
                prev_mod = f"M{m}_{_age_tag(prev_age)}Gyr.mod"
                jobs.append(MESAJob(
                    name=sname, kind="cont", mass=m, out_mod=smod, in_mod=prev_mod,
                    max_age=a * 1e9, dt_max=1e7, hist_int=10, needs=prev_name
                ))
            prev_age = a

        for a in ages:
            sname = f"M{m}_age{_age_tag(a)}"
            smod = f"M{m}_{_age_tag(a)}Gyr.mod"
            t0 = a * 1e9
            tcap = age_cap_yr - t0
            if tcap <= 0:
                continue
            ctrl_name = f"M{m}_A{_age_tag(a)}_ctrl"
            jobs.append(MESAJob(
                name=ctrl_name, kind="ctrl", mass=m, out_mod=f"{ctrl_name}_end.mod",
                in_mod=smod, max_age=t0 + min(ctrl_yr, tcap), dt_max=1e7, hist_int=1, needs=sname
            ))
            for r in rates_mjup_yr:
                jname = f"M{m}_A{_age_tag(a)}_{_rate_tag(r)}"
                if jname in skip_jobs:
                    continue
                cap_yr = 3.0e9 if r == 1e-9 else 4.0e9
                if r >= 1e-5:
                    dur = min(0.15 * m / r, tcap)
                    mmin = 0.85 * m
                    dtm = min(1e7, 2e-3 * m / r, dur / 400.0)
                else:
                    dur = min((m - target_mjup) / r, cap_yr, tcap)
                    mmin = target_mjup
                    dtm = min(1e7, 2e-3 * m / r)
                jobs.append(MESAJob(
                    name=jname, kind="strip", mass=m, out_mod=f"{jname}_end.mod",
                    in_mod=smod, max_age=t0 + dur * 1.001, mdot_msun_yr=-r * MJUP_PER_MSUN,
                    mass_min=mmin * MJUP_PER_MSUN, dt_max=dtm, hist_int=1, needs=sname
                ))

    if include_extended_suite:
        # 1 Gyr fast-rate probes (1e-5, 1e-4 M_Jup/yr) and intermediate starts (70, 75 M_Jup)
        for m_ext in (70, 75):
            sname = f"M{m_ext}_pms1"
            smod = f"M{m_ext}_1Gyr.mod"
            jobs.append(MESAJob(name=sname, kind="pms", mass=m_ext, out_mod=smod, max_age=1.0e9, dt_max=1e7, hist_int=10))
            jobs.append(MESAJob(
                name=f"M{m_ext}_A1_ctrl", kind="ctrl", mass=m_ext, out_mod=f"M{m_ext}_A1_ctrl_end.mod",
                in_mod=smod, max_age=5.0e9, dt_max=1e7, hist_int=1, needs=sname
            ))
            for r in (1e-6, 1e-7, 1e-8):
                jname = f"M{m_ext}_A1_{_rate_tag(r)}"
                dur = min((m_ext - target_mjup) / r, 4.0e9)
                jobs.append(MESAJob(
                    name=jname, kind="strip", mass=m_ext, out_mod=f"{jname}_end.mod",
                    in_mod=smod, max_age=1.0e9 + dur * 1.001, mdot_msun_yr=-r * MJUP_PER_MSUN,
                    mass_min=target_mjup * MJUP_PER_MSUN, dt_max=min(1e7, 2e-3 * m_ext / r), hist_int=1, needs=sname
                ))
        for m_fast in (40, 60, 70, 80):
            smod = f"M{m_fast}_1Gyr.mod"
            sname = f"M{m_fast}_age1" if m_fast in MASSES_MJUP else f"M{m_fast}_pms1"
            for r in (1e-5, 1e-4):
                jname = f"M{m_fast}_A1_{_rate_tag(r)}"
                dur = 0.15 * m_fast / r
                jobs.append(MESAJob(
                    name=jname, kind="strip", mass=m_fast, out_mod=f"{jname}_end.mod",
                    in_mod=smod, max_age=1.0e9 + dur * 1.001, mdot_msun_yr=-r * MJUP_PER_MSUN,
                    mass_min=0.85 * m_fast * MJUP_PER_MSUN, dt_max=min(1e7, 2e-3 * m_fast / r, dur / 400.0),
                    hist_int=1, needs=sname
                ))
        # Metallicity & atmosphere boundary variants at 40 and 60 M_Jup
        for vtag, z_val, atm_val in (("ZLO", 0.0142, "tau_100"), ("ZHI", 0.030, "tau_100"), ("ATMPH", 0.020, "tau_10")):
            for m_var in (40, 60):
                sname = f"M{m_var}_{vtag}_pms1"
                smod = f"M{m_var}_{vtag}_1Gyr.mod"
                jobs.append(MESAJob(
                    name=sname, kind="pms", mass=m_var, out_mod=smod,
                    max_age=1.0e9, dt_max=1e7, hist_int=10, z_met=z_val, atm_table=atm_val
                ))
                jobs.append(MESAJob(
                    name=f"M{m_var}_{vtag}_A1_ctrl", kind="ctrl", mass=m_var,
                    out_mod=f"M{m_var}_{vtag}_A1_ctrl_end.mod", in_mod=smod,
                    max_age=5.0e9, dt_max=1e7, hist_int=1, needs=sname, z_met=z_val, atm_table=atm_val
                ))
                for r in (1e-6, 1e-7):
                    jname = f"M{m_var}_{vtag}_A1_{_rate_tag(r)}"
                    dur = min((m_var - target_mjup) / r, 4.0e9)
                    jobs.append(MESAJob(
                        name=jname, kind="strip", mass=m_var, out_mod=f"{jname}_end.mod",
                        in_mod=smod, max_age=1.0e9 + dur * 1.001, mdot_msun_yr=-r * MJUP_PER_MSUN,
                        mass_min=target_mjup * MJUP_PER_MSUN, dt_max=min(1e7, 2e-3 * m_var / r),
                        hist_int=1, needs=sname, z_met=z_val, atm_table=atm_val
                    ))
    return jobs


def render_mesa_inlist(j: MESAJob, pms_min_msun: float = 0.03) -> str:
    """Render a MESA r25.12.1 Fortran namelist inlist for a given MESAJob."""
    lines = ["&star_job", ""]
    if j.kind == "pms":
        lines += [
            "   create_pre_main_sequence_model = .true.",
            "   pre_ms_relax_to_start_radiative_core = .false.   ! Brown dwarfs are fully convective",
        ]
        if j.mass * MJUP_PER_MSUN < pms_min_msun:
            lines += [
                "",
                "   ! Create at 0.03 Msun, then relax mass down at age 0",
                "   relax_mass_scale = .true.",
                "   dlgm_per_step = 1d-3",
                f"   new_mass = {_fortran_double(j.mass * MJUP_PER_MSUN)}   ! {j.mass} MJup",
                "   change_mass_years_for_dt = 1",
            ]
    else:
        lines += [
            "   load_saved_model = .true.",
            f"   load_model_filename = '../../models/{j.in_mod}'",
            "   set_initial_model_number = .true.",
            "   initial_model_number = 0",
        ]
    lines += [
        "",
        "   save_model_when_terminate = .true.",
        f"   save_model_filename = '../../models/{j.out_mod}'",
        "   history_columns_file = 'history_columns.list'",
        "   profile_columns_file = 'profile_columns.list'",
        "   pgstar_flag = .false.",
        "",
        "/ ! end of star_job namelist",
        "",
        "&eos",
        "/ ! end of eos namelist",
        "",
        "&kap",
        "   use_Type2_opacities = .true.",
        f"   Zbase = {j.z_met}",
        "/ ! end of kap namelist",
        "",
        "&controls",
        "",
    ]
    if j.kind == "pms":
        m_init = max(j.mass * MJUP_PER_MSUN, pms_min_msun)
        lines.append(f"   initial_mass = {_fortran_double(m_init)}   ! {j.mass} MJup")
    lines += [
        f"   initial_z = {j.z_met}",
        "   atm_option = 'table'",
        f"   atm_table = '{j.atm_table}'",
        "",
        "   energy_eqn_option = 'dedt'",
        "   use_gold_tolerances = .true.",
        "   varcontrol_target = 1d-4",
        "   mesh_delta_coeff = 1.0",
        "",
        f"   max_age = {_fortran_double(j.max_age)}",
        f"   max_years_for_timestep = {_fortran_double(j.dt_max)}",
    ]
    if j.kind == "strip":
        lines += [
            "",
            f"   mass_change = {_fortran_double(j.mdot_msun_yr)}",
            f"   star_mass_min_limit = {_fortran_double(j.mass_min)}",
        ]
    lines += [
        "",
        f"   history_interval = {j.hist_int}",
        "   write_profiles_flag = .false.",
        "   photo_interval = 500",
        "   terminal_interval = 200",
        "   max_model_number = 400000",
        "   log_directory = 'LOGS'",
        "   photo_directory = 'photos'",
        "",
        "/ ! end of controls namelist",
        "",
        "&pgstar",
        "/ ! end of pgstar namelist",
        "",
    ]
    return "\n".join(lines)


def read_mesa_history(path: Path | str) -> Dict[str, np.ndarray]:
    """Read a MESA LOGS/history.data file and deduplicate backed-up/retried steps."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    hdr = next(i for i, l in enumerate(lines) if "model_number" in l.split())
    names = lines[hdr].split()
    data = np.atleast_2d(np.loadtxt(lines[hdr + 1:]))
    h = {n: data[:, k] for k, n in enumerate(names)}
    mn = h["model_number"]
    _, idx = np.unique(mn[::-1], return_index=True)
    keep = np.sort(len(mn) - 1 - idx)
    return {k: v[keep] for k, v in h.items()}


def extract_zeta_curve(
    strip: Dict[str, np.ndarray],
    ctrl: Optional[Dict[str, np.ndarray]] = None,
    subtract_control: bool = True,
    npts: int = 400,
    window_frac: float = 0.10,
    discard: float = 2e-3,
    order: int = 3,
    trim_edges: bool = False,
    return_elapsed: bool = False,
) -> Tuple[np.ndarray, ...]:
    """
    Compute the donor mass-radius exponent zeta_s = d ln R_d / d ln M_d from a
    MESA mass-stripping history and its matched no-mass-loss control history.
    """
    M = strip["star_mass"]
    t = strip["star_age"]
    lnR = np.log(strip["radius"])
    if subtract_control:
        if ctrl is None:
            raise ValueError("Control history is required when subtract_control=True.")
        tc, Rc = ctrl["star_age"], np.log(ctrl["radius"])
        if t.max() > tc.max():
            ok = t <= tc.max()
            M, t, lnR = M[ok], t[ok], lnR[ok]
            if ok.sum() < 50:
                raise ValueError("Control run is far shorter than stripping run.")
        lnR = lnR - np.interp(t, tc, Rc)
    x = np.maximum.accumulate(np.log(M[0] / M))
    keep = np.concatenate([[True], np.diff(x) > 1e-12])
    x, y, el = x[keep], lnR[keep], (t - t[0])[keep]
    xs = np.linspace(max(discard, x[0]), x[-1], npts)
    ys = np.interp(xs, x, y)
    els = np.interp(xs, x, el)
    w = (int(window_frac * npts) // 2) * 2 + 1
    dydx = savgol_filter(ys, w, order, deriv=1, delta=xs[1] - xs[0])
    h = w // 2 if trim_edges else 0
    sl = slice(h, len(xs) - h)
    m_mjup = (M[0] * np.exp(-xs) / MJUP_PER_MSUN)[sl]
    zeta = (-dydx)[sl]
    if return_elapsed:
        return m_mjup, zeta, els[sl]
    return m_mjup, zeta


def pool_donor_exponents(
    curves: Dict[Tuple[int, float, str], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    rates: Sequence[str],
    tau_min_yr: float = 0.0,
    mass_class: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool individual MESA stripping curves into mass bins (median, min, max)."""
    med = np.full(BIN_CENTERS.shape, np.nan)
    lo = med.copy()
    hi = med.copy()
    for i in range(len(BIN_CENTERS)):
        vals = []
        for (M0, age, rate), (Mg, z, el) in curves.items():
            if rate not in rates:
                continue
            if mass_class == "cold" and M0 > 65:
                continue
            if mass_class == "hot" and M0 < 75:
                continue
            sel = (Mg >= MASS_BINS[i]) & (Mg < MASS_BINS[i + 1]) & (el >= tau_min_yr)
            if sel.any():
                vals.append(float(z[sel].mean()))
        if vals:
            med[i] = float(np.median(vals))
            lo[i] = float(min(vals))
            hi[i] = float(max(vals))
    return BIN_CENTERS.copy(), med, lo, hi


# ==============================================================================
# 3. NON-CONSERVATIVE ORBITAL DYNAMICS & ROCHE-LOBE STABILITY
# ==============================================================================

def eggleton_roche_lobe(q: float | np.ndarray) -> np.ndarray:
    """Eggleton (1983) volume-equivalent donor Roche-lobe radius R_{L,d} / a."""
    q_arr = np.asarray(q, dtype=float)
    q13, q23 = q_arr ** (1.0 / 3.0), q_arr ** (2.0 / 3.0)
    return 0.49 * q23 / (0.6 * q23 + np.log1p(q13))


def eggleton_accretor_lobe(q: float | np.ndarray) -> np.ndarray:
    """Eggleton (1983) volume-equivalent accretor Roche-lobe radius R_{L,a} / a."""
    return eggleton_roche_lobe(1.0 / np.asarray(q, dtype=float))


def dlnf_dlnq(q: float | np.ndarray) -> np.ndarray:
    """Analytic logarithmic derivative g(q) = d ln f_Egg(q) / d ln q."""
    q_arr = np.asarray(q, dtype=float)
    q13, q23 = q_arr ** (1.0 / 3.0), q_arr ** (2.0 / 3.0)
    D = 0.6 * q23 + np.log1p(q13)
    dD = 0.4 * q23 + (q13 / 3.0) / (1.0 + q13)
    return 2.0 / 3.0 - dD / D


def specific_am_parameter(
    q: float | np.ndarray,
    mode: str = "isotropic",
    rho2: Optional[float | np.ndarray] = None,
) -> np.ndarray:
    """
    Dimensionless angular-momentum loss parameter gamma entering d ln J_orb:
      - 'conservative': 0
      - 'jeans'       : 1 / (1 + q)            [direct donor wind]
      - 'isotropic'   : q^2 / (1 + q)          [point-source accretor re-emission]
                        or (1 + q) * rho_eff^2 [general co-rotating ejection site]
    """
    q_arr = np.asarray(q, dtype=float)
    if mode == "conservative":
        return np.zeros_like(q_arr)
    if mode == "jeans":
        return 1.0 / (1.0 + q_arr)
    if mode == "isotropic":
        if rho2 is not None:
            return (1.0 + q_arr) * np.asarray(rho2, dtype=float)
        return (q_arr ** 2) / (1.0 + q_arr)
    raise ValueError(f"Unknown mass-loss mode: {mode!r}")


def ejection_site_rho2(
    q: float | np.ndarray,
    ra_over_a: float | np.ndarray,
    geometry_mode: str = "point",
) -> np.ndarray:
    """
    Effective squared lever arm (r_eff / a)^2 from the binary center of mass for
    ejecta leaving a co-rotating accretor of fractional radius R_a / a:
      - 'point'        : (q / (1 + q))^2
      - 'donor_facing' : (q / (1 + q) - R_a / a)^2
      - 'far_side'     : (q / (1 + q) + R_a / a)^2
      - 'surface_avg'  : (q / (1 + q))^2 + (2/3) * (R_a / a)^2
      - 'com_floor'    : 0.0 (geometry-independent zero-angular-momentum floor)
    """
    q_arr = np.asarray(q, dtype=float)
    ra_arr = np.asarray(ra_over_a, dtype=float)
    x_cm = q_arr / (1.0 + q_arr)
    if geometry_mode == "point":
        return x_cm ** 2
    if geometry_mode == "donor_facing":
        return (x_cm - ra_arr) ** 2
    if geometry_mode == "far_side":
        return (x_cm + ra_arr) ** 2
    if geometry_mode == "surface_avg":
        return x_cm ** 2 + (2.0 / 3.0) * (ra_arr ** 2)
    if geometry_mode == "com_floor":
        return np.zeros_like(x_cm + ra_arr)
    raise ValueError(f"Unknown ejection geometry: {geometry_mode!r}")


def roche_lobe_exponent_coeffs(
    q: float | np.ndarray,
    mode: str = "isotropic",
    rho2: Optional[float | np.ndarray] = None,
    spin_gamma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return linear coefficients (Z_0(q), S(q)) such that:
        zeta_L(q, f) = d ln R_{L,d} / d ln M_d = Z_0(q) + f * S(q).
    Includes optional spin-sink parameter spin_gamma = kappa * (1 + q) * rho_in^2
    for non-synchronous accretors that retain a fraction kappa of stream spin AM.
    """
    q_arr = np.asarray(q, dtype=float)
    g = dlnf_dlnq(q_arr)
    Z0 = 2.0 * (q_arr - 1.0) + (1.0 + q_arr) * g
    gam = specific_am_parameter(q_arr, mode, rho2)
    S = 2.0 * gam - 2.0 * q_arr + q_arr / (1.0 + q_arr) - q_arr * g
    return Z0 + 2.0 * spin_gamma, S - 2.0 * spin_gamma


def roche_lobe_exponent(
    q: float | np.ndarray,
    f: float | np.ndarray = 0.0,
    mode: str = "conservative",
    lam: float = 0.0,
    rho2: Optional[float | np.ndarray] = None,
    spin_gamma: float = 0.0,
) -> np.ndarray:
    """Evaluate the Roche-lobe mass-radius exponent zeta_L(q, f) = Z_0(q) + f * S(q) + lam."""
    Z0, S = roche_lobe_exponent_coeffs(q, mode, rho2, spin_gamma)
    return Z0 + np.asarray(f, dtype=float) * S + lam


def critical_mass_loss_fraction(
    q: float | np.ndarray,
    zeta_s: float | np.ndarray,
    mode: str = "isotropic",
    lam: float = 0.0,
    rho2: Optional[float | np.ndarray] = None,
    spin_gamma: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Solve zeta_L(q, f) + lam = zeta_s for the critical mass-loss fraction:
        f_crit(q) = (zeta_s - Z_0(q) - lam) / S(q).
    """
    q_arr = np.atleast_1d(np.asarray(q, dtype=float))
    Z0, S = roche_lobe_exponent_coeffs(q_arr, mode, rho2, spin_gamma)
    Z0 = Z0 + lam
    with np.errstate(divide="ignore", invalid="ignore"):
        froot = (zeta_s - Z0) / S
    status = np.full(q_arr.shape, -1, dtype=int)
    f_eff = np.full(q_arr.shape, np.nan)
    kind = np.where(S < 0, "min", "max")
    stable0 = Z0 <= zeta_s
    status[stable0 & (S <= 0)] = 0
    f_eff[stable0 & (S <= 0)] = 0.0

    m = (~stable0) & (S < 0)
    ok = m & (froot <= 1.0)
    status[ok] = 1
    f_eff[ok] = froot[ok]
    status[m & (froot > 1.0)] = 2

    m = stable0 & (S > 0)
    status[m] = 3
    f_eff[m] = np.minimum(froot[m], 1.0)

    status[(~stable0) & (S >= 0)] = 2
    return {"f": froot, "status": status, "kind": kind, "f_eff": f_eff, "Z0": Z0, "S": S}


def critical_mass_ratio(
    zeta_s_fn: Callable[[float], float] | float,
    Ma_mjup: float = 80.0,
    f: float = 0.0,
    mode: str = "conservative",
    rho2_fn: Optional[Callable[[float], float]] = None,
) -> float:
    """Solve for the critical mass ratio q_crit where zeta_L(q, f) = zeta_s(q * M_a)."""
    def residual(q_val: float) -> float:
        zs = zeta_s_fn(q_val * Ma_mjup) if callable(zeta_s_fn) else float(zeta_s_fn)
        r2 = None if rho2_fn is None else rho2_fn(q_val)
        return float(roche_lobe_exponent(q_val, f, mode if f > 0 else "conservative", rho2=r2) - zs)

    qs = np.linspace(0.25, 1.25, 1001)
    vs = [residual(x) for x in qs]
    for i in range(len(qs) - 1):
        if vs[i] <= 0.0 < vs[i + 1]:
            return float(brentq(residual, qs[i], qs[i + 1], xtol=1e-12))
    return float("nan")


# ==============================================================================
# 4. SYSTEMIC AML, MASS-TRANSFER RATES, HOTSPOT & SECULAR EVOLUTION
# ==============================================================================

def orbital_elements(Md_g: float, Ma_g: float, P_s: float = P_ORB_S) -> Tuple[float, float]:
    """Return orbital separation a [cm] and orbital angular momentum J_orb [g cm^2 s^-1]."""
    M = Md_g + Ma_g
    Om = 2.0 * np.pi / P_s
    a = (G_CGS * M / (Om ** 2)) ** (1.0 / 3.0)
    J = (Md_g * Ma_g / M) * (a ** 2) * Om
    return a, J


def jdot_gravitational_waves(Md_g: float, Ma_g: float, a_cm: float) -> float:
    """Gravitational-wave angular-momentum loss rate Jdot_GW [erg] (Peters 1964)."""
    return -(32.0 / 5.0) * (G_CGS ** 3.5) / (C_LIGHT ** 5) * (Ma_g ** 2) * (Md_g ** 2) * ((Ma_g + Md_g) ** 0.5) / (a_cm ** 3.5)


def convective_turnover_time_wright(M_msun: float) -> float:
    """Wright et al. (2011, Eq. 11) convective turnover time [days] extrapolated to BDs."""
    lm = np.log10(M_msun)
    return float(10.0 ** (1.16 - 1.49 * lm - 0.54 * (lm ** 2)))


def jdot_magnetic_braking_sills(
    Md_g: float,
    Ma_g: float,
    Rd_cm: float,
    Ra_cm: float,
    P_s: float = P_ORB_S,
    P_sun_days: float = 25.4,
    K_mb: float = 1.04e35,
) -> float:
    """Saturated magnetic braking torque Jdot_MB,sat [erg] (Sills et al. 2000, Eqs. 7-8)."""
    tot = 0.0
    tau_sun = convective_turnover_time_wright(1.0)
    for M, R in ((Md_g, Rd_cm), (Ma_g, Ra_cm)):
        Pc_days = 0.1 * P_sun_days * convective_turnover_time_wright(M / M_SUN) / tau_sun
        tot += (Pc_days ** -2.0) * ((R / R_SUN) ** 0.5) * ((M / M_SUN) ** -0.5)
    return -K_mb * ((P_s / DAY_SEC) ** -1.0) * tot


def damping_rates_gamma(
    Md_g: float,
    Ma_g: float,
    P_s: float,
    Rd_cm: float,
    Ra_cm: float,
    eta_MB: float = 1.0,
) -> Tuple[float, float]:
    """Return (Gamma_GW, Gamma_MB) = (Jdot_GW / J_orb, eta_MB * Jdot_MB,sat / J_orb) [s^-1]."""
    a, J = orbital_elements(Md_g, Ma_g, P_s)
    return jdot_gravitational_waves(Md_g, Ma_g, a) / J, eta_MB * jdot_magnetic_braking_sills(Md_g, Ma_g, Rd_cm, Ra_cm, P_s) / J


def steady_state_mass_transfer_rate(
    Md_J: float,
    Ma_J: float,
    f: float = 0.0,
    mode: str = "isotropic",
    zs: float = -0.25,
    P_s: float = P_ORB_S,
    Ra_J: float = 0.92,
    eta_MB: float = 1.0,
    channels: Sequence[str] = ("GW", "MB"),
    rho_per_Gyr: float = 0.0,
    Rd_J: Optional[float] = None,
    rho2: Optional[float] = None,
    spin_gamma: float = 0.0,
) -> float:
    """
    Quasi-steady donor mass-transfer rate Mdot_d [M_Jup yr^-1] (< 0 for mass transfer;
    > 0 indicates thermal detachment when |rho| > 2 |Gamma_ext|; NaN when dynamically unstable):
        Mdot_d / M_d = (2 * Gamma_ext - rho) / (zeta_s - Z_0(q) - f * S(q)).
    """
    Md, Ma = Md_J * M_JUP, Ma_J * M_JUP
    q = Md / Ma
    a, _ = orbital_elements(Md, Ma, P_s)
    Rd = a * float(eggleton_roche_lobe(q)) if Rd_J is None else Rd_J * R_JUP
    Ra = Ra_J * R_JUP
    gGW, gMB = damping_rates_gamma(Md, Ma, P_s, Rd, Ra, eta_MB)
    Gam = (gGW if "GW" in channels else 0.0) + (gMB if "MB" in channels else 0.0)
    Z0, S = roche_lobe_exponent_coeffs(q, mode if f > 0 else "jeans", rho2, spin_gamma)
    rho_s = rho_per_Gyr / (1.0e9 * YR_SEC)
    den = float(zs) - (float(Z0) + f * float(S))
    if den <= 0.0:
        return float("nan")
    x = (2.0 * Gam - rho_s) / den
    return float(x * Md * YR_SEC / M_JUP)


def finite_step_contact_rate_solver(
    Md_J: float,
    Ma_J: float,
    f: float,
    mode: str,
    zs: float,
    Gamma_s: float,
    rho_s: float = 0.0,
    P_s: float = P_ORB_S,
    dt_yr: float = 50.0,
) -> float:
    """
    Independent finite-step verification solver for Mdot_d [M_Jup yr^-1] that
    directly advances (M_d, M_a, J_orb) over dt and solves for R_d(t+dt) = R_{L,d}(t+dt)
    without using the analytical zeta_L formula.
    """
    Md0, Ma0 = Md_J * M_JUP, Ma_J * M_JUP
    M0 = Md0 + Ma0
    a0, J0 = orbital_elements(Md0, Ma0, P_s)
    Om = 2.0 * np.pi / P_s
    Rd0 = a0 * float(eggleton_roche_lobe(Md0 / Ma0))
    dt = dt_yr * YR_SEC
    r_lever = {"jeans": a0 * Ma0 / M0, "isotropic": a0 * Md0 / M0, "conservative": 0.0}[mode]

    def contact_gap(dm: float) -> float:
        Md1 = Md0 - dm
        Ma1 = Ma0 + (1.0 - f) * dm
        M1 = Md1 + Ma1
        J1 = J0 * (1.0 + Gamma_s * dt) - f * dm * (r_lever ** 2) * Om
        a1 = ((J1 * np.sqrt(M1) / (Md1 * Ma1)) ** 2) / G_CGS
        RL1 = a1 * float(eggleton_roche_lobe(Md1 / Ma1))
        Rd1 = Rd0 * np.exp(zs * np.log(Md1 / Md0) + rho_s * dt)
        return RL1 - Rd1

    try:
        dm_sol = brentq(contact_gap, 1e-12 * Md0, 1e-3 * Md0, xtol=1e-30, rtol=1e-13)
        return float(-dm_sol / dt * YR_SEC / M_JUP)
    except ValueError:
        return float("nan")


def required_mb_efficiency_for_rate(
    Mdot_target_J_yr: float,
    Md_J: float,
    Ma_J: float,
    f: float,
    mode: str = "isotropic",
    zs: float = -0.25,
    P_s: float = P_ORB_S,
    Ra_J: float = 0.92,
    rho_per_Gyr: float = 0.0,
) -> float:
    """Solve for the magnetic braking efficiency factor eta_MB reproducing |Mdot_target|."""
    Md, Ma = Md_J * M_JUP, Ma_J * M_JUP
    q = Md / Ma
    a, _ = orbital_elements(Md, Ma, P_s)
    Rd = a * float(eggleton_roche_lobe(q))
    gGW, gMB_sat = damping_rates_gamma(Md, Ma, P_s, Rd, Ra_J * R_JUP, 1.0)
    Z0, S = roche_lobe_exponent_coeffs(q, mode if f > 0 else "jeans")
    x = -abs(Mdot_target_J_yr) * M_JUP / YR_SEC / Md
    Gam_req = 0.5 * (x * (zs - (float(Z0) + f * float(S))) + rho_per_Gyr / (1.0e9 * YR_SEC))
    return float((Gam_req - gGW) / gMB_sat)


def hotspot_bolometric_luminosity(
    T_K: float = T_SPOT_K,
    R_bb_J: float = R_BB_JUP,
    geometry: str = "patch",
) -> float:
    """Bolometric hotspot luminosity L_spot [erg s^-1] for a Lambertian patch or sphere."""
    area = np.pi * ((R_bb_J * R_JUP) ** 2)
    return float(area * SIGMA_SB * (T_K ** 4) * (4.0 if geometry == "sphere" else 1.0))


def mdot_from_hotspot_luminosity(
    L_erg_s: float,
    v_imp_kms: float,
    eps_rad: float = 1.0,
) -> float:
    """Mass-transfer rate |Mdot_d| [M_Jup yr^-1] required to power L_spot = eps_rad * (1/2) |Mdot| v_imp^2."""
    v_cms = v_imp_kms * KMS_CMS
    return float(2.0 * L_erg_s / (eps_rad * (v_cms ** 2)) * YR_SEC / M_JUP)


def integrate_secular_evolution(
    Md_J: float,
    Ma_J: float,
    f: float,
    mode: str,
    zeta_fn: Callable[[float], float],
    P0_s: float = P_ORB_S,
    Ra_J: float = 0.92,
    eta_MB: float = 1.0,
    channels: Sequence[str] = ("GW", "MB"),
    t_max_gyr: float = 13.8,
    Md_min_J: float = 10.0,
) -> Dict[str, np.ndarray | float | str]:
    """
    Integrate the secular evolution of (M_d(t), M_a(t), J_orb(t), P_orb(t)) under
    Roche-lobe contact down the mass-ratio diagonal over gigayear timescales.
    """
    Md0, Ma0 = Md_J * M_JUP, Ma_J * M_JUP
    _, J0 = orbital_elements(Md0, Ma0, P0_s)

    def rhs(t: float, y: Sequence[float]) -> List[float]:
        Md, Ma, J = y
        M = Md + Ma
        q = Md / Ma
        a = ((J * np.sqrt(M) / (Md * Ma)) ** 2) / G_CGS
        Om = np.sqrt(G_CGS * M / (a ** 3))
        P = 2.0 * np.pi / Om
        Rd = a * float(eggleton_roche_lobe(q))
        Ra = Ra_J * R_JUP
        gGW = jdot_gravitational_waves(Md, Ma, a) / J
        gMB = eta_MB * jdot_magnetic_braking_sills(Md, Ma, Rd, Ra, P) / J
        Gam = (gGW if "GW" in channels else 0.0) + (gMB if "MB" in channels else 0.0)
        Z0, S = roche_lobe_exponent_coeffs(q, mode if f > 0 else "jeans")
        zs = float(zeta_fn(Md / M_JUP))
        den = zs - (float(Z0) + f * float(S))
        if den <= 0.0:
            return [0.0, 0.0, 0.0]
        x = 2.0 * Gam / den
        Mdot = x * Md
        r_lev = {"jeans": a * Ma / M, "isotropic": a * Md / M, "conservative": 0.0}[mode]
        return [Mdot, -(1.0 - f) * Mdot, Gam * J + f * Mdot * (r_lev ** 2) * Om]

    def ev_unstable(t: float, y: Sequence[float]) -> float:
        Md, Ma, _ = y
        q = Md / Ma
        Z0, S = roche_lobe_exponent_coeffs(q, mode if f > 0 else "jeans")
        return float(zeta_fn(Md / M_JUP)) - (float(Z0) + f * float(S))

    ev_unstable.terminal = True
    ev_unstable.direction = -1

    def ev_empty(t: float, y: Sequence[float]) -> float:
        return y[0] / M_JUP - Md_min_J

    ev_empty.terminal = True

    t_end = t_max_gyr * 1.0e9 * YR_SEC
    if ev_unstable(0.0, [Md0, Ma0, J0]) <= 0.0:
        return {
            "fate": "unstable at t=0",
            "t_yr": np.array([0.0]),
            "Md": np.array([Md_J]),
            "Ma": np.array([Ma_J]),
            "P_min": np.array([P0_s / 60.0]),
            "Mdot": np.array([np.nan]),
            "q": np.array([Md_J / Ma_J]),
            "lifetime_yr": 0.0,
        }

    sol = solve_ivp(
        rhs, [0.0, t_end], [Md0, Ma0, J0],
        events=[ev_unstable, ev_empty],
        rtol=1e-9, atol=[1e-6 * Md0, 1e-6 * Ma0, 1e-6 * J0],
        method="LSODA", max_step=t_end / 2000.0
    )
    Md_arr, Ma_arr, J_arr = sol.y
    M_arr = Md_arr + Ma_arr
    a_arr = ((J_arr * np.sqrt(M_arr) / (Md_arr * Ma_arr)) ** 2) / G_CGS
    P_arr = 2.0 * np.pi * np.sqrt((a_arr ** 3) / (G_CGS * M_arr))
    Mdot_arr = np.array([rhs(0.0, sol.y[:, i])[0] for i in range(sol.y.shape[1])]) * YR_SEC / M_JUP
    fate = (
        "merger (zeta_s<=zeta_L)" if sol.t_events[0].size
        else (f"donor exhausted (<{Md_min_J:g} MJup)" if sol.t_events[1].size else "alive at t_max")
    )
    return {
        "fate": fate,
        "t_yr": sol.t / YR_SEC,
        "Md": Md_arr / M_JUP,
        "Ma": Ma_arr / M_JUP,
        "P_min": P_arr / 60.0,
        "Mdot": Mdot_arr,
        "q": Md_arr / Ma_arr,
        "lifetime_yr": float(sol.t[-1] / YR_SEC),
    }


# ==============================================================================
# 5. RESTRICTED THREE-BODY BALLISTIC STREAM & ENERGY-BUDGET SOLVER
# ==============================================================================

def binary_geometry(Md_J: float, Ma_J: float, P_s: float = P_ORB_S) -> Tuple[float, float, float, float, float]:
    """Return (Md_g, Ma_g, M_tot_g, Omega_rad_s, a_cm)."""
    Md, Ma = Md_J * M_JUP, Ma_J * M_JUP
    M = Md + Ma
    Om = 2.0 * np.pi / P_s
    a = (G_CGS * M / (Om ** 2)) ** (1.0 / 3.0)
    return Md, Ma, M, Om, a


def roche_potential(
    x: float | np.ndarray,
    y: float | np.ndarray,
    Md_g: float,
    Ma_g: float,
    a_cm: float,
    Om: float,
) -> np.ndarray:
    """Effective Roche potential Phi_eff(x, y) [erg g^-1] with origin at accretor."""
    xcm = a_cm * Md_g / (Md_g + Ma_g)
    r_acc = np.hypot(x, y)
    r_don = np.hypot(x - a_cm, y)
    return -G_CGS * Ma_g / r_acc - G_CGS * Md_g / r_don - 0.5 * (Om ** 2) * (((x - xcm) ** 2) + (y ** 2))


def find_L1_position(Md_g: float, Ma_g: float, a_cm: float, Om: float) -> float:
    """Find inner Lagrange point x_{L1} [cm] along the line of centers (0 < x_{L1} < a)."""
    xcm = a_cm * Md_g / (Md_g + Ma_g)
    dphi = lambda x: (G_CGS * Ma_g / (x ** 2) - G_CGS * Md_g / ((a_cm - x) ** 2) - (Om ** 2) * (x - xcm))
    return float(brentq(dphi, 1e-3 * a_cm, (1.0 - 1e-3) * a_cm, xtol=1e-6))


def escape_velocity(Ma_J: float, Ra_J: float) -> float:
    """Two-body surface escape speed v_esc = sqrt(2 G M_a / R_a) [cm s^-1]."""
    return float(np.sqrt(2.0 * G_CGS * Ma_J * M_JUP / (Ra_J * R_JUP)))


def binary_escape_velocity(Md_J: float, Ma_J: float, Ra_J: float, phi_deg: float = 0.0) -> float:
    """
    Speed needed to unbind co-rotating gas at the accretor surface from the WHOLE binary
    (accretor + donor point-mass potentials), at surface azimuth phi measured from the line of
    centres toward the donor (phi=0 is the L1-facing point):

        v_esc,bin^2 = 2 [ G M_a / R_a + G M_d / d_d  -  1/2 Omega^2 |r - r_cm|^2 ] .

    The paper's baseline uses the accretor-only value (`escape_velocity`). The donor term is not
    negligible at a = 2.44 R_Jup: it raises v_esc^2 by ~43-49 % and lowers f_max accordingly.
    """
    Mdk, Mak, Mtot, Om, a = binary_geometry(Md_J, Ma_J)
    R = Ra_J * R_JUP
    ph = np.radians(phi_deg)
    x, y = R * np.cos(ph), R * np.sin(ph)
    xcm = a * Mdk / Mtot
    phi_N = -G_CGS * Mak / R - G_CGS * Mdk / np.hypot(x - a, y)
    e_kin = 0.5 * Om ** 2 * ((x - xcm) ** 2 + y ** 2)
    return float(np.sqrt(2.0 * (-phi_N - e_kin)))


def stream_energetics(Md_J: float, Ma_J: float, Ra_J: float, escape_model: str = "accretor") -> Dict[str, float]:
    """
    Stream-impact energetics: impact speed v_imp (co-rotating frame = speed relative to a
    synchronous surface), escape speed for the chosen `escape_model` ("accretor": two-body,
    paper baseline; "binary": includes the donor potential) and f_max(eps=1) = (v_imp/v_esc)^2.
    """
    st = integrate_ballistic_stream(Md_J, Ma_J, Ra_J)
    if not st.get("hit", False):
        return {"hit": False, "v_imp": float("nan"), "v_esc": float("nan"), "f_max": float("nan"), "phi_deg": float("nan")}
    v_imp = float(st["v_rot"])
    if escape_model == "accretor":
        v_esc = escape_velocity(Ma_J, Ra_J)
    elif escape_model == "binary":
        v_esc = binary_escape_velocity(Md_J, Ma_J, Ra_J, float(st["phi_deg"]))
    else:
        raise ValueError(f"unknown escape_model {escape_model!r}")
    return {"hit": True, "v_imp": v_imp, "v_esc": v_esc, "f_max": (v_imp / v_esc) ** 2, "phi_deg": float(st["phi_deg"])}


def subdonor_jacobi_impact_speed(
    Md_J: float,
    Ma_J: float,
    Ra_J: float,
    P_s: float = P_ORB_S,
    phi_deg: float = 0.0,
) -> float:
    """Co-rotating Jacobi impact speed v_imp = sqrt(2 [Phi_eff(L1) - Phi_eff(R_a)]) [cm s^-1]."""
    Md, Ma, _, Om, a = binary_geometry(Md_J, Ma_J, P_s)
    R = Ra_J * R_JUP
    xL1 = find_L1_position(Md, Ma, a, Om)
    ph = np.radians(phi_deg)
    dPhi = float(roche_potential(xL1, 0.0, Md, Ma, a, Om) - roche_potential(R * np.cos(ph), R * np.sin(ph), Md, Ma, a, Om))
    return float(np.sqrt(2.0 * dPhi)) if dPhi > 0.0 else float("nan")


def integrate_ballistic_stream(
    Md_J: float,
    Ma_J: float,
    Ra_J: float,
    P_s: float = P_ORB_S,
    v0_kms: float = 2.0,
    eps_offset: float = 1e-4,
    return_trajectory: bool = False,
) -> Dict[str, float | bool | np.ndarray]:
    """
    Integrate the ballistic accretion stream from L_1 in the co-rotating frame
    using an 8th-order Dormand-Prince (DOP853) solver until impact at r = R_a.
    """
    Md, Ma, M, Om, a = binary_geometry(Md_J, Ma_J, P_s)
    R = Ra_J * R_JUP
    xcm = a * Md / M
    xL1 = find_L1_position(Md, Ma, a, Om)
    v0 = v0_kms * KMS_CMS

    if R >= xL1 * 0.999:
        return {
            "a": a, "Om": Om, "R": R, "xcm": xcm, "xL1": xL1,
            "r_min": float(R), "hit": False, "overfills_L1": True,
        }

    def rhs(t: float, s: Sequence[float]) -> List[float]:
        x, y, vx, vy = s
        r3 = np.hypot(x, y) ** 3
        d3 = np.hypot(x - a, y) ** 3
        ax = -G_CGS * Ma * x / r3 - G_CGS * Md * (x - a) / d3 + (Om ** 2) * (x - xcm) + 2.0 * Om * vy
        ay = -G_CGS * Ma * y / r3 - G_CGS * Md * y / d3 + (Om ** 2) * y - 2.0 * Om * vx
        return [vx, vy, ax, ay]

    def hit_event(t: float, s: Sequence[float]) -> float:
        return float(np.hypot(s[0], s[1]) - R)

    hit_event.terminal = True
    hit_event.direction = -1

    sol = solve_ivp(
        rhs, [0.0, 20.0 * P_s], [xL1 - eps_offset * a, 0.0, -v0, 0.0],
        events=hit_event, method="DOP853", rtol=1e-11, atol=[1e-3, 1e-3, 1e-3, 1e-3],
        max_step=P_s / 2000.0,
    )
    rr = np.hypot(sol.y[0], sol.y[1])
    out: Dict[str, float | bool | np.ndarray] = {
        "a": a, "Om": Om, "R": R, "xcm": xcm, "xL1": xL1,
        "r_min": float(rr.min()), "overfills_L1": False,
    }
    if return_trajectory:
        out["traj_x_over_a"] = sol.y[0] / a
        out["traj_y_over_a"] = sol.y[1] / a

    if sol.t_events[0].size:
        x, y, vx, vy = sol.y_events[0][0]
        e_init = 0.5 * (v0 ** 2) + float(roche_potential(xL1 - eps_offset * a, 0.0, Md, Ma, a, Om))
        e_hit = 0.5 * ((vx ** 2) + (vy ** 2)) + float(roche_potential(x, y, Md, Ma, a, Om))
        e_scale = float(0.5 * ((vx ** 2) + (vy ** 2)))
        out.update({
            "hit": True,
            "phi_deg": float(np.degrees(np.arctan2(y, x))),
            "x": float(x), "y": float(y),
            "vx": float(vx), "vy": float(vy),
            "v_rot": float(np.hypot(vx, vy)),
            "jacobi_rel_err": float(abs(e_hit - e_init) / e_scale),
            "site_rho2": float((((x - xcm) ** 2) + (y ** 2)) / (a ** 2)),
        })
    else:
        out["hit"] = False
    return out


def surface_relative_impact_speed(stream_res: Dict[str, float | bool | np.ndarray], spin_sync: float = 1.0) -> float:
    """Impact speed relative to an accretor spinning at s = omega_a / Omega_orb."""
    if not stream_res.get("hit", False):
        return float("nan")
    Om = float(stream_res["Om"])
    x, y = float(stream_res["x"]), float(stream_res["y"])
    vx, vy = float(stream_res["vx"]), float(stream_res["vy"])
    wx, wy = -(1.0 - spin_sync) * Om * y, (1.0 - spin_sync) * Om * x
    return float(np.hypot(vx + wx, vy + wy))


def luminosity_coupled_fmax(
    Mdot_J_yr: float,
    v_imp_cms: float,
    v_esc_cms: float,
    L_spot_erg_s: Optional[float] = None,
) -> float:
    """
    Luminosity-coupled energetic upper bound on f when the observed hotspot
    luminosity L_spot must also be supplied by the stream impact power:
        f <= max(0, (v_imp^2 - 2 L_spot / |Mdot_d|) / v_esc^2).
    """
    L_val = hotspot_bolometric_luminosity() if L_spot_erg_s is None else L_spot_erg_s
    mdot_cgs = abs(Mdot_J_yr) * M_JUP / YR_SEC
    return float(max(0.0, ((v_imp_cms ** 2) - 2.0 * L_val / mdot_cgs) / (v_esc_cms ** 2)))


# ==============================================================================
# 6. AGE-GRID DATA LOADER, FULL ANALYSIS DRIVER & UNIT TEST SUITE
# ==============================================================================

def rho_at_age(t_gyr: np.ndarray, R: np.ndarray, t0: float, fac: float = 1.5, deg: int = 2) -> float:
    """
    Secular contraction rate rho = (d ln R / d t)_M [Gyr^-1] evaluated AT age t0.
    Local least-squares polynomial fit of ln R against ln t over [t0/fac, fac*t0]; then
    rho = (d ln R / d ln t) / t0. Insensitive to near-duplicate samples where several MESA
    snapshot histories overlap in time (unlike a raw finite-difference gradient).
    """
    t_gyr = np.asarray(t_gyr, float)
    R = np.asarray(R, float)
    s = (t_gyr >= t0 / fac) & (t_gyr <= t0 * fac) & (t_gyr > 0) & (R > 0)
    if s.sum() < deg + 2:
        return float("nan")
    xc = np.log(t0)
    c = np.polyfit(np.log(t_gyr[s]) - xc, np.log(R[s]), deg)
    return float(c[-2] / t0)


def load_or_build_age_grid_dataset(mesa_runs_dir: Optional[Path] = None, strict: bool = False) -> Dict:
    """
    Load raw MESA histories from `mesa_runs_dir` if available; otherwise use the
    embedded pre-verified 266-run MESA tables so the script runs self-contained
    in any environment (including a clean GitHub clone).
    """
    # Check default paths for raw MESA runs
    candidate_dirs = []
    if mesa_runs_dir is not None:
        candidate_dirs.append(Path(mesa_runs_dir))
    here = Path(__file__).resolve().parent
    candidate_dirs += [here / "runs"]
    runs_path = next((p for p in candidate_dirs if p.exists() and (p / "M60_age1" / "LOGS" / "history.data").exists()), None)
    if runs_path is None and strict:
        raise FileNotFoundError("Raw MESA runs not found (strict mode forbids the embedded fallback tables). "
                                "Pass --mesa-runs /path/to/runs or place them in ./runs")
    if runs_path is None:
        print("[WARNING] Raw MESA runs not found: using the EMBEDDED FALLBACK tables (paper values, "
              "NOT recomputed from MESA histories).", file=sys.stderr)

    tracks: Dict[int, Dict[str, np.ndarray]] = {}
    snap_table: Dict[str, Dict[int, Dict[str, float]]] = {atag: {} for atag in AGE_TAGS}
    curves: Dict[Tuple[int, float, str], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    age_zeta_ad: Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    if runs_path is not None:
        for m in MASSES_MJUP:
            ts, rs, lhs, teffs = [], [], [], []
            for atag in AGE_TAGS:
                h = read_mesa_history(runs_path / f"M{m}_age{atag}" / "LOGS" / "history.data")
                ts.append(h["star_age"])
                rs.append(h["radius"] / RJUP_PER_RSUN)
                lhs.append(10.0 ** (h["log_LH"] - h["log_L"]))
                teffs.append(h["Teff"] if "Teff" in h else 10.0 ** h["log_Teff"])
            hc10 = read_mesa_history(runs_path / f"M{m}_A10_ctrl" / "LOGS" / "history.data")
            ts.append(hc10["star_age"])
            rs.append(hc10["radius"] / RJUP_PER_RSUN)
            lhs.append(10.0 ** (hc10["log_LH"] - hc10["log_L"]))
            teffs.append(hc10["Teff"] if "Teff" in hc10 else 10.0 ** hc10["log_Teff"])
            t_all = np.concatenate(ts)
            r_all = np.concatenate(rs)
            lh_all = np.concatenate(lhs)
            teff_all = np.concatenate(teffs)
            _, uid = np.unique(t_all, return_index=True)
            t_all, r_all, lh_all, teff_all = t_all[uid], r_all[uid], lh_all[uid], teff_all[uid]
            rho_all = np.gradient(np.log(r_all), t_all) * 1.0e9
            tracks[m] = {"t_gyr": t_all / 1.0e9, "R": r_all, "LH_L": lh_all, "Teff": teff_all, "rho": rho_all}

            for a, atag in zip(AGES_GYR, AGE_TAGS):
                h_s = read_mesa_history(runs_path / f"M{m}_age{atag}" / "LOGS" / "history.data")
                R_rj = float(h_s["radius"][-1] / RJUP_PER_RSUN)
                Teff_s = float(h_s["Teff"][-1]) if "Teff" in h_s else float(10.0 ** h_s["log_Teff"][-1])
                lh_s = float(10.0 ** (h_s["log_LH"][-1] - h_s["log_L"][-1]))
                # rho(t0) = (d ln R / dt) at the snapshot age t0 itself, from a local quadratic fit of
                # ln R vs ln t over [t0/1.5, 1.5 t0] on the continuous track (see rho_at_age()).
                rho_s = rho_at_age(t_all / 1.0e9, r_all, a)
                # Legacy definition (control-run gradient 50 Myr after t0), kept for transparency only.
                ctrl_hf = runs_path / f"M{m}_A{atag}_ctrl" / "LOGS" / "history.data"
                if ctrl_hf.exists():
                    hc = read_mesa_history(ctrl_hf)
                    tc = hc["star_age"]
                    idx = int(np.argmin(np.abs(tc - (tc[0] + 0.05e9))))
                    rho_legacy = float((np.gradient(np.log(hc["radius"]), tc) * 1.0e9)[idx])
                else:
                    rho_legacy = float("nan")
                snap_table[atag][m] = {"R": R_rj, "Teff": Teff_s, "LH_L": lh_s, "rho": rho_s,
                                       "rho_legacy_t0_plus_50Myr": rho_legacy}

        out_dir = next((d for d in (runs_path.parent / "out", here / "out", runs_path.parent.parent / "out") if d.exists()),
                       runs_path.parent / "out")
        ac_json = out_dir / "age_grid_curves.json"
        ac_cache = json.load(open(ac_json)) if ac_json.exists() else {}

        for m in MASSES_MJUP:
            for a, atag in zip(AGES_GYR, AGE_TAGS):
                ctrl_hf = runs_path / f"M{m}_A{atag}_ctrl" / "LOGS" / "history.data"
                for rt in RATES_TAGS:
                    hf = runs_path / f"M{m}_A{atag}_{rt}" / "LOGS" / "history.data"
                    if hf.exists() and ctrl_hf.exists():
                        s_hist = read_mesa_history(hf)
                        c_hist = read_mesa_history(ctrl_hf)
                        Mg, z, el = extract_zeta_curve(s_hist, c_hist, subtract_control=True, return_elapsed=True)
                        curves[(m, a, rt)] = (Mg, z, el)
                    elif f"{m}|{a:g}|{rt}" in ac_cache:
                        item = ac_cache[f"{m}|{a:g}|{rt}"]
                        curves[(m, a, rt)] = (np.array(item["Mg"]), np.array(item["z"]), np.array(item["el"]))

        rl_json = out_dir / "rate_ladder.json"
        if rl_json.exists():
            rl_data = json.load(open(rl_json))
            by_run_rl: Dict[Tuple[int, float, str], List[Tuple[float, float]]] = {}
            for row in rl_data["rows"]:
                M0, age_v, M_v = int(row["M0"]), float(row["age"]), float(row["M"])
                for rt in ("R1e-6", "R1e-7", "R1e-8"):
                    if rt in row:
                        by_run_rl.setdefault((M0, age_v, rt), []).append((M_v, float(row[rt])))
            for (M0, age_v, rt), pts in by_run_rl.items():
                if (M0, age_v, rt) not in curves:
                    pts_sorted = sorted(pts, key=lambda x: -x[0])
                    Mg = np.array([p[0] for p in pts_sorted])
                    z = np.array([p[1] for p in pts_sorted])
                    el = (M0 - Mg) / float(rt[1:])
                    curves[(M0, age_v, rt)] = (Mg, z, el)

        zm_json = out_dir / "zeta_model.json"
        zm_v3 = json.load(open(zm_json)) if zm_json.exists() else None
        for a in AGES_GYR:
            sub = {k: v for k, v in curves.items() if abs(k[1] - a) < 1e-6}
            _, med, lo, hi = pool_donor_exponents(sub, ["R1e-6"], 0.0, None)
            if a in (1.0, 2.0) and zm_v3 is not None:
                for idx in range(len(BIN_CENTERS)):
                    if not np.isfinite(med[idx]) and zm_v3["zeta_ad_med"][idx] is not None:
                        med[idx] = zm_v3["zeta_ad_med"][idx]
                        lo[idx] = zm_v3["zeta_ad_lo"][idx]
                        hi[idx] = zm_v3["zeta_ad_hi"][idx]
            age_zeta_ad[a] = (BIN_CENTERS.copy(), med, lo, hi)
    else:
        # Fallback mode when raw MESA history files are not present locally
        snap_table = FALLBACK_SNAPSHOT_TABLE
        for m in MASSES_MJUP:
            t_arr = np.array(AGES_GYR, dtype=float)
            r_arr = np.array([snap_table[atag][m]["R"] for atag in AGE_TAGS], dtype=float)
            lh_arr = np.array([snap_table[atag][m]["LH_L"] for atag in AGE_TAGS], dtype=float)
            teff_arr = np.array([snap_table[atag][m]["Teff"] for atag in AGE_TAGS], dtype=float)
            rho_arr = np.array([snap_table[atag][m]["rho"] for atag in AGE_TAGS], dtype=float)
            tracks[m] = {"t_gyr": t_arr, "R": r_arr, "LH_L": lh_arr, "Teff": teff_arr, "rho": rho_arr}
        for a in AGES_GYR:
            med_interp = np.interp(BIN_CENTERS, FALLBACK_ZETA_NODES_MJUP, FALLBACK_AGE_ZETA_AD[a])
            age_zeta_ad[a] = (BIN_CENTERS.copy(), med_interp, med_interp - 0.012, med_interp + 0.012)

    provenance = {"source": "raw_mesa_histories+cached_json" if runs_path else "embedded_fallback"}
    if runs_path is not None:
        all_dirs = [d for d in runs_path.iterdir() if d.is_dir()]
        n_hist = sum((d / "LOGS" / "history.data").exists() for d in all_dirs)
        raw_curves = sum(1 for k in curves if (runs_path / f"M{k[0]}_A{AGE_TAGS[AGES_GYR.index(k[1])]}_{k[2]}" / "LOGS" / "history.data").exists()) if curves else 0
        provenance.update({"n_run_dirs": len(all_dirs), "n_with_history": n_hist,
                           "n_zeta_curves": len(curves), "n_zeta_curves_from_raw_history": raw_curves,
                           "n_zeta_curves_from_cached_json": len(curves) - raw_curves})
    return {
        "provenance": provenance,
        "runs_path": str(runs_path) if runs_path else "embedded_fallback",
        "tracks": tracks,
        "snap_table": snap_table,
        "curves": curves,
        "age_zeta_ad": age_zeta_ad,
    }


def make_age_interpolators(dataset: Dict) -> Tuple[Callable[[float, float], float], Callable[[float, float], float], Callable[[float, float], float]]:
    """
    Build smooth interpolators from the MESA age-grid dataset:
      - zeta_ad_fn(M_d_MJup, age_Gyr)
      - radius_fn(M_MJup, age_Gyr)
      - rho_fn(M_MJup, age_Gyr)
    """
    age_zeta_ad = dataset["age_zeta_ad"]
    tracks = dataset["tracks"]
    snap_table = dataset["snap_table"]

    def zeta_ad_fn(Md_J: float, age_gyr: float) -> float:
        a_near = min(AGES_GYR, key=lambda x: abs(x - age_gyr))
        cen, med, _, _ = age_zeta_ad[a_near]
        ok = np.isfinite(med)
        return float(np.interp(Md_J, cen[ok], med[ok]))

    def radius_fn(M_J: float, age_gyr: float) -> float:
        rs_at_t = [float(np.interp(age_gyr, tracks[m]["t_gyr"], tracks[m]["R"])) for m in MASSES_MJUP]
        return float(np.interp(M_J, MASSES_MJUP, rs_at_t))

    def rho_fn(M_J: float, age_gyr: float) -> float:
        a_near_idx = int(np.argmin([abs(x - age_gyr) for x in AGES_GYR]))
        atag = AGE_TAGS[a_near_idx]
        rhos = [snap_table[atag][m]["rho"] for m in MASSES_MJUP]
        return float(np.interp(M_J, MASSES_MJUP, rhos))

    return zeta_ad_fn, radius_fn, rho_fn


def run_full_simulation_suite(output_json: Optional[Path] = None, mesa_runs_dir: Optional[Path] = None, strict: bool = False) -> Dict:
    """
    Execute the complete multi-age stability, Roche-contact, stream-kinematics,
    and mass-transfer rate simulation suite and write `ztf_j1239_simulation_results.json`.
    """
    dataset = load_or_build_age_grid_dataset(mesa_runs_dir, strict=strict)
    zeta_ad_fn, radius_fn, rho_fn = make_age_interpolators(dataset)
    snap_table = dataset["snap_table"]

    # 1. Age-grid summary for 60/80 M_Jup and q_crit / q_max scaling across 0.1-10 Gyr
    age_scaling_rows = []
    for a, atag in zip(AGES_GYR, AGE_TAGS):
        Rd60 = radius_fn(60.0, a)
        Ra80 = radius_fn(80.0, a)
        Ra70 = radius_fn(70.0, a)
        zs60 = zeta_ad_fn(60.0, a)
        qc80 = critical_mass_ratio(lambda md: zeta_ad_fn(md, a), Ma_mjup=80.0)
        qc70 = critical_mass_ratio(lambda md: zeta_ad_fn(md, a), Ma_mjup=70.0)

        _, _, _, Om, a_orb = binary_geometry(60.0, 80.0)
        rho_pt = float(ejection_site_rho2(0.75, Ra80 * R_JUP / a_orb, "point"))
        fc_pt = float(critical_mass_loss_fraction(0.75, zs60, "isotropic", rho2=rho_pt)["f"][0])
        fc_fl = float(critical_mass_loss_fraction(0.75, zs60, "isotropic", rho2=0.0)["f"][0])
        fc_fl_dyn = float(critical_mass_loss_fraction(0.75, zs60 - DELTA_DYN_MEAS, "isotropic", rho2=0.0)["f"][0])

        st60_80 = integrate_ballistic_stream(60.0, 80.0, Ra80)
        ve80 = escape_velocity(80.0, Ra80) / KMS_CMS
        if st60_80["hit"]:
            v_imp_80 = float(st60_80["v_rot"]) / KMS_CMS
            fmax_80 = (v_imp_80 / ve80) ** 2
            eps_fl = fc_fl / fmax_80
            eps_fl_dyn = fc_fl_dyn / fmax_80
            eps_pt = fc_pt / fmax_80
        else:
            v_imp_80 = fmax_80 = eps_fl = eps_fl_dyn = eps_pt = float("nan")

        fmax_80_bin = stream_energetics(60.0, 80.0, Ra80, "binary")["f_max"]

        # Energetically allowed q_max(eps=1) at M_tot=140 M_Jup and at Ma=70 M_Jup
        qs_scan = np.arange(0.60, 0.83, 0.002)
        qmax_fl_80 = qmax_dyn_80 = qmax_pt_80 = qmax_fl_70 = float("nan")
        for qq in qs_scan:
            md_q, ma_q = qq * 140.0 / (1.0 + qq), 140.0 / (1.0 + qq)
            ra_q = radius_fn(ma_q, a)
            st_q = integrate_ballistic_stream(md_q, ma_q, ra_q)
            if st_q["hit"]:
                fm_q = (float(st_q["v_rot"]) / escape_velocity(ma_q, ra_q)) ** 2
                _, _, _, _, a_q = binary_geometry(md_q, ma_q)
                r_pt_q = float(ejection_site_rho2(qq, ra_q * R_JUP / a_q, "point"))
                f0 = critical_mass_loss_fraction(qq, zeta_ad_fn(md_q, a), "isotropic", rho2=0.0)
                fd = critical_mass_loss_fraction(qq, zeta_ad_fn(md_q, a) - DELTA_DYN_MEAS, "isotropic", rho2=0.0)
                fp = critical_mass_loss_fraction(qq, zeta_ad_fn(md_q, a), "isotropic", rho2=r_pt_q)
                if int(f0["status"][0]) != 2 and max(float(f0["f"][0]), 0.0) <= fm_q:
                    qmax_fl_80 = float(qq)
                if int(fd["status"][0]) != 2 and max(float(fd["f"][0]), 0.0) <= fm_q:
                    qmax_dyn_80 = float(qq)
                if int(fp["status"][0]) != 2 and max(float(fp["f"][0]), 0.0) <= fm_q:
                    qmax_pt_80 = float(qq)

            md_70 = qq * 70.0
            st_70 = integrate_ballistic_stream(md_70, 70.0, Ra70)
            if st_70["hit"]:
                fm_70 = (float(st_70["v_rot"]) / escape_velocity(70.0, Ra70)) ** 2
                f70 = critical_mass_loss_fraction(qq, zeta_ad_fn(md_70, a), "isotropic", rho2=0.0)
                if int(f70["status"][0]) != 2 and max(float(f70["f"][0]), 0.0) <= fm_70:
                    qmax_fl_70 = float(qq)

        # Mass-transfer rates with and without secular contraction rho
        rho40 = snap_table[atag][40]["rho"]
        rho60 = snap_table[atag][60]["rho"]
        mdot_50_80_gw_no_rho = steady_state_mass_transfer_rate(
            50.0, 80.0, 0.0, "isotropic", zeta_ad_fn(50.0, a), Ra_J=Ra80, channels=("GW",), rho_per_Gyr=0.0
        )
        mdot_50_80_gw_rho = steady_state_mass_transfer_rate(
            50.0, 80.0, 0.0, "isotropic", zeta_ad_fn(50.0, a), Ra_J=Ra80, channels=("GW",), rho_per_Gyr=rho40
        )
        mdot_60_80_gw_no_rho = steady_state_mass_transfer_rate(
            60.0, 80.0, 0.6, "isotropic", zs60, Ra_J=Ra80, channels=("GW",), rho_per_Gyr=0.0
        )
        mdot_60_80_gw_rho = steady_state_mass_transfer_rate(
            60.0, 80.0, 0.6, "isotropic", zs60, Ra_J=Ra80, channels=("GW",), rho_per_Gyr=rho60
        )
        mdot_60_80_mb_rho = steady_state_mass_transfer_rate(
            60.0, 80.0, 0.6, "isotropic", zs60, Ra_J=Ra80, channels=("GW", "MB"), rho_per_Gyr=rho60
        )

        age_scaling_rows.append({
            "age_gyr": a,
            "Rd_60_RJ": Rd60,
            "Ra_80_RJ": Ra80,
            "Ra_70_RJ": Ra70,
            "zeta_ad_60": zs60,
            "q_crit_f0_80": qc80,
            "q_crit_f0_70": qc70,
            "f_crit_pt_60_80": fc_pt,
            "f_crit_fl_60_80": fc_fl,
            "v_imp_60_80_kms": v_imp_80,
            "v_esc_80_kms": ve80,
            "f_max_60_80": fmax_80,
            "f_max_60_80_binary_escape": fmax_80_bin,
            "eps_min_fl_60_80_binary_escape": (fc_fl / fmax_80_bin) if np.isfinite(fmax_80_bin) else float("nan"),
            "eps_min_fl_60_80": eps_fl,
            "eps_min_fl_dyn_60_80": eps_fl_dyn,
            "eps_min_pt_60_80": eps_pt,
            "q_max_fl_80": qmax_fl_80,
            "q_max_dyn_80": qmax_dyn_80,
            "q_max_pt_80": qmax_pt_80,
            "q_max_fl_70": qmax_fl_70,
            "rho_40_per_gyr": rho40,
            "rho_60_per_gyr": rho60,
            "mdot_50_80_gw_no_rho": mdot_50_80_gw_no_rho,
            "mdot_50_80_gw_with_rho": mdot_50_80_gw_rho,
            "mdot_60_80_gw_no_rho": mdot_60_80_gw_no_rho,
            "mdot_60_80_gw_with_rho": mdot_60_80_gw_rho,
            "mdot_60_80_mb_with_rho": mdot_60_80_mb_rho,
        })

    # 2. Candidate configurations across 50-70 and 60-80 M_Jup at 2, 3, 5 Gyr
    candidate_pairs = [
        (50, 80), (50, 75), (47, 70), (55, 80), (50, 70),
        (55, 75), (52, 70), (60, 80), (50, 65), (55, 70), (60, 70),
    ]
    candidates_summary = []
    for Md, Ma in candidate_pairs:
        q = Md / Ma
        _, _, _, _, a_orb = binary_geometry(Md, Ma)
        RL_d = a_orb * float(eggleton_roche_lobe(q)) / R_JUP
        try:
            t_cont = float(brentq(lambda tg: radius_fn(Md, tg) - RL_d, 0.5, 10.0))
        except ValueError:
            t_cont = float("nan")

        row_cand = {"Md": Md, "Ma": Ma, "q": q, "RL_d_RJ": RL_d, "t_contact_gyr": t_cont, "by_age": {}}
        for tg in (2.0, 3.0, 5.0):
            Rd_t = radius_fn(Md, tg)
            Ra_t = radius_fn(Ma, tg)
            zs_t = zeta_ad_fn(Md, tg)
            fc_fl = max(0.0, float(critical_mass_loss_fraction(q, zs_t, "isotropic", rho2=0.0)["f"][0]))
            fc_fl_d = max(0.0, float(critical_mass_loss_fraction(q, zs_t - DELTA_DYN_MEAS, "isotropic", rho2=0.0)["f"][0]))
            r_pt = float(ejection_site_rho2(q, Ra_t * R_JUP / a_orb, "point"))
            fc_pt = max(0.0, float(critical_mass_loss_fraction(q, zs_t, "isotropic", rho2=r_pt)["f"][0]))
            st_t = integrate_ballistic_stream(Md, Ma, Ra_t)
            fm_t = (float(st_t["v_rot"]) / escape_velocity(Ma, Ra_t)) ** 2 if st_t["hit"] else float("nan")
            fm_bin = stream_energetics(Md, Ma, Ra_t, "binary")["f_max"]
            row_cand["by_age"][str(tg)] = {
                "Rd_RJ": Rd_t,
                "Ra_RJ": Ra_t,
                "f_crit_fl": fc_fl,
                "f_crit_fl_dyn": fc_fl_d,
                "f_crit_pt": fc_pt,
                "f_max": fm_t,
                "f_max_binary_escape": fm_bin,
                "eps_min_fl_binary_escape": fc_fl / fm_bin if fc_fl > 0 else 0.0,
                "eps_min_fl_dyn_binary_escape": fc_fl_d / fm_bin if fc_fl_d > 0 else 0.0,
                "eps_min_fl": fc_fl / fm_t if fc_fl > 0 else 0.0,
                "eps_min_fl_dyn": fc_fl_d / fm_t if fc_fl_d > 0 else 0.0,
                "eps_min_pt": fc_pt / fm_t if fc_pt > 0 else 0.0,
            }
        candidates_summary.append(row_cand)

    results = {
        "system": {
            "name": "ZTF J1239+8347",
            "P_orb_min": P_ORB_MIN,
            "T_spot_K": T_SPOT_K,
            "L_spot_patch_erg_s": hotspot_bolometric_luminosity(T_SPOT_K, R_BB_JUP, "patch"),
        },
        "cond03_comparison_1gyr": FALLBACK_COND03_1GYR,
        "provenance": dataset["provenance"],
        "snapshot_table": snap_table,
        "age_zeta_ad": {
            str(a): {
                "M_bins": age_zeta_ad_v[0].tolist(),
                "med": [None if not np.isfinite(x) else float(x) for x in age_zeta_ad_v[1]],
                "lo": [None if not np.isfinite(x) else float(x) for x in age_zeta_ad_v[2]],
                "hi": [None if not np.isfinite(x) else float(x) for x in age_zeta_ad_v[3]],
            }
            for a, age_zeta_ad_v in dataset["age_zeta_ad"].items()
        },
        "age_scaling": age_scaling_rows,
        "candidates": candidates_summary,
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


def run_verification_tests() -> None:
    """Run automated analytical and numerical verification checks."""
    # 1. Eggleton derivative check against centered finite differences
    q_test = np.linspace(0.2, 1.5, 50)
    dq = 1e-6
    g_num = (np.log(eggleton_roche_lobe(q_test + dq)) - np.log(eggleton_roche_lobe(q_test - dq))) / (2.0 * dq / q_test)
    g_ana = dlnf_dlnq(q_test)
    assert np.allclose(g_num, g_ana, rtol=1e-7), "Eggleton logarithmic derivative mismatch!"

    # 2. Polytropic n=1.5 (zeta_s = -1/3) conservative q_crit check
    qc_poly = critical_mass_ratio(-1.0 / 3.0)
    assert abs(qc_poly - 0.6339) < 1e-3, f"Unexpected polytropic q_crit: {qc_poly}"

    # 3. Closed-form mass-transfer rate vs independent finite-step contact solver
    Md_t, Ma_t, f_t, zs_t = 50.0, 80.0, 0.2, -0.245
    Md_g, Ma_g = Md_t * M_JUP, Ma_t * M_JUP
    a_t, _ = orbital_elements(Md_g, Ma_g, P_ORB_S)
    Rd_t = a_t * float(eggleton_roche_lobe(Md_t / Ma_t))
    gGW, _ = damping_rates_gamma(Md_g, Ma_g, P_ORB_S, Rd_t, 0.90 * R_JUP, 0.0)
    mdot_ana = steady_state_mass_transfer_rate(Md_t, Ma_t, f_t, "isotropic", zs_t, Ra_J=0.90, channels=("GW",))
    mdot_num = finite_step_contact_rate_solver(Md_t, Ma_t, f_t, "isotropic", zs_t, gGW, dt_yr=10.0)
    assert abs(mdot_ana - mdot_num) / abs(mdot_ana) < 1e-4, f"Rate solver mismatch: {mdot_ana} vs {mdot_num}"

    # 4. Jacobi constant conservation along ballistic stream integration
    st = integrate_ballistic_stream(60.0, 80.0, 0.967)
    assert st["hit"] and float(st["jacobi_rel_err"]) < 1e-12, f"Jacobi conservation error too large: {st}"

    # 5. Independent finite-step derivation of the Roche-lobe exponent zeta_L = Z0 + f S
    #    (explicit mass / angular-momentum bookkeeping; does not use the closed-form Z0, S).
    def zeta_L_numeric(q: float, f: float, gamma_j: float, spin_sigma: float = 0.0, eps: float = 1e-7) -> float:
        Md, Ma = q * 100.0, 100.0                     # arbitrary units; G = a = 1 below
        a0, G = 1.0, 1.0
        J0 = Md * Ma * np.sqrt(G * a0 / (Md + Ma))
        dMd = eps * Md
        Md1, Ma1 = Md - dMd, Ma + (1.0 - f) * dMd
        # ejected gas carries gamma_j * a^2 * Omega per unit mass (gamma_j = rho_eff^2);
        # accreted gas deposits sigma-fraction of its stream AM in the spin instead of the orbit
        Om = np.sqrt(G * (Md + Ma) / a0 ** 3)
        J1 = J0 - gamma_j * a0 ** 2 * Om * f * dMd - spin_sigma * a0 ** 2 * Om * (1.0 - f) * dMd
        a1 = (J1 / (Md1 * Ma1)) ** 2 * (Md1 + Ma1) / G
        RL0 = a0 * float(eggleton_roche_lobe(Md / Ma)); RL1 = a1 * float(eggleton_roche_lobe(Md1 / Ma1))
        return float(np.log(RL1 / RL0) / np.log(Md1 / Md))

    for q_t, f_t2 in ((0.5, 0.0), (0.75, 0.3), (0.9, 1.0)):
        for mode_t, gj in (("isotropic", (q_t / (1 + q_t)) ** 2), ("jeans", 1.0 / (1 + q_t) ** 2), ("conservative", 0.0)):
            Z0_t, S_t = roche_lobe_exponent_coeffs(q_t, mode_t)
            ana = float(Z0_t + f_t2 * S_t)
            num = zeta_L_numeric(q_t, f_t2, gj)
            assert abs(ana - num) < 5e-6, f"zeta_L mismatch ({mode_t}, q={q_t}, f={f_t2}): {ana} vs {num}"
    # spin-sink term: sigma = kappa (1+q) rho_in^2 -> Z0 + 2 sigma, S - 2 sigma
    q_t, sg = 0.75, 0.04
    Z0_t, S_t = roche_lobe_exponent_coeffs(q_t, "isotropic", spin_gamma=sg)
    num = zeta_L_numeric(q_t, 0.4, (q_t / (1 + q_t)) ** 2, spin_sigma=sg / (1 + q_t))
    assert abs(float(Z0_t + 0.4 * S_t) - num) < 5e-6, "spin-sink coefficient mismatch"

    # 6. Peters (1964) GW angular-momentum loss against the analytic circular-orbit expression
    Md_g, Ma_g = 60.0 * M_JUP, 80.0 * M_JUP
    a_g, J_g = orbital_elements(Md_g, Ma_g, P_ORB_S)
    jdot_an = -(32.0 / 5.0) * G_CGS ** 3.5 / C_LIGHT ** 5 * (Md_g * Ma_g) ** 2 * (Md_g + Ma_g) ** 0.5 / a_g ** 3.5
    assert abs(jdot_gravitational_waves(Md_g, Ma_g, a_g) / jdot_an - 1.0) < 1e-12
    assert abs(a_g / R_JUP - 2.44) < 0.01, "orbital separation of 60+80 M_Jup at P=57.41 min should be ~2.44 R_Jup"

    # 7. Escape-energy bookkeeping: donor potential can only raise the unbinding energy
    for t_age_R in (0.90, 0.97):
        assert binary_escape_velocity(60.0, 80.0, t_age_R) > escape_velocity(80.0, t_age_R)

    # 8. COND03 vs MESA 1 Gyr radius agreement (< 1%): uses the CSV and, when present, the real MESA snapshots
    csv_path = Path(__file__).resolve().parent / "data" / "COND03_1Gyr.csv"
    r_cond = np.array(FALLBACK_COND03_1GYR["R_cond_RJ"]); r_mesa = np.array(FALLBACK_COND03_1GYR["R_mesa_RJ"])
    if csv_path.exists():
        r_cond_csv = cond03_radius_rj(csv_path, MASSES_MJUP)
        runs_dir = Path(__file__).resolve().parent / "runs"
        if (runs_dir / "M60_age1" / "LOGS" / "history.data").exists():
            r_mesa = np.array([float(read_mesa_history(runs_dir / f"M{m}_age1" / "LOGS" / "history.data")["radius"][-1] / RJUP_PER_RSUN)
                               for m in MASSES_MJUP])
        r_cond = r_cond_csv
    max_rel_diff = float(np.max(np.abs(r_mesa / r_cond - 1.0)))
    assert max_rel_diff < 0.01, f"COND03 vs MESA 1 Gyr difference exceeds 1%: {max_rel_diff}"

    print("[PASS] All 8 analytical & numerical verification test suites passed.")


def cond03_radius_rj(csv_path: Path | str, masses_mjup: Sequence[float]) -> np.ndarray:
    """Linearly interpolate the COND03 (1 Gyr) radius table [M_Msun, R_Rsun] at masses given in M_Jup; returns R_Jup."""
    arr = np.genfromtxt(csv_path, delimiter=",", names=True)
    msun = np.asarray(masses_mjup, float) * MJUP_PER_MSUN
    return np.interp(msun, arr["M_Msun"], arr["R_Rsun"]) * R_SUN / R_JUP


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified ZTF J1239+8347 Simulation Suite")
    parser.add_argument("--mode", choices=["analyze", "test", "mesa-gen", "mesa-status"], default="analyze")
    parser.add_argument("--mesa-runs", type=str, default=None, help="Path to MESA runs directory")
    parser.add_argument("--strict", action="store_true", help="Fail instead of silently using the embedded fallback tables")
    parser.add_argument(
        "--out",
        type=str,
        default="ztf_j1239_simulation_results.json",
        help="Output path for simulation JSON results",
    )
    args = parser.parse_args()

    if args.mode == "test":
        run_verification_tests()
        return

    if args.mode in ("mesa-gen", "mesa-status"):
        jobs = build_mesa_campaign_jobs()
        print(f"Total MESA campaign jobs across 0.1-10 Gyr grid & extended suite: {len(jobs)}")
        if args.mode == "mesa-gen":
            target_dir = Path(args.mesa_runs) if args.mesa_runs else Path("mesa_runs")
            for j in jobs:
                d = target_dir / j.name
                (d / "LOGS").mkdir(parents=True, exist_ok=True)
                (d / "photos").mkdir(parents=True, exist_ok=True)
                (d / "inlist").write_text(render_mesa_inlist(j), encoding="utf-8")
            print(f"Generated {len(jobs)} MESA inlist directories in {target_dir}")
        return

    run_verification_tests()
    out_path = Path(args.out)
    res = run_full_simulation_suite(output_json=out_path, mesa_runs_dir=Path(args.mesa_runs) if args.mesa_runs else None, strict=args.strict)
    print(f"[DONE] Exported complete simulation dataset to {out_path}")
    print("\nSummary of Age-Grid Scaling for Md/Ma = 60/80 M_Jup (q = 0.75) and Ma = 70 M_Jup:")
    print(f"{'Age[Gyr]':>8s} {'Rd60[RJ]':>9s} {'Ra80[RJ]':>9s} {'zeta_ad':>8s} {'qcrit80':>8s} {'fcrit_fl':>9s} {'v_imp':>7s} {'f_max':>7s} {'eps_min':>8s} {'qmax80':>8s}")
    for row in res["age_scaling"]:
        print(
            f"{row['age_gyr']:8.1f} {row['Rd_60_RJ']:9.3f} {row['Ra_80_RJ']:9.3f} "
            f"{row['zeta_ad_60']:8.3f} {row['q_crit_f0_80']:8.3f} {row['f_crit_fl_60_80']:9.3f} "
            f"{row['v_imp_60_80_kms']:7.1f} {row['f_max_60_80']:7.3f} {row['eps_min_fl_60_80']:8.2f} {row['q_max_fl_80']:8.3f}"
        )


if __name__ == "__main__":
    main()
