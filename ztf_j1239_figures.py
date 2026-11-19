#!/usr/bin/env python3
"""
ztf_j1239_figures.py

Generates all six essential vector PDF figures directly from the unified
simulation module (`ztf_j1239_simulations.py`):

  - fig1_roche_geometry.pdf       : Co-rotating Roche geometry, ballistic stream,
                                    and L-dwarf accretor Teff-R_a degeneracy.
  - fig2_cooling_and_contact.pdf  : Unstripped MESA cooling tracks vs. COND03,
                                    secular contraction rate rho, and exact
                                    57.41 min Roche-lobe-filling donor mass.
  - fig3_zeta_age_grid.pdf        : Control-subtracted donor exponent zeta_s
                                    across all 266 MESA runs and pooled age grid.
  - fig4_qcrit_fcrit_vs_age.pdf   : Scaling of q_crit, f_crit, f_max, eps_min,
                                    and (q, f) stability windows across 0.1-10 Gyr.
  - fig5_stability_maps_age.pdf   : 2D (M_a, M_d) stability, energy-budget, and
                                    Roche-contact contour maps at 0.5, 1, 2, 5 Gyr.
  - fig6_mdot_vs_f_and_age.pdf    : Mass-transfer rates |Mdot_d| vs. f and age t_0,
                                    hotspot luminosity band, and young detachment.

Usage:
  python3 ztf_j1239_figures.py --outdir ./figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq

# Ensure local import of unified simulation module works from any working directory
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ztf_j1239_simulations as sim

# Publication typography and color-blind-safe Okabe-Ito palette
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PALETTE = {
    "blue": "#0072B2",
    "verm": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "grey": "#555555",
}


def two_gamma_gw_per_gyr(Md_J: float, Ma_J: float) -> float:
    """2 |Gamma_GW| in Gyr^-1 at P = 57.41 min (J_GW / J_orb), computed rather than hard-coded."""
    Md, Ma = Md_J * sim.M_JUP, Ma_J * sim.M_JUP
    a, _ = sim.orbital_elements(Md, Ma, sim.P_ORB_S)
    Rd = a * float(sim.eggleton_roche_lobe(Md / Ma))
    g_gw, _ = sim.damping_rates_gamma(Md, Ma, sim.P_ORB_S, Rd, 0.9 * sim.R_JUP, 0.0)
    return 2.0 * g_gw * 1.0e9 * sim.YR_SEC


def roche_radii_rj(Md_J: float, Ma_J: float) -> tuple:
    """(R_L,d, R_L,a) in R_Jup for a pair at the observed period."""
    a = sim.binary_geometry(Md_J, Ma_J)[4] / sim.R_JUP
    q = Md_J / Ma_J
    return a * float(sim.eggleton_roche_lobe(q)), a * float(sim.eggleton_accretor_lobe(q))


def plot_fig1_roche_and_accretor_radius(dataset: dict, outdir: Path) -> Path:
    """
    Generate Figure 1: co-rotating Roche geometry & ballistic stream for
    60/80 M_Jup at P = 57.41 min.
    """
    snap_table = dataset["snap_table"]
    fig, ax_single = plt.subplots(1, 1, figsize=(4.9, 3.7))
    axs = [ax_single]

    # Panel (a): Roche geometry in co-rotating frame
    ax = axs[0]
    Md, Ma = 60.0, 80.0
    Mdk, Mak, Mtot, Om, a_orb = sim.binary_geometry(Md, Ma)
    xg = np.linspace(-0.62 * a_orb, 1.62 * a_orb, 360)
    yg = np.linspace(-1.02 * a_orb, 1.02 * a_orb, 360)
    X, Y = np.meshgrid(xg, yg)
    Phi = sim.roche_potential(X, Y, Mdk, Mak, a_orb, Om)
    xL1 = sim.find_L1_position(Mdk, Mak, a_orb, Om)
    Phi_L1 = float(sim.roche_potential(xL1, 0.0, Mdk, Mak, a_orb, Om))
    xL2 = float(brentq(
        lambda x: sim.G_CGS * Mak / (x ** 2) + sim.G_CGS * Mdk / ((x - a_orb) ** 2) - (Om ** 2) * (x - a_orb * Mdk / Mtot),
        a_orb * 1.01, a_orb * 2.5
    ))
    xL3 = float(brentq(
        lambda x: -sim.G_CGS * Mak / (x ** 2) - sim.G_CGS * Mdk / ((x - a_orb) ** 2) - (Om ** 2) * (x - a_orb * Mdk / Mtot),
        -a_orb * 2.5, -a_orb * 0.01
    ))
    Phi_L2 = float(sim.roche_potential(xL2, 0.0, Mdk, Mak, a_orb, Om))
    Phi_L3 = float(sim.roche_potential(xL3, 0.0, Mdk, Mak, a_orb, Om))

    ax.contour(X / a_orb, Y / a_orb, Phi, levels=[Phi_L1], colors=[PALETTE["verm"]], linewidths=1.3, linestyles="--")
    ax.contour(X / a_orb, Y / a_orb, Phi, levels=[Phi_L2, Phi_L3], colors=[PALETTE["blue"]], linewidths=0.85, linestyles="--")

    ra_1g = snap_table["1"][80]["R"] * sim.R_JUP / a_orb
    ra_2g = snap_table["2"][80]["R"] * sim.R_JUP / a_orb
    ra_5g = snap_table["5"][80]["R"] * sim.R_JUP / a_orb
    ra_03g = snap_table["0p3"][80]["R"] * sim.R_JUP / a_orb
    ax.add_patch(plt.Circle((0, 0), ra_1g, color=PALETTE["sky"], alpha=0.42, label=f"accretor 1 Gyr (${snap_table['1'][80]['R']:.2f}\\,R_{{\\rm Jup}}$)"))
    ax.add_patch(plt.Circle((0, 0), ra_2g, fill=False, edgecolor=PALETTE["orange"], lw=1.1, ls="--", label=f"accretor 2 Gyr (${snap_table['2'][80]['R']:.2f}\\,R_{{\\rm Jup}}$)"))
    ax.add_patch(plt.Circle((0, 0), ra_5g, fill=False, edgecolor=PALETTE["blue"], lw=1.3, ls="-", label=f"accretor 5 Gyr (${snap_table['5'][80]['R']:.2f}\\,R_{{\\rm Jup}}$)"))
    ax.add_patch(plt.Circle((0, 0), ra_03g, fill=False, edgecolor=PALETTE["purple"], lw=1.0, ls=":", label=f"accretor 0.3 Gyr (${snap_table['0p3'][80]['R']:.2f}\\,R_{{\\rm Jup}}$)"))

    st_traj = sim.integrate_ballistic_stream(60.0, 80.0, snap_table["5"][80]["R"], return_trajectory=True)
    ax.plot(st_traj["traj_x_over_a"], st_traj["traj_y_over_a"], color=PALETTE["green"], lw=2.0, label="ballistic stream")

    xcm = Mdk / Mtot
    ax.plot([xcm], [0], "k+", ms=6.5)
    ax.text(xcm - 0.08, 0.05, "CM", fontsize=7)
    for lx, ly, lt in (
        (xL1 / a_orb, 0, r"$L_1$"),
        (xL2 / a_orb, 0, r"$L_2$"),
        (xL3 / a_orb, 0, r"$L_3$"),
        (0.5, np.sqrt(3) / 2.0, r"$L_4$"),
        (0.5, -np.sqrt(3) / 2.0, r"$L_5$"),
    ):
        ax.plot([lx], [ly], "ko", ms=2.8)
        ax.text(lx - 0.04, ly - 0.10 if ly <= 0 else ly + 0.04, lt, fontsize=7)

    ax.text(0.0, -0.04, "accretor\n$80\\,M_{\\rm Jup}$", ha="center", va="center", fontsize=7)
    ax.text(1.0, 0.0, "donor\n$60\\,M_{\\rm Jup}$\n(fills lobe)", ha="center", va="center", fontsize=7, color=PALETTE["verm"])
    ax.set(
        xlim=(-0.62, 1.62), ylim=(-1.02, 1.18), aspect="equal",
        xlabel=r"$x/a$", ylabel=r"$y/a$",
        title=r"Roche geometry ($60/80\,M_{\rm Jup}$, $P=57.41$ min)"
    )
    ax.legend(fontsize=5.6, loc="upper left")

    fig.tight_layout()
    out_path = outdir / "fig1_roche_geometry.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fig2_cooling_and_contact(dataset: dict, outdir: Path) -> Path:
    """
    Generate Figure 2: (a) Unstripped MESA cooling tracks & 1 Gyr COND03 points;
    (b) Secular contraction rate rho vs GW driving; (c) Exact 57.41 min Roche-lobe
    contact donor mass M_d,cont(t_0) vs age.
    """
    tracks = dataset["tracks"]
    snap_table = dataset["snap_table"]
    _, radius_fn, _ = sim.make_age_interpolators(dataset)

    fig, axs = plt.subplots(1, 3, figsize=(7.5, 2.65))
    m_cols = {20: PALETTE["blue"], 40: PALETTE["green"], 60: PALETTE["orange"], 80: PALETTE["verm"]}

    for m in sim.MASSES_MJUP:
        tr = tracks[m]
        sel = (tr["t_gyr"] >= 0.08) & (tr["t_gyr"] <= 10.5)
        axs[0].plot(tr["t_gyr"][sel], tr["R"][sel], color=m_cols[m], label=f"${m}\\,M_{{\\rm Jup}}$")
        rho_pts = [snap_table[atag][m]["rho"] for atag in sim.AGE_TAGS]
        axs[1].plot(sim.AGES_GYR, rho_pts, color=m_cols[m], marker="o", ms=3, label=f"${m}\\,M_{{\\rm Jup}}$")

    # Overlay 1 Gyr COND03 benchmark radii
    axs[0].scatter(
        [1.0] * 4, sim.FALLBACK_COND03_1GYR["R_cond_RJ"],
        marker="D", s=18, facecolors="none", edgecolors="k", zorder=5, label="COND03 (1 Gyr)"
    )
    rl_d_6080, rl_a_6080 = roche_radii_rj(60.0, 80.0)
    axs[0].axhline(rl_a_6080, color="k", ls="--", lw=0.8, label=r"$R_{L,\rm a}$ (60/80)")
    axs[0].axhline(rl_d_6080, color=PALETTE["grey"], ls=":", lw=0.8, label=r"$R_{L,\rm d}$ (60/80)")
    axs[0].set(
        xscale="log", xlabel=r"age $t_0$ [Gyr]", ylabel=r"radius $R\ [R_{\rm Jup}]$",
        xlim=(0.09, 11), ylim=(0.72, 2.0), title="(a) Cooling tracks & Roche lobes"
    )
    axs[0].legend(fontsize=5.2, loc="upper right", ncol=2)

    g2_5080 = two_gamma_gw_per_gyr(50.0, 80.0)
    g2_6080 = two_gamma_gw_per_gyr(60.0, 80.0)
    axs[1].axhline(-g2_5080, color="k", ls="--", lw=0.9, label=rf"$-2\Gamma_{{\rm GW}}$ (50/80) $={-g2_5080:.3f}$")
    axs[1].axhline(-g2_6080, color="k", ls=":", lw=0.9, label=rf"$-2\Gamma_{{\rm GW}}$ (60/80) $={-g2_6080:.3f}$")
    axs[1].fill_between([0.08, 12], -2.0, -g2_5080, color=PALETTE["grey"], alpha=0.12)
    axs[1].text(1.2, -1.88, "$|\\rho|>2|\\Gamma_{\\rm GW}|$ (GW-detached)", fontsize=5.8, color=PALETTE["grey"])
    axs[1].set(
        xscale="log", xlabel=r"age $t_0$ [Gyr]", ylabel=r"$\rho=(\partial\ln R/\partial t)_M$ [Gyr$^{-1}$]",
        xlim=(0.09, 11), ylim=(-1.95, 0.1), title="(b) Secular contraction rate"
    )
    axs[1].legend(fontsize=5.0, loc="center right", bbox_to_anchor=(1.0, 0.42))

    t_grid_c = np.geomspace(1.25, 10.0, 60)
    for Ma_c, col_c, ls_c in (
        (80, PALETTE["verm"], "-"),
        (75, PALETTE["orange"], "--"),
        (70, PALETTE["blue"], "-."),
        (65, PALETTE["green"], ":"),
    ):
        mds_c = []
        for tg in t_grid_c:
            fn = lambda md: radius_fn(md, tg) - (sim.binary_geometry(md, Ma_c)[4] * float(sim.eggleton_roche_lobe(md / Ma_c)) / sim.R_JUP)
            try:
                mds_c.append(float(brentq(fn, 25.0, 65.0)))
            except Exception:
                mds_c.append(np.nan)
        axs[2].plot(t_grid_c, mds_c, color=col_c, ls=ls_c, lw=1.5, label=f"$M_{{\\rm a}}={Ma_c}\\,M_{{\\rm Jup}}$")

    axs[2].axhspan(50, 56, color=PALETTE["green"], alpha=0.12, lw=0, label=r"$50$--$56\,M_{\rm Jup}$ ($2$--$5$ Gyr)")
    axs[2].axhline(60, color="k", ls=":", lw=0.8)
    axs[2].set(
        xlabel=r"age $t_0$ [Gyr]", ylabel=r"contact donor mass $M_{\rm d}\ [M_{\rm Jup}]$",
        xlim=(1.0, 10.2), ylim=(44, 66), title=r"(c) Donor mass filling $R_{L,\rm d}$ ($57.4$ min)"
    )
    axs[2].legend(fontsize=5.3, loc="upper right")

    fig.tight_layout()
    out_path = outdir / "fig2_cooling_and_contact.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fig3_zeta_age_grid(dataset: dict, outdir: Path) -> Path:
    """
    Generate Figure 3: (a) Individual MESA mass-stripping curves; (b) Pooled
    adiabatic exponent zeta_ad(M_d; t_0); (c) Two-stage evolution of zeta_ad vs age.
    """
    curves = dataset["curves"]
    age_zeta_ad = dataset["age_zeta_ad"]

    fig, axs = plt.subplots(1, 3, figsize=(7.5, 2.65))
    col_rt = {
        "R1e-6": PALETTE["blue"],
        "R1e-7": PALETTE["sky"],
        "R1e-8": PALETTE["orange"],
        "R1e-9": PALETTE["verm"],
    }
    seen = set()
    if curves:
        for (M0, age, rate), (Mg, z, el) in sorted(curves.items()):
            if rate not in col_rt:
                continue
            ok = (el >= 1.0e9) if rate in ("R1e-8", "R1e-9") else np.ones_like(Mg, dtype=bool)
            if not ok.any():
                continue
            lab = None if rate in seen else rate.replace("R", r"$\dot M=$").replace("e-", r"e$-$") + r" $M_{\rm Jup}$/yr"
            seen.add(rate)
            axs[0].plot(Mg[ok], z[ok], color=col_rt[rate], lw=0.55, alpha=0.55, label=lab)
    else:
        for a in sim.AGES_GYR:
            cen, med, _, _ = age_zeta_ad[a]
            axs[0].plot(cen, med, color=PALETTE["blue"], lw=0.8, alpha=0.7)

    axs[0].axhline(-1.0 / 3.0, color="k", ls=":", lw=0.8)
    axs[0].text(12, -0.38, r"$-1/3$ ($n=1.5$)", fontsize=6.0)
    axs[0].set(
        xlabel=r"$M_{\rm d}\ [M_{\rm Jup}]$", ylabel=r"$\zeta_s = d\ln R_{\rm d}/d\ln M_{\rm d}$",
        xlim=(10, 81), ylim=(-0.45, 1.35), title=r"(a) All stripping runs ($0.1$--$10$ Gyr)"
    )
    axs[0].legend(fontsize=5.5, loc="upper left")

    cmap = plt.get_cmap("viridis")
    norm = mcolors.LogNorm(vmin=0.1, vmax=10.0)
    for a in sim.AGES_GYR:
        col = cmap(norm(a))
        cen, med, _, _ = age_zeta_ad[a]
        ok = np.isfinite(med)
        lbl = f"{a:g} Gyr" if a in (0.1, 0.3, 0.5, 1.0, 3.0, 10.0) else None
        axs[1].plot(cen[ok], med[ok], color=col, lw=1.35, label=lbl)
    axs[1].axhline(-0.2, color="k", ls=":", lw=0.7)
    axs[1].axhline(-1.0 / 3.0, color="k", ls="--", lw=0.7)
    axs[1].set(
        xlabel=r"$M_{\rm d}\ [M_{\rm Jup}]$", ylabel=r"$\zeta_{\rm ad}(M_{\rm d};t_0)$",
        xlim=(11, 80), ylim=(-0.31, -0.04), title=r"(b) Pooled $\zeta_{\rm ad}$ by start age"
    )
    axs[1].legend(fontsize=5.5, loc="upper right", ncol=2)

    for md, c, mk in (
        (20, PALETTE["sky"], "o"),
        (30, PALETTE["blue"], "s"),
        (40, PALETTE["green"], "^"),
        (50, PALETTE["orange"], "v"),
        (60, PALETTE["verm"], "D"),
        (70, PALETTE["purple"], "p"),
    ):
        ys = []
        for a in sim.AGES_GYR:
            cen_a, med_a = age_zeta_ad[a][0], age_zeta_ad[a][1]
            ok_a = np.isfinite(med_a)
            ys.append(float(np.interp(md, cen_a[ok_a], med_a[ok_a])))
        axs[2].plot(sim.AGES_GYR, ys, color=c, marker=mk, ms=3.5, label=f"$M_{{\\rm d}}={md}$")
    axs[2].set(
        xscale="log", xlabel=r"start age $t_0$ [Gyr]", ylabel=r"$\zeta_{\rm ad}$ ($\dot M=10^{-6}$)",
        xlim=(0.09, 11), ylim=(-0.30, -0.02), title=r"(c) $\zeta_{\rm ad}$ vs start age $t_0$"
    )
    axs[2].legend(fontsize=5.5, loc="upper right", ncol=2)

    fig.tight_layout()
    out_path = outdir / "fig3_zeta_age_grid.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fig4_qcrit_fcrit_vs_age(dataset: dict, outdir: Path) -> Path:
    """
    Generate Figure 4: 4-panel scaling of q_crit, f_crit, stream energy ceiling
    f_max, required efficiency eps_min, and (q, f) stability windows across age.
    """
    zeta_ad_fn, radius_fn, _ = sim.make_age_interpolators(dataset)
    fig, axs = plt.subplots(2, 2, figsize=(7.5, 5.4))

    qc_80, qc_70 = [], []
    qmax_fl_80, qmax_fl_dyn_80, qmax_pt_80, qmax_fl_70 = [], [], [], []
    for a in sim.AGES_GYR:
        qc_80.append(sim.critical_mass_ratio(lambda md: zeta_ad_fn(md, a), Ma_mjup=80.0))
        qc_70.append(sim.critical_mass_ratio(lambda md: zeta_ad_fn(md, a), Ma_mjup=70.0))
        qs2 = np.arange(0.60, 0.82, 0.002)
        qm_fl = qm_fl_d = qm_pt = qm_70 = np.nan
        for qq in qs2:
            md_q, ma_q = qq * 140.0 / (1.0 + qq), 140.0 / (1.0 + qq)
            ra_q = radius_fn(ma_q, a)
            st_q = sim.integrate_ballistic_stream(md_q, ma_q, ra_q)
            if st_q["hit"]:
                fm_q = (float(st_q["v_rot"]) / sim.escape_velocity(ma_q, ra_q)) ** 2
                _, _, _, _, a_q = sim.binary_geometry(md_q, ma_q)
                rho_q_pt = float(sim.ejection_site_rho2(qq, ra_q * sim.R_JUP / a_q, "point"))
                f0 = sim.critical_mass_loss_fraction(qq, zeta_ad_fn(md_q, a), "isotropic", rho2=0.0)
                fd = sim.critical_mass_loss_fraction(qq, zeta_ad_fn(md_q, a) - sim.DELTA_DYN_MEAS, "isotropic", rho2=0.0)
                fp = sim.critical_mass_loss_fraction(qq, zeta_ad_fn(md_q, a), "isotropic", rho2=rho_q_pt)
                if int(f0["status"][0]) != 2 and max(float(f0["f"][0]), 0.0) <= fm_q:
                    qm_fl = float(qq)
                if int(fd["status"][0]) != 2 and max(float(fd["f"][0]), 0.0) <= fm_q:
                    qm_fl_d = float(qq)
                if int(fp["status"][0]) != 2 and max(float(fp["f"][0]), 0.0) <= fm_q:
                    qm_pt = float(qq)
            md_70 = qq * 70.0
            ra_70 = radius_fn(70.0, a)
            st_70 = sim.integrate_ballistic_stream(md_70, 70.0, ra_70)
            if st_70["hit"]:
                fm_70 = (float(st_70["v_rot"]) / sim.escape_velocity(70.0, ra_70)) ** 2
                f70 = sim.critical_mass_loss_fraction(qq, zeta_ad_fn(md_70, a), "isotropic", rho2=0.0)
                if int(f70["status"][0]) != 2 and max(float(f70["f"][0]), 0.0) <= fm_70:
                    qm_70 = float(qq)
        qmax_fl_80.append(qm_fl)
        qmax_fl_dyn_80.append(qm_fl_d)
        qmax_pt_80.append(qm_pt)
        qmax_fl_70.append(qm_70)

    ax = axs[0, 0]
    ax.plot(sim.AGES_GYR, qc_80, color="k", marker="o", ms=3, ls=":", label=r"$q_{\rm crit}^{f=0}$ ($M_{\rm a}=80$)")
    ax.plot(sim.AGES_GYR, qc_70, color=PALETTE["grey"], marker="s", ms=2.8, ls=":", label=r"$q_{\rm crit}^{f=0}$ ($M_{\rm a}=70$)")
    ax.plot(sim.AGES_GYR, qmax_pt_80, color=PALETTE["purple"], marker="v", ms=3, ls="-.", label=r"$q_{\max}$ point ($M_{\rm tot}=140$)")
    ax.plot(sim.AGES_GYR, qmax_fl_dyn_80, color=PALETTE["verm"], marker="^", ms=3.2, ls="--", label=r"$q_{\max}$ floor ($\delta=0.025$)")
    ax.plot(sim.AGES_GYR, qmax_fl_80, color=PALETTE["blue"], marker="D", ms=3.5, lw=1.7, label=r"$q_{\max}$ floor ($\delta=0$, $M_{\rm tot}=140$)")
    ax.plot(sim.AGES_GYR, qmax_fl_70, color=PALETTE["green"], marker="o", ms=3.2, lw=1.4, label=r"$q_{\max}$ floor ($\delta=0$, $M_{\rm a}=70$)")
    ax.axhline(0.75, color="k", ls="-", lw=0.7, alpha=0.6)
    ax.text(0.11, 0.752, r"$q=0.75$ (60/80)", fontsize=5.8)
    ax.set(
        xscale="log", xlabel=r"system age $t_0$ [Gyr]", ylabel=r"critical mass ratio $q$",
        xlim=(0.09, 11), ylim=(0.66, 0.81), title=r"(a) $q_{\rm crit}^{f=0}(t_0)$ & $q_{\max}(\epsilon=1;t_0)$ vs age"
    )
    ax.legend(fontsize=5.0, loc="upper left", ncol=2)

    # Panel (b): f_crit and f_max vs age
    ax = axs[0, 1]
    cand_b = [
        ((55, 80), PALETTE["green"], "o", "55/80 ($q=0.69$)"),
        ((50, 70), PALETTE["sky"], "s", "50/70 ($q=0.71$)"),
        ((55, 75), PALETTE["orange"], "^", "55/75 ($q=0.73$)"),
        ((60, 80), PALETTE["blue"], "D", "60/80 ($q=0.75$)"),
        ((60, 70), PALETTE["purple"], "v", "60/70 ($q=0.86$)"),
    ]
    for (md_b, ma_b), col_b, mk_b, lab_b in cand_b:
        q_b = md_b / ma_b
        fc_vals = [max(0.0, float(sim.critical_mass_loss_fraction(q_b, zeta_ad_fn(md_b, a), "isotropic", rho2=0.0)["f"][0])) for a in sim.AGES_GYR]
        ax.plot(sim.AGES_GYR, fc_vals, color=col_b, marker=mk_b, ms=3.2, ls="--", label=f"$f_{{\\rm crit}}^{{\\rm fl}}$ {lab_b}")

    fmax_80, fmax_70 = [], []
    for a in sim.AGES_GYR:
        st80 = sim.integrate_ballistic_stream(60.0, 80.0, radius_fn(80.0, a))
        fmax_80.append((float(st80["v_rot"]) / sim.escape_velocity(80.0, radius_fn(80.0, a))) ** 2 if st80["hit"] else np.nan)
        st70 = sim.integrate_ballistic_stream(50.0, 70.0, radius_fn(70.0, a))
        fmax_70.append((float(st70["v_rot"]) / sim.escape_velocity(70.0, radius_fn(70.0, a))) ** 2 if st70["hit"] else np.nan)

    fmax_80_bin = []
    for a in sim.AGES_GYR:
        e_b = sim.stream_energetics(60.0, 80.0, radius_fn(80.0, a), escape_model="binary")
        fmax_80_bin.append(e_b["f_max"] if e_b["hit"] else np.nan)
    ax.plot(sim.AGES_GYR, fmax_80_bin, color=PALETTE["verm"], marker="o", ms=2.5, lw=1.0, ls=":",
            label=r"$f_{\max}$, binary-potential $v_{\rm esc}$ ($M_{\rm a}=80$)")
    ax.plot(sim.AGES_GYR, fmax_80, color=PALETTE["verm"], marker="o", ms=3.5, lw=1.8, label=r"$f_{\max}(\epsilon=1)$ ($M_{\rm a}=80$)")
    ax.plot(sim.AGES_GYR, fmax_70, color="k", marker="s", ms=3.0, lw=1.4, ls="-", label=r"$f_{\max}(\epsilon=1)$ ($M_{\rm a}=70$)")
    ax.set(
        xscale="log", xlabel=r"system age $t_0$ [Gyr]", ylabel=r"ejected fraction $f$",
        xlim=(0.09, 11), ylim=(0.0, 0.38), title=r"(b) $f_{\rm crit}^{\rm fl}(t_0)$ vs stream ceiling $f_{\max}(t_0)$"
    )
    ax.legend(fontsize=4.9, loc="upper left", ncol=2)

    # Panel (c): Minimum required efficiency eps_min vs age
    ax = axs[1, 0]
    cand_c = [
        ((55, 80), PALETTE["green"], "o", "-", "55/80 floor ($q=0.69$)"),
        ((50, 70), PALETTE["sky"], "s", "-", "50/70 floor ($q=0.71$)"),
        ((55, 75), PALETTE["orange"], "^", "-", "55/75 floor ($q=0.73$)"),
        ((52, 70), PALETTE["purple"], "v", "--", "52/70 floor ($q=0.74$)"),
        ((60, 80), PALETTE["blue"], "D", "-", r"60/80 floor ($\delta=0$)"),
        ((60, 80), PALETTE["verm"], "p", ":", r"60/80 floor ($\delta=0.025$)"),
        ((60, 70), PALETTE["grey"], "x", "-.", "60/70 floor ($q=0.86$)"),
    ]
    for (md_c, ma_c), col_c, mk_c, ls_c, lab_c in cand_c:
        q_c = md_c / ma_c
        eps_vals = []
        for a in sim.AGES_GYR:
            zs_c = zeta_ad_fn(md_c, a) - (sim.DELTA_DYN_MEAS if "0.025" in lab_c else 0.0)
            fc_v = max(0.0, float(sim.critical_mass_loss_fraction(q_c, zs_c, "isotropic", rho2=0.0)["f"][0]))
            ra_c = radius_fn(ma_c, a)
            st_c = sim.integrate_ballistic_stream(md_c, ma_c, ra_c)
            if st_c["hit"]:
                fm_v = (float(st_c["v_rot"]) / sim.escape_velocity(ma_c, ra_c)) ** 2
                eps_vals.append(fc_v / fm_v)
            else:
                eps_vals.append(np.nan)
        ax.plot(sim.AGES_GYR, eps_vals, color=col_c, marker=mk_c, ms=3.2, ls=ls_c, lw=1.4, label=lab_c)
    ax.axhline(1.0, color="k", lw=1.0, ls="-")
    ax.fill_between([0.2, 12], 0.05, 1.0, color=PALETTE["green"], alpha=0.10)
    ax.text(0.32, 0.22, r"energetically allowed ($\epsilon_{\min}\leq 1$)", fontsize=6.0, color="darkgreen")
    ax.set(
        xscale="log", yscale="log", xlabel=r"system age $t_0$ [Gyr]",
        ylabel=r"required efficiency $\epsilon_{\min} = f_{\rm crit}/f_{\max}$",
        xlim=(0.25, 11), ylim=(0.15, 25), title=r"(c) Required efficiency $\epsilon_{\min}(t_0)$ by pair"
    )
    ax.legend(fontsize=4.9, loc="upper right", ncol=2)

    # Panel (d): (q, f) stability & energy window across ages
    ax = axs[1, 1]
    qs_d = np.linspace(0.62, 0.84, 111)
    for a_d, col_d, ls_d in (
        (0.5, PALETTE["purple"], ":"),
        (1.0, PALETTE["verm"], "--"),
        (2.0, PALETTE["orange"], "-."),
        (5.0, PALETTE["blue"], "-"),
    ):
        fc_curve, fm_curve = [], []
        for qq in qs_d:
            md_q, ma_q = qq * 140.0 / (1.0 + qq), 140.0 / (1.0 + qq)
            fc_q = sim.critical_mass_loss_fraction(qq, zeta_ad_fn(md_q, a_d), "isotropic", rho2=0.0)
            fc_curve.append(np.nan if int(fc_q["status"][0]) == 2 else float(max(fc_q["f"][0], 0.0)))
            ra_q = radius_fn(ma_q, a_d)
            st_q = sim.integrate_ballistic_stream(md_q, ma_q, ra_q)
            fm_curve.append((float(st_q["v_rot"]) / sim.escape_velocity(ma_q, ra_q)) ** 2 if st_q["hit"] else np.nan)
        ax.plot(qs_d, fc_curve, color=col_d, ls=ls_d, lw=1.4, label=f"$f_{{\\rm crit}}^{{\\rm fl}}$ ({a_d:g} Gyr)")
        ax.plot(qs_d, fm_curve, color=col_d, ls="-", lw=1.1, alpha=0.8, label=f"$f_{{\\max}}$ ({a_d:g} Gyr)")
    ax.axvline(0.75, color="k", ls="-.", lw=0.7)
    ax.set(
        xlabel=r"mass ratio $q = M_{\rm d}/M_{\rm a}$", ylabel=r"ejected fraction $f$",
        xlim=(0.62, 0.84), ylim=(0.0, 0.25), title=r"(d) $(q,f)$ stability & energy window by age"
    )
    ax.legend(fontsize=4.9, loc="upper left", ncol=2)

    fig.tight_layout()
    out_path = outdir / "fig4_qcrit_fcrit_vs_age.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fig5_stability_maps_age(dataset: dict, outdir: Path) -> Path:
    """
    Generate Figure 5: 2D (M_a, M_d) stability, stream energy-budget, and
    exact 57.41 min Roche-lobe contact contour maps across t_0 = 0.5, 1, 2, 5 Gyr.
    """
    zeta_ad_fn, radius_fn, _ = sim.make_age_interpolators(dataset)
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 6.2), sharex=True, sharey=True)
    grid_m = np.linspace(35.0, 80.0, 181)
    Ma_g, Md_g = np.meshgrid(grid_m, grid_m)
    Q_g = Md_g / Ma_g

    for ax, a_m in zip(axs.flat, (0.5, 1.0, 2.0, 5.0)):
        Z_g = np.vectorize(lambda md: zeta_ad_fn(md, a_m))(Md_g)
        Z0_g, S_g = sim.roche_lobe_exponent_coeffs(Q_g, "isotropic", rho2=0.0)
        fc_g = np.maximum(0.0, (Z_g - Z0_g) / S_g)
        fc_g = np.where(Z0_g <= Z_g, 0.0, fc_g)
        fc_g = np.where(fc_g > 1.0, np.nan, fc_g)

        g_coarse = np.linspace(35.0, 80.0, 13)
        fm_coarse = np.zeros((len(g_coarse), len(g_coarse)))
        for i_d, md_v in enumerate(g_coarse):
            for j_a, ma_v in enumerate(g_coarse):
                ra_v = radius_fn(ma_v, a_m)
                en_v = sim.stream_energetics(md_v, ma_v, ra_v)  # integrated ballistic stream (co-rotating v_rot)
                fm_coarse[i_d, j_a] = en_v["f_max"] if en_v["hit"] else 0.0

        interp_fm = RegularGridInterpolator((g_coarse, g_coarse), fm_coarse)
        pts = np.stack([Md_g.ravel(), Ma_g.ravel()], axis=-1)
        fm_g = interp_fm(pts).reshape(Md_g.shape)
        eps_g = np.where(fm_g > 0, fc_g / fm_g, np.nan)

        cm_v = plt.get_cmap("viridis").copy()
        cm_v.set_bad("#e0e0e0")
        im = ax.pcolormesh(grid_m, grid_m, np.ma.masked_invalid(fc_g), vmin=0.0, vmax=0.5, cmap=cm_v, shading="auto", rasterized=True)
        ax.contour(grid_m, grid_m, np.nan_to_num(fc_g, nan=9.0), levels=[1e-6], colors="w", linewidths=1.3)
        ax.contour(grid_m, grid_m, np.nan_to_num(eps_g, nan=99.0), levels=[1.0], colors=[PALETTE["sky"]], linewidths=1.5, linestyles="--")

        ma_line = np.linspace(45.0, 80.0, 70)
        md_contact_line = []
        for mav in ma_line:
            fn_c = lambda mdv: radius_fn(mdv, a_m) - (sim.binary_geometry(mdv, mav)[4] * float(sim.eggleton_roche_lobe(mdv / mav)) / sim.R_JUP)
            try:
                md_contact_line.append(float(brentq(fn_c, 25.0, 79.0)))
            except Exception:
                md_contact_line.append(np.nan)
        ax.plot(ma_line, md_contact_line, color="#ff3333", lw=1.8, ls="-", label=r"$R_{\rm d}=R_{L,\rm d}$")

        ax.plot([60, 80, 80, 60, 60], [60, 60, 80, 80, 60], color=PALETTE["verm"], lw=1.2, ls="-")
        ax.plot([50, 70, 70, 50, 50], [50, 50, 70, 70, 50], color=PALETTE["orange"], lw=1.2, ls=":")
        ax.plot([35, 80], [35, 80], "k:", lw=0.6)
        ax.set(xlim=(35, 80), ylim=(35, 80), title=f"$t_0 = {a_m:g}$ Gyr")

    for a in axs[1]:
        a.set_xlabel(r"$M_{\rm a}\ [M_{\rm Jup}]$")
    for a in axs[:, 0]:
        a.set_ylabel(r"$M_{\rm d}\ [M_{\rm Jup}]$")
    fig.colorbar(im, ax=axs, shrink=0.82, label=r"$f_{\rm crit}^{\rm fl}$ (zero-AM floor)")

    out_path = outdir / "fig5_stability_maps_age.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fig6_mdot_vs_f_and_age(dataset: dict, outdir: Path) -> Path:
    """
    Generate Figure 6: Mass-transfer rate |Mdot_d| vs ejected fraction f (GW only
    and GW + saturated MB) and vs system age t_0 with and without donor contraction rho.
    """
    zeta_ad_fn, radius_fn, _ = sim.make_age_interpolators(dataset)
    snap_table = dataset["snap_table"]

    Llo = sim.hotspot_bolometric_luminosity(R_bb_J=sim.R_BB_RANGE_JUP[0])
    Lhi = sim.hotspot_bolometric_luminosity(R_bb_J=sim.R_BB_RANGE_JUP[1])
    # impact speeds of the integrated stream for the candidate pairs at 1-5 Gyr
    v_all = []
    for (md_v, ma_v) in ((50, 80), (55, 80), (50, 70), (55, 75), (60, 80)):
        for a_v in (1.0, 2.0, 3.0, 5.0):
            en_v = sim.stream_energetics(md_v, ma_v, radius_fn(ma_v, a_v))
            if en_v["hit"]:
                v_all.append(en_v["v_imp"] / sim.KMS_CMS)
    v_lo, v_hi = float(min(v_all)), float(max(v_all))
    # Mdot >= 2 L / v^2: smallest for the lowest L and highest v, largest for the highest L and lowest v
    MLR = [sim.mdot_from_hotspot_luminosity(Llo, v_hi), sim.mdot_from_hotspot_luminosity(Lhi, v_lo)]

    systems = [(80, 60), (80, 55), (70, 50), (80, 50), (80, 40)]
    fgrid = np.linspace(0.0, 1.0, 501)
    fig, axs = plt.subplots(1, 3, figsize=(7.5, 2.65))
    cols_s = [PALETTE["blue"], PALETTE["green"], PALETTE["sky"], PALETTE["orange"], PALETTE["purple"]]

    for (Ma_s, Md_s), c_s in zip(systems, cols_s):
        zs_s = zeta_ad_fn(Md_s, 1.0)
        q_s = Md_s / Ma_s
        for ch, ax in ((("GW",), axs[0]), (("GW", "MB"), axs[1])):
            m_s = np.array([
                abs(sim.steady_state_mass_transfer_rate(Md_s, Ma_s, ff, "isotropic", zs_s, Ra_J=radius_fn(Ma_s, 1.0), channels=ch))
                for ff in fgrid
            ])
            ax.plot(fgrid, m_s, color=c_s, label=f"${Md_s}/{Ma_s}$ ($q={q_s:.2f}$)")

    for ax, t_s in ((axs[0], "(a) Gravitational radiation only"), (axs[1], r"(b) GW + saturated MB ($\eta_{\rm MB}=1$)")):
        ax.set(yscale="log", xlabel=r"ejected fraction $f$", title=t_s, ylim=(1e-10, 1e-3), xlim=(0, 1))
        ax.axhline(1e-9, color="k", ls=":", lw=0.9)
        ax.axhspan(MLR[0], MLR[1], color=PALETTE["grey"], alpha=0.18, lw=0)

    axs[0].set_ylabel(r"$|\dot M_{\rm d}|\ [M_{\rm Jup}\,{\rm yr}^{-1}]$")
    axs[0].legend(fontsize=5.5, loc="upper right")
    axs[1].legend(fontsize=5.5, loc="upper right")
    axs[0].text(0.55, 1.4e-9, r"$10^{-9}\,M_{\rm Jup}\,{\rm yr}^{-1}$", fontsize=6.0)
    axs[0].text(0.40, 1.4 * MLR[1], r"min. $\dot M$ to power $L_{\rm spot}$ ($\epsilon_{\rm rad}{=}1$)", fontsize=5.5)

    m50_gw_0, m50_gw_r, m60_gw_0, m60_gw_r, m60_mb_r = [], [], [], [], []
    for a, atag in zip(sim.AGES_GYR, sim.AGE_TAGS):
        Ra = radius_fn(80.0, a)
        rho40 = snap_table[atag][40]["rho"]
        rho60 = snap_table[atag][60]["rho"]
        v1 = sim.steady_state_mass_transfer_rate(50.0, 80.0, 0.0, "isotropic", zeta_ad_fn(50.0, a), Ra_J=Ra, channels=("GW",), rho_per_Gyr=0.0)
        v2 = sim.steady_state_mass_transfer_rate(50.0, 80.0, 0.0, "isotropic", zeta_ad_fn(50.0, a), Ra_J=Ra, channels=("GW",), rho_per_Gyr=rho40)
        v3 = sim.steady_state_mass_transfer_rate(60.0, 80.0, 0.6, "isotropic", zeta_ad_fn(60.0, a), Ra_J=Ra, channels=("GW",), rho_per_Gyr=0.0)
        v4 = sim.steady_state_mass_transfer_rate(60.0, 80.0, 0.6, "isotropic", zeta_ad_fn(60.0, a), Ra_J=Ra, channels=("GW",), rho_per_Gyr=rho60)
        v5 = sim.steady_state_mass_transfer_rate(60.0, 80.0, 0.6, "isotropic", zeta_ad_fn(60.0, a), Ra_J=Ra, channels=("GW", "MB"), rho_per_Gyr=rho60)
        m50_gw_0.append(abs(v1))
        m50_gw_r.append(abs(v2) if v2 < 0 else np.nan)
        m60_gw_0.append(abs(v3))
        m60_gw_r.append(abs(v4) if v4 < 0 else np.nan)
        m60_mb_r.append(abs(v5))

    axs[2].plot(sim.AGES_GYR, m50_gw_0, color=PALETTE["orange"], ls="--", marker="o", ms=3, label=r"50/80 GW ($\rho=0$)")
    axs[2].plot(sim.AGES_GYR, m50_gw_r, color=PALETTE["orange"], ls="-", marker="s", ms=3.5, label=r"50/80 GW (with $\rho$)")
    axs[2].plot(sim.AGES_GYR, m60_gw_0, color=PALETTE["blue"], ls="--", marker="^", ms=3, label=r"60/80 $f{=}0.6$ GW ($\rho=0$)")
    axs[2].plot(sim.AGES_GYR, m60_gw_r, color=PALETTE["blue"], ls="-", marker="D", ms=3.5, label=r"60/80 $f{=}0.6$ GW (with $\rho$)")
    axs[2].plot(sim.AGES_GYR, m60_mb_r, color=PALETTE["verm"], ls="-", marker="p", ms=3.5, label=r"60/80 $f{=}0.6$ GW+MB")
    axs[2].axhspan(MLR[0], MLR[1], color=PALETTE["grey"], alpha=0.18, lw=0)
    axs[2].set(
        xscale="log", yscale="log", xlabel=r"system age $t_0$ [Gyr]",
        ylabel=r"$|\dot M_{\rm d}|\ [M_{\rm Jup}\,{\rm yr}^{-1}]$",
        xlim=(0.09, 11), ylim=(2e-9, 2e-5), title=r"(c) $|\dot M_{\rm d}|$ vs age & contraction $\rho$"
    )
    axs[2].legend(fontsize=5.0, loc="upper left")

    fig.tight_layout()
    out_path = outdir / "fig6_mdot_vs_f_and_age.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def generate_all_figures(outdir: Path, mesa_runs_dir: Path | None = None, strict: bool = False) -> list[Path]:
    """Generate all six vector PDF figures for the ZTF J1239+8347 research paper."""
    outdir.mkdir(parents=True, exist_ok=True)
    dataset = sim.load_or_build_age_grid_dataset(mesa_runs_dir, strict=strict)
    paths = [
        plot_fig1_roche_and_accretor_radius(dataset, outdir),
        plot_fig2_cooling_and_contact(dataset, outdir),
        plot_fig3_zeta_age_grid(dataset, outdir),
        plot_fig4_qcrit_fcrit_vs_age(dataset, outdir),
        plot_fig5_stability_maps_age(dataset, outdir),
        plot_fig6_mdot_vs_f_and_age(dataset, outdir),
    ]
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate vector PDF figures for ZTF J1239+8347 paper")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory for vector PDF figures")
    parser.add_argument("--mesa-runs", type=str, default=None, help="Optional path to MESA runs directory")
    parser.add_argument("--strict", action="store_true", help="Fail instead of falling back to embedded tables if raw MESA runs are missing")
    args = parser.parse_args()

    out_paths = generate_all_figures(Path(args.outdir), Path(args.mesa_runs) if args.mesa_runs else None, strict=args.strict)
    for p in out_paths:
        print(f"[GENERATED] {p}")


if __name__ == "__main__":
    main()
