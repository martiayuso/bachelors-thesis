#!/usr/bin/env python3
"""
nfw_truncated_shell_justification.py

Analytic NFW shell study to justify a truncated-halo / hybrid potential model.

What this script produces
-------------------------
1) Shell mass fraction vs radius
2) Shell contribution to the central potential (and escape-speed scale)
3) Fraction of the potential at selected probe radii coming from material
   outside a chosen cutoff radius
4) A 2D "source shell radius vs probe radius" map of outer-potential fraction
5) A force contribution diagnostic at a small finite radius

Physics note
------------
For an exactly spherical halo:
- Outer shells do NOT contribute to the force at radius r.
- Outer shells DO contribute to the gravitational potential everywhere inside.
So the strongest justification for a truncated model is usually potential/escape-speed
depth and numerical smoothness, while force diagnostics are used to keep the logic honest.

The script is self-contained and does not require SWIFT data.
It is meant to be run before comparing with the actual simulation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # safe for headless execution
import matplotlib.pyplot as plt


# -----------------------------
# NFW profile + useful formulas
# -----------------------------
G = 4.3009e-6  # kpc (km/s)^2 / Msun


def f_nfw(x: np.ndarray | float) -> np.ndarray | float:
    """Dimensionless NFW mass function f(x)=ln(1+x)-x/(1+x)."""
    x = np.asarray(x)
    return np.log1p(x) - x / (1.0 + x)


def nfw_mass_enclosed(r: np.ndarray | float, rho_s: float, r_s: float) -> np.ndarray | float:
    """M(<r) for an NFW halo."""
    r = np.asarray(r)
    x = r / r_s
    return 4.0 * np.pi * rho_s * r_s**3 * f_nfw(x)


def nfw_rho(r: np.ndarray | float, rho_s: float, r_s: float) -> np.ndarray | float:
    """NFW density profile."""
    r = np.asarray(r)
    x = r / r_s
    return rho_s / (x * (1.0 + x) ** 2)


def nfw_total_potential_truncated(
    r: np.ndarray | float, rho_s: float, r_s: float, r_vir: float
) -> np.ndarray | float:
    """
    Truncated NFW potential with Phi(r_vir)=0.

    Phi(r) = -G M(<r)/r - 4 pi G rho_s r_s^2 [1/(1+x) - 1/(1+c)]
    for 0 <= r <= r_vir.
    """
    r = np.asarray(r, dtype=float)
    x = r / r_s
    c = r_vir / r_s

    phi = np.empty_like(x, dtype=float)
    mask = r > 0

    phi[mask] = (
        -G * nfw_mass_enclosed(r[mask], rho_s, r_s) / r[mask]
        - 4.0 * np.pi * G * rho_s * r_s**2 * (1.0 / (1.0 + x[mask]) - 1.0 / (1.0 + c))
    )

    # r -> 0 limit: enclosed-mass term vanishes, outer term remains finite
    phi[~mask] = -4.0 * np.pi * G * rho_s * r_s**2 * (1.0 - 1.0 / (1.0 + c))
    return phi


def nfw_total_force_magnitude(r: np.ndarray | float, rho_s: float, r_s: float) -> np.ndarray | float:
    """Radial acceleration magnitude |a(r)| = G M(<r) / r^2 for spherical symmetry."""
    r = np.asarray(r, dtype=float)
    a = np.empty_like(r)
    mask = r > 0
    a[mask] = G * nfw_mass_enclosed(r[mask], rho_s, r_s) / r[mask] ** 2
    a[~mask] = 0.0
    return a


# ----------------------------------------------------------
# Exact contribution of an NFW shell [r_in, r_out] to Phi, a
# ----------------------------------------------------------
def shell_mass(r_in: float, r_out: float, rho_s: float, r_s: float) -> float:
    """Exact mass contained between r_in and r_out."""
    return float(nfw_mass_enclosed(r_out, rho_s, r_s) - nfw_mass_enclosed(r_in, rho_s, r_s))


def shell_potential(
    r: np.ndarray | float, r_in: float, r_out: float, rho_s: float, r_s: float
) -> np.ndarray | float:
    """
    Potential contribution from the shell [r_in, r_out] at radius r.

    Piecewise exact for a spherically symmetric NFW shell.
    """
    r = np.asarray(r, dtype=float)
    x = r / r_s
    a = r_in / r_s
    b = r_out / r_s

    out = np.empty_like(x, dtype=float)

    # r <= r_in: entire shell lies outside the test point
    mask_outer = r <= r_in
    out[mask_outer] = -G * 4.0 * np.pi * rho_s * r_s**2 * (
        1.0 / (1.0 + a) - 1.0 / (1.0 + b)
    )

    # r >= r_out: entire shell lies inside the test point
    mask_inner = r >= r_out
    m_shell = 4.0 * np.pi * rho_s * r_s**3 * (f_nfw(b) - f_nfw(a))
    out[mask_inner] = -G * m_shell / r[mask_inner]

    # r_in < r < r_out: split the shell into interior and exterior parts
    mask_mid = (~mask_outer) & (~mask_inner)
    if np.any(mask_mid):
        xm = x[mask_mid]
        m_inside = 4.0 * np.pi * rho_s * r_s**3 * (f_nfw(xm) - f_nfw(a))
        exterior = 4.0 * np.pi * rho_s * r_s**2 * (1.0 / (1.0 + xm) - 1.0 / (1.0 + b))
        out[mask_mid] = -(G * (m_inside / r[mask_mid] + exterior))

    return out


def shell_force_magnitude(
    r: np.ndarray | float, r_in: float, r_out: float, rho_s: float, r_s: float
) -> np.ndarray | float:
    """
    Radial acceleration magnitude contributed by the shell [r_in, r_out].

    For spherical symmetry:
    - zero if the shell lies entirely outside the test point
    - only the enclosed part of the shell contributes if the test point is inside the shell
    - full shell mass contributes if the shell lies entirely inside the test point
    """
    r = np.asarray(r, dtype=float)
    x = r / r_s
    a = r_in / r_s
    b = r_out / r_s

    out = np.zeros_like(x, dtype=float)

    # r_in < r < r_out: only material interior to r contributes
    mask_mid = (r > r_in) & (r < r_out)
    if np.any(mask_mid):
        xm = x[mask_mid]
        m_inside = 4.0 * np.pi * rho_s * r_s**3 * (f_nfw(xm) - f_nfw(a))
        out[mask_mid] = G * m_inside / r[mask_mid] ** 2

    # r >= r_out: the whole shell is enclosed
    mask_inner = r >= r_out
    m_shell = 4.0 * np.pi * rho_s * r_s**3 * (f_nfw(b) - f_nfw(a))
    out[mask_inner] = G * m_shell / r[mask_inner] ** 2

    return out


def outer_potential_from_cut(
    r_probe: float, r_cut: float, rho_s: float, r_s: float, r_vir: float
) -> float:
    """
    Potential at probe radius r_probe due only to the mass outside r_cut.
    Exact for a truncated spherical NFW halo.

    This is the quantity you want for the thesis argument:
    "How much of the inner potential depth comes from the outer halo?"
    """
    if r_cut >= r_vir:
        return 0.0

    c = r_vir / r_s
    x_probe = r_probe / r_s
    x_cut = r_cut / r_s

    # No outer material inside the cutoff
    if r_probe <= r_cut:
        return float(
            -4.0 * np.pi * G * rho_s * r_s**2
            * (1.0 / (1.0 + x_cut) - 1.0 / (1.0 + c))
        )

    # r_probe > r_cut: split the outer material into [r_cut, r_probe] and [r_probe, r_vir]
    mass_between = 4.0 * np.pi * rho_s * r_s**3 * (f_nfw(x_probe) - f_nfw(x_cut))
    ext_integral = 4.0 * np.pi * rho_s * r_s**2 * (
        1.0 / (1.0 + x_probe) - 1.0 / (1.0 + c)
    )
    return float(-(G * (mass_between / r_probe + ext_integral)))


def outer_fraction_from_cut(
    r_probe: float, r_cut: float, rho_s: float, r_s: float, r_vir: float
) -> float:
    """Fraction of |Phi(r_probe)| coming from material outside r_cut."""
    phi_tot = nfw_total_potential_truncated(r_probe, rho_s, r_s, r_vir)
    phi_out = outer_potential_from_cut(r_probe, r_cut, rho_s, r_s, r_vir)
    return float(np.abs(phi_out) / np.maximum(np.abs(phi_tot), 1e-30))


def suggest_cutoff_radius(
    r_probe: float,
    thresholds: list[float],
    r_grid: np.ndarray,
    rho_s: float,
    r_s: float,
    r_vir: float,
) -> dict[float, float | None]:
    """
    For a given probe radius, find the smallest cutoff radius for which
    the outer-potential fraction falls below each threshold.
    """
    fracs = np.array([outer_fraction_from_cut(r_probe, rc, rho_s, r_s, r_vir) for rc in r_grid])
    result: dict[float, float | None] = {}
    for th in thresholds:
        idx = np.where(fracs <= th)[0]
        result[th] = float(r_grid[idx[0]]) if idx.size else None
    return result


# -----------------------------------------
# Figure helpers
# -----------------------------------------
def style_log_axes(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", alpha=0.45)


def savefig(fig: plt.Figure, outdir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(outdir / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")


# -----------------------------------------
# Main analysis
# -----------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="NFW shell study for truncated-halo justification.")
    parser.add_argument("--rho-s", type=float, default=1.0e7, help="Characteristic density [Msun/kpc^3].")
    parser.add_argument("--r-s", type=float, default=10.0, help="NFW scale radius [kpc].")
    parser.add_argument("--r-vir", type=float, default=100.0, help="Truncation/virial radius [kpc].")
    parser.add_argument("--r-min", type=float, default=0.02, help="Minimum radius for sampled profiles [kpc].")
    parser.add_argument("--n-shells", type=int, default=80, help="Number of logarithmic shells.")
    parser.add_argument("--outdir", type=str, default="nfw_justification_output", help="Output directory.")
    args = parser.parse_args()

    rho_s = args.rho_s
    r_s = args.r_s
    r_vir = args.r_vir
    r_min = args.r_min
    n_shells = args.n_shells

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Radial grids
    r_grid = np.logspace(np.log10(r_min), np.log10(r_vir), 700)
    shell_edges = np.logspace(np.log10(r_min), np.log10(r_vir), n_shells + 1)
    shell_mid = np.sqrt(shell_edges[:-1] * shell_edges[1:])

    # Shell diagnostics
    shell_m = np.array([shell_mass(shell_edges[i], shell_edges[i + 1], rho_s, r_s) for i in range(n_shells)])
    shell_phi_center = np.array([
        shell_potential(0.0, shell_edges[i], shell_edges[i + 1], rho_s, r_s) for i in range(n_shells)
    ])
    shell_a_probe = np.array([
        shell_force_magnitude(0.5 * r_s, shell_edges[i], shell_edges[i + 1], rho_s, r_s)
        for i in range(n_shells)
    ])

    phi0 = float(nfw_total_potential_truncated(0.0, rho_s, r_s, r_vir))
    esc0 = np.sqrt(np.maximum(-2.0 * phi0, 0.0))

    # ---------------------------------------------------------
    # Plot 1: Mass profile + shell mass fraction
    # ---------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(8.6, 6.2))
    ax1.plot(r_grid, nfw_mass_enclosed(r_grid, rho_s, r_s), lw=2.2, label=r"$M(<r)$")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"Radius $r$ [kpc]")
    ax1.set_ylabel(r"Enclosed mass $M(<r)$ [$M_\odot$]")
    ax1.grid(True, which="both", ls="--", alpha=0.45)
    ax1.set_title("NFW enclosed mass")

    ax2 = ax1.twinx()
    ax2.plot(shell_mid, shell_m / np.sum(shell_m), marker="o", ms=3.5, lw=1.2, alpha=0.85, label="Shell mass fraction")
    ax2.set_yscale("log")
    ax2.set_ylabel("Shell mass fraction")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    savefig(fig, outdir, "01_mass_profile_and_shell_fraction")
    plt.close(fig)

    # ---------------------------------------------------------
    # Plot 2: Shell contribution to the central potential
    # ---------------------------------------------------------
    frac_phi0 = np.abs(shell_phi_center) / np.maximum(np.abs(phi0), 1e-30)
    cum_outer_frac_phi0 = np.array([
        np.sum(np.abs(shell_phi_center[shell_mid >= rc])) / np.maximum(np.abs(phi0), 1e-30)
        for rc in shell_mid
    ])

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    ax.plot(shell_mid, frac_phi0, marker="o", lw=1.5, ms=3.5, label=r"$|\Delta\Phi_{\rm shell}(0)|/|\Phi(0)|$")
    ax.plot(shell_mid, cum_outer_frac_phi0, lw=2.0, label=r"Cumulative outer fraction beyond $r_{\rm cut}$")
    ax.axvline(r_s, color="gray", ls="--", lw=1.3, label=fr"$r_s={r_s:g}$ kpc")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Shell radius [kpc]")
    ax.set_ylabel("Fraction of central potential")
    ax.set_title("How each shell contributes to the central potential / escape-speed scale")
    ax.grid(True, which="both", ls="--", alpha=0.45)
    ax.legend(loc="best")
    savefig(fig, outdir, "02_central_potential_shell_contribution")
    plt.close(fig)

    # ---------------------------------------------------------
    # Plot 3: Outer potential fraction vs cutoff radius
    # ---------------------------------------------------------
    probe_radii = [0.0, 0.2 * r_s, 0.5 * r_s, 1.0 * r_s, 2.0 * r_s, 5.0 * r_s]
    cutoff_grid = np.logspace(np.log10(r_min), np.log10(r_vir), 250)

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    for rp in probe_radii:
        frac = np.array([outer_fraction_from_cut(rp, rc, rho_s, r_s, r_vir) for rc in cutoff_grid])
        label = fr"$r_{{\rm probe}}={rp:g}$ kpc"
        ax.plot(cutoff_grid, frac, lw=2.0, label=label)

    for th in [0.5, 0.1, 0.05, 0.01]:
        ax.axhline(th, color="gray", ls=":", lw=0.9)

    ax.axvline(r_s, color="black", ls="--", lw=1.2, alpha=0.8, label=fr"$r_s={r_s:g}$ kpc")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Cutoff radius $r_{\rm cut}$ [kpc]")
    ax.set_ylabel(r"Outer-potential fraction $|\Phi_{>r_{\rm cut}}|/|\Phi|$")
    ax.set_title("How much of the inner potential comes from the outer halo?")
    ax.grid(True, which="both", ls="--", alpha=0.45)
    ax.legend(loc="best", fontsize=9)
    savefig(fig, outdir, "03_outer_fraction_vs_cutoff")
    plt.close(fig)

    # ---------------------------------------------------------
    # Plot 4: 2D map of outer fraction as a function of probe radius and cutoff
    # ---------------------------------------------------------
    probe_grid = np.logspace(np.log10(r_min), np.log10(r_vir), 180)
    cutoff_grid_2d = np.logspace(np.log10(r_min), np.log10(r_vir), 180)
    frac_map = np.empty((cutoff_grid_2d.size, probe_grid.size), dtype=float)

    for i, rc in enumerate(cutoff_grid_2d):
        frac_map[i, :] = [outer_fraction_from_cut(rp, rc, rho_s, r_s, r_vir) for rp in probe_grid]

    fig, ax = plt.subplots(figsize=(8.7, 6.4))
    levels = np.linspace(0.0, 1.0, 21)
    cf = ax.contourf(probe_grid, cutoff_grid_2d, frac_map, levels=levels, cmap="viridis")
    ax.plot([r_min, r_vir], [r_min, r_vir], color="white", lw=1.4, ls="--", alpha=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Probe radius $r_{\rm probe}$ [kpc]")
    ax.set_ylabel(r"Cutoff radius $r_{\rm cut}$ [kpc]")
    ax.set_title("Outer-potential fraction map")
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label(r"$|\Phi_{>r_{\rm cut}}(r_{\rm probe})| / |\Phi(r_{\rm probe})|$")
    ax.grid(False)
    savefig(fig, outdir, "04_outer_fraction_map")
    plt.close(fig)

    # ---------------------------------------------------------
    # Plot 5: Force contribution diagnostic at a finite radius
    # ---------------------------------------------------------
    r_probe_force = 0.5 * r_s
    shell_force_probe = np.array([
        shell_force_magnitude(r_probe_force, shell_edges[i], shell_edges[i + 1], rho_s, r_s)
        for i in range(n_shells)
    ])
    total_force_probe = float(nfw_total_force_magnitude(r_probe_force, rho_s, r_s))

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    ax.plot(shell_mid, shell_force_probe / np.maximum(total_force_probe, 1e-30), marker="o", lw=1.5, ms=3.5)
    ax.axvline(r_probe_force, color="gray", ls="--", lw=1.3, label=fr"$r_{{\rm probe}}={r_probe_force:g}$ kpc")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Shell radius [kpc]")
    ax.set_ylabel(r"Shell force contribution / total force")
    ax.set_title("Force contribution at a finite inner radius")
    ax.grid(True, which="both", ls="--", alpha=0.45)
    ax.legend(loc="best")
    savefig(fig, outdir, "05_force_contribution_finite_radius")
    plt.close(fig)

    # ---------------------------------------------------------
    # Print a cutoff recommendation table
    # ---------------------------------------------------------
    thresholds = [0.10, 0.05, 0.01]
    rcut_grid = np.logspace(np.log10(r_min), np.log10(r_vir), 600)

    print("\n=== NFW shell-study summary ===")
    print(f"rho_s = {rho_s:.6g} Msun/kpc^3")
    print(f"r_s   = {r_s:.6g} kpc")
    print(f"r_vir = {r_vir:.6g} kpc")
    print(f"Phi(0) = {phi0:.6e} (km/s)^2")
    print(f"v_esc(0) = {esc0:.6e} km/s")
    print("\nSuggested cutoff radii based on the outer-potential fraction:")
    for rp in probe_radii:
        suggestions = suggest_cutoff_radius(rp, thresholds, rcut_grid, rho_s, r_s, r_vir)
        print(f"\n  Probe radius r = {rp:.6g} kpc")
        for th in thresholds:
            val = suggestions[th]
            if val is None:
                print(f"    outer fraction <= {th:5.2%} : not reached")
            else:
                print(f"    outer fraction <= {th:5.2%} : r_cut ≈ {val:.4g} kpc")

    print(f"\nFigures written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
