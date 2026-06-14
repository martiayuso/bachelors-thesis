#!/usr/bin/env python3
"""
kinematic_heating_analysis.py

Compute:
    sigma_1d(r) = sqrt(1/3 * (sigma_r^2 + sigma_t^2))
    beta(r) = 1 - sigma_t^2/(2 sigma_r^2)

python3 kin_heating.py --rmax 20 --nr 664 --output kinematic_profiles
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.signal import savgol_filter  # Smooths profiles while preserving local curves

from pNbody import *


# ============================================================
# Radial binning identical to plotSphericalProfile
# ============================================================

def build_bins(rmax, nr):
    """
    Create logarithmically compressed spherical bins similar to
    plotSphericalProfile.
    """
    rc = 1.0

    def f(r):
        return np.log(r / rc + 1.0)

    def fm(x):
        return rc * (np.exp(x) - 1.0)

    x = np.linspace(f(0.0), f(rmax), nr + 1)
    edges = fm(x)
    centers = 0.5 * (edges[:-1] + edges[1:])

    return edges, centers


# ============================================================
# Main profile computation
# ============================================================

def compute_kinematic_profiles(nb, rmax=10.0, nr=64):
    """
    Compute sigma_1d(r), sigma_r(r), sigma_t(r), and beta(r).
    Locks the bulk frame of reference to the innermost core particles.
    """
    # Extract positions and velocities
    pos = nb.Pos(units="kpc")
    vel = nb.Vel(units="km/s")

    # --------------------------------------------------------
    # Core-locked bulk velocity correction
    # --------------------------------------------------------
    r_initial = np.linalg.norm(pos, axis=1)
    core_mask = r_initial < 1.0
    
    if np.count_nonzero(core_mask) > 10:
        v_bulk = np.mean(vel[core_mask], axis=0)
    else:
        v_bulk = np.mean(vel, axis=0)  # Fallback to global if core empty
        
    vel -= v_bulk

    # Recompute radii relative to center following bulk correction
    r = np.linalg.norm(pos, axis=1)
    valid = r > 0.0

    pos = pos[valid]
    vel = vel[valid]
    r = r[valid]

    # --------------------------------------------------------
    # Geometric Projection (Clean frame vector manipulation)
    # --------------------------------------------------------
    r_hat = pos / r[:, np.newaxis]
    v_r = np.sum(vel * r_hat, axis=1)

    v2_total = np.sum(vel**2, axis=1)      # v_x^2 + v_y^2 + v_z^2
    v2_radial = v_r**2                     # v_r^2
    v2_tangential = v2_total - v2_radial   # v_t^2 = v_theta^2 + v_phi^2

    # --------------------------------------------------------
    # Radial bins calculation
    # --------------------------------------------------------
    edges, r_centers = build_bins(rmax, nr)

    sigma_1d = np.full(nr, np.nan)
    sigma_r = np.full(nr, np.nan)
    sigma_t = np.full(nr, np.nan)
    beta = np.full(nr, np.nan)
    counts = np.zeros(nr, dtype=int)

    for i in range(nr):
        mask = (r >= edges[i]) & (r < edges[i + 1])
        n = np.count_nonzero(mask)
        counts[i] = n

        if n < 10:
            continue

        vr2_bin = v2_radial[mask]
        vt2_bin = v2_tangential[mask]

        sigma_r2 = np.mean(vr2_bin)
        sigma_t2 = np.mean(vt2_bin)

        sigma_r[i] = np.sqrt(sigma_r2)
        sigma_t[i] = np.sqrt(sigma_t2)
        
        sigma_1d[i] = np.sqrt((sigma_r2 + sigma_t2) / 3.0)

        if sigma_r2 > 0:
            beta[i] = 1.0 - sigma_t2 / (2.0 * sigma_r2)

    # --------------------------------------------------------
    # Savitzky-Golay Smoothing Implementation
    # --------------------------------------------------------
    # window_length must be an odd integer less than total bins (nr).
    # polyorder=2 fits local parabolic variations cleanly.
    window_length = 101
    polyorder = 2

    def smooth_profile(arr):
        # Create a clean local copy to avoid contaminating raw arrays
        smoothed_arr = np.copy(arr)
        bad_mask = np.isnan(smoothed_arr)
        
        if np.all(bad_mask):
            return smoothed_arr
            
        # Linearly interpolate missing data points so the filter moves cleanly across boundaries
        if np.any(bad_mask):
            x_indices = np.arange(len(smoothed_arr))
            smoothed_arr[bad_mask] = np.interp(
                x_indices[bad_mask], x_indices[~bad_mask], smoothed_arr[~bad_mask]
            )
            
        return savgol_filter(smoothed_arr, window_length=window_length, polyorder=polyorder)

    beta_smoothed = smooth_profile(beta)
    sigma_1d_smoothed = smooth_profile(sigma_1d)

    return {
        "radius": r_centers,
        "sigma_1d": sigma_1d_smoothed,
        "sigma_1d_raw": sigma_1d,
        "sigma_r": sigma_r,
        "sigma_t": sigma_t,
        "beta": beta_smoothed,
        "beta_raw": beta,
        "counts": counts,
    }


# ============================================================
# Plotting
# ============================================================

def make_plots(results, output_prefix):
    """
    Generate a publication-quality 2-panel figure using LaTeX and rainbow color mappings.
    """
    # Apply your specific typographic and scaling configuration
    params = {
        "text.usetex": True,
        "font.size": 12,
        "axes.labelsize": 16,
        "legend.fontsize": 12,
        "figure.figsize": (10, 7)
    }
    plt.rcParams.update(params)

    fig, axes = plt.subplots(
        2,
        1,
        sharex=True
    )

    # Replicate your exact rainbow sequence mapping context
    colors = cm.rainbow(np.linspace(0, 1, len(results)))

    for i, (label, data) in enumerate(results.items()):
        r = data["radius"]
        
        axes[0].plot(r, data["sigma_1d"], label=label, color=colors[i], linewidth=1.5)
        axes[1].plot(r, data["beta"], label=label, color=colors[i], linewidth=1.5)

    # Formatting and Math LaTeX Labels
    axes[0].set_ylabel(r'$\sigma_{\rm{1D}}\ [\rm{km/s}]$')
    axes[1].set_ylabel(r'$\beta$')
    axes[1].set_xlabel(r'$r\ [\rm{kpc}]$')

    # Scaling and Limit Bounds
    axes[0].set_xscale('log')
    axes[1].set_xscale('log')
    axes[1].set_xlim(2.8 * 0.01, 10)
    
    # Constrain beta y-axis bounds to shield from outer truncation chatter
    axes[1].set_ylim(-0.6, 0.6)

    # Grids
    axes[0].grid(alpha=0.2)
    axes[1].grid(alpha=0.2)

    axes[0].set_title(
        r"Velocity Dispersion ($\sigma_{\rm{1D}}$) and Orbital Anisotropy ($\beta$) for $\epsilon = 0.01\ \mathrm{kpc}$"
    )

    axes[0].legend(loc="best", frameon=True, ncol=2)

    plt.tight_layout()
    outfile = f"{output_prefix}_kinematics.png"
    plt.savefig(outfile, dpi=300)
    print(f"Saved figure: {outfile}")
    plt.show()


# ============================================================
# Main Execution Entry
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Kinematic profiles code running on fixed snapshot configurations with Savitzky-Golay filtering."
    )
    parser.add_argument(
        "--rmax",
        type=float,
        default=10.0,
        help="Maximum binning radius [kpc]"
    )
    parser.add_argument(
        "--nr",
        type=int,
        default=64,
        help="Number of radial bins"
    )
    parser.add_argument(
        "--ftype",
        default=None,
        help="pNbody snapshot file type format override"
    )
    parser.add_argument(
        "--output",
        default="kinematic_profiles",
        help="Output graphics and archive prefix"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Pipeline hardcoded snapshot configuration mappings
    # --------------------------------------------------------
    ic = "snap/centered/snapshot_0000.hdf5"

    sims = [
        "snap/centered/snapshot_0010.hdf5",
        "snap/centered/snapshot_0020.hdf5",
        "snap/centered/snapshot_0030.hdf5",
        "snap/centered/snapshot_0040.hdf5",
        "snap/centered/snapshot_0050.hdf5",
        "snap/centered/snapshot_0060.hdf5",
        "snap/centered/snapshot_0070.hdf5",
        "snap/centered/snapshot_0080.hdf5",
        "snap/centered/snapshot_0090.hdf5",
        "snap/centered/snapshot_0100.hdf5"
    ]

    labels = [
        "10 Gyr", "20 Gyr", "30 Gyr", "40 Gyr", "50 Gyr",
        "60 Gyr", "70 Gyr", "80 Gyr", "90 Gyr", "100 Gyr"
    ]

    tasks = [ (ic, "Initial condition") ] + list(zip(sims, labels))

    all_results = {}

    for snapshot, label in tasks:
        print(f"Processing snapshot: {snapshot} ({label})")

        # Instantiating the pNbody framework tracking container
        nb = Nbody(snapshot, ftype=args.ftype)

        # Run velocity extraction profiles logic
        profile = compute_kinematic_profiles(
            nb,
            rmax=args.rmax,
            nr=args.nr
        )

        all_results[label] = profile

        # Binary archive array dumps structured using safe names
        safe_label = label.replace(" ", "_")
        savefile = f"{args.output}_{safe_label}"
        np.savez(
            savefile,
            radius=profile["radius"],
            sigma_1d=profile["sigma_1d"],
            sigma_1d_raw=profile["sigma_1d_raw"],
            sigma_r=profile["sigma_r"],
            sigma_t=profile["sigma_t"],
            beta=profile["beta"],
            beta_raw=profile["beta_raw"],
            counts=profile["counts"],
        )

    # Plot out the final aggregated two-panel canvas graphics file
    make_plots(all_results, args.output)


if __name__ == "__main__":
    main()
