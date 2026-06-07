#!/usr/bin/env python3
"""
Visualize the halo mass distribution and identify a mass transition radius.

This script is designed to help detect a sharp change in particle mass
(e.g. at RCUT where particles outside have different masses).

Features
--------
1. Scatter plot: particle radius vs particle mass
2. Radial median mass profile
3. Particle count histogram
4. Optional vertical RCUT line
5. Supports SWIFT snapshots

Example
-------
python3 massDiag.py snapshot_0000.hdf5 --type dm --rcut 5.0 --center-mode shrinking

"""

from __future__ import annotations

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt


PARTTYPE_MAP = {
    "gas": 0,
    "dm": 1,
    "stars": 4,
    "bh": 5,
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("snapshot")

    parser.add_argument("--type", default="dm")
    parser.add_argument("--rcut", type=float, default=None)

    parser.add_argument(
        "--center-mode",
        choices=["com", "median", "shrinking"],
        default="shrinking"
    )

    parser.add_argument("--sample", type=int, default=200000)
    parser.add_argument("--bins", type=int, default=100)

    parser.add_argument("--rmax", type=float, default=None)

    return parser.parse_args()


def load_particles(snapshot, ptype):
    idx = PARTTYPE_MAP[ptype]

    with h5py.File(snapshot, "r") as f:
        g = f[f"PartType{idx}"]

        pos = np.asarray(g["Coordinates"])

        if "Masses" in g:
            mass = np.asarray(g["Masses"])
        else:
            mt = f["Header"].attrs["MassTable"]
            mass = np.full(len(pos), mt[idx])

    return pos, mass


def shrinking_sphere_center(pos, mass):
    center = np.average(pos, axis=0, weights=mass)

    radius = np.max(np.linalg.norm(pos - center, axis=1))

    while True:

        r = np.linalg.norm(pos - center, axis=1)

        mask = r < radius

        if np.sum(mask) < 100:
            break

        new_center = np.average(
            pos[mask],
            axis=0,
            weights=mass[mask]
        )

        shift = np.linalg.norm(new_center - center)

        center = new_center

        radius *= 0.9

        if shift < 1e-5:
            break

    return center


def compute_center(pos, mass, mode):

    if mode == "median":
        return np.median(pos, axis=0)

    if mode == "shrinking":
        return shrinking_sphere_center(pos, mass)

    return np.average(pos, axis=0, weights=mass)


def radial_mass_profile(r, mass, bins):

    edges = np.linspace(r.min(), r.max(), bins + 1)

    centers = 0.5 * (edges[:-1] + edges[1:])

    median_mass = np.zeros(bins)
    mean_mass = np.zeros(bins)
    count = np.zeros(bins)

    for i in range(bins):

        mask = (r >= edges[i]) & (r < edges[i + 1])

        if np.any(mask):
            median_mass[i] = np.median(mass[mask])
            mean_mass[i] = np.mean(mass[mask])
            count[i] = np.sum(mask)
        else:
            median_mass[i] = np.nan
            mean_mass[i] = np.nan

    return centers, median_mass, mean_mass, count


def main():

    args = parse_args()

    pos, mass = load_particles(args.snapshot, args.type)

    print(f"Loaded {len(pos):,} particles")

    center = compute_center(pos, mass, args.center_mode)

    print("Center =", center)

    r = np.linalg.norm(pos - center, axis=1)

    if args.rmax is not None:
        mask = r <= args.rmax

        r = r[mask]
        mass = mass[mask]

    #
    # RANDOM SUBSAMPLING
    #
    if len(r) > args.sample:

        idx = np.random.choice(
            len(r),
            size=args.sample,
            replace=False
        )

        r_plot = r[idx]
        mass_plot = mass[idx]

    else:
        r_plot = r
        mass_plot = mass

    #
    # RADIAL MASS PROFILE
    #
    rc, median_mass, mean_mass, count = radial_mass_profile(
        r,
        mass,
        args.bins
    )

    #
    # FIGURE
    #
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 12),
        sharex=True
    )

    #
    # 1. SCATTER
    #
    ax = axes[0]

    ax.scatter(
        r_plot,
        mass_plot,
        s=1,
        alpha=0.3
    )

    ax.set_yscale("log")

    ax.set_ylabel("Particle Mass")

    ax.set_title("Radius vs Particle Mass")

    #
    # 2. MEDIAN PROFILE
    #
    ax = axes[1]

    ax.plot(
        rc,
        median_mass,
        label="Median mass"
    )

    ax.plot(
        rc,
        mean_mass,
        label="Mean mass",
        linestyle="--"
    )

    ax.set_yscale("log")

    ax.set_ylabel("Mass")

    ax.legend()

    ax.set_title("Radial Mass Profile")

    #
    # 3. COUNTS
    #
    ax = axes[2]

    ax.plot(rc, count)

    ax.set_ylabel("Particle Count")

    ax.set_xlabel("Radius")

    ax.set_title("Particles per Radial Bin")

    #
    # RCUT LINE
    #
    if args.rcut is not None:

        for ax in axes:

            ax.axvline(
                args.rcut,
                color="red",
                linestyle="--",
                label="RCUT"
            )

    #
    # CLEANUP
    #
    for ax in axes:
        ax.grid(alpha=0.3)

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
