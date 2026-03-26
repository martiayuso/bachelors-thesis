#!/usr/bin/env python3
"""
Compute and plot enclosed (cumulative) mass profile M(<R) from SWIFT snapshots.

Usage examples:
python3 plotMass.py ../snapshot_0000.hdf5 ../snapshot_0100.hdf5 --rmin 0.01 --rmax 10 --nbins 100 --log xy --types dm --center-mode shrinking -o mass_plot.png


Notes:
- By default the script assumes the snapshot positions and masses are already in
  the units you want to plot. If not, use --pos-scale and --mass-scale.
- For cosmological/periodic boxes, --periodic applies a minimum-image wrapping
  around the chosen center.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
import matplotlib.pyplot as plt


PARTTYPE_MAP = {
    "gas": 0,
    "dm": 1,
    "dark_matter": 1,
    "tracers": 3,
    "stars": 4,
    "star": 4,
    "bh": 5,
    "black_holes": 5,
    "black_hole": 5,
}


@dataclass
class ParticleSet:
    name: str
    positions: np.ndarray
    masses: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--save-data", default=None)
    parser.add_argument("--rmax", type=float, default=50.0)
    parser.add_argument("--rmin", type=float, default=0.0)
    parser.add_argument("--nbins", type=int, default=100)
    parser.add_argument("--logbins", action="store_true")
    parser.add_argument("--log", choices=["none", "x", "y", "xy"], default="none")
    parser.add_argument("--types", nargs="+", default=["gas", "stars", "dm"])
    parser.add_argument("--center-mode", choices=["com", "median", "manual", "shrinking"], default="com")
    parser.add_argument("--center", nargs=3, type=float, default=None)
    parser.add_argument("--periodic", action="store_true")
    parser.add_argument("--pos-scale", type=float, default=1.0)
    parser.add_argument("--mass-scale", type=float, default=1.0)
    parser.add_argument("--xlabel", default=r"$R$")
    parser.add_argument("--ylabel", default=r"$M(<R)$")
    parser.add_argument("--title", default="")
    parser.add_argument("--legend-loc", default="best")
    parser.add_argument("--linewidth", type=float, default=2.0)
    return parser.parse_args()


def get_boxsize(handle):
    try:
        return float(np.asarray(handle["Header"].attrs["BoxSize"]).reshape(-1)[0])
    except:
        return None


def get_mass_table(handle):
    try:
        return np.asarray(handle["Header"].attrs["MassTable"], dtype=float)
    except:
        return np.zeros(6)


def load_particle_set(handle, type_name, pos_scale, mass_scale):
    idx = PARTTYPE_MAP[type_name]
    group_name = f"PartType{idx}"
    if group_name not in handle:
        return None

    g = handle[group_name]
    if "Coordinates" not in g:
        return None

    pos = np.asarray(g["Coordinates"]) * pos_scale

    if "Masses" in g:
        mass = np.asarray(g["Masses"]) * mass_scale
    else:
        mt = get_mass_table(handle)
        mass = np.full(len(pos), mt[idx] * mass_scale)

    return ParticleSet(type_name, pos, mass)


def load_selected_particles(snapshot, types, pos_scale, mass_scale):
    particles = []
    with h5py.File(snapshot, "r") as f:
        boxsize = get_boxsize(f)
        for t in types:
            ps = load_particle_set(f, t, pos_scale, mass_scale)
            if ps is not None:
                particles.append(ps)
    return particles, boxsize


def combine_particles(particles):
    pos = np.vstack([p.positions for p in particles])
    mass = np.concatenate([p.masses for p in particles])
    return pos, mass


def shrinking_sphere_center(pos, mass):
    center = np.average(pos, axis=0, weights=mass)
    radius = np.max(np.linalg.norm(pos - center, axis=1))

    while True:
        r = np.linalg.norm(pos - center, axis=1)
        mask = r < radius
        if np.sum(mask) < 100:
            break

        new_center = np.average(pos[mask], axis=0, weights=mass[mask])
        if np.linalg.norm(new_center - center) < 1e-5:
            break

        center = new_center
        radius *= 0.9

    return center


def compute_center(pos, mass, mode, manual):
    if mode == "manual":
        return np.asarray(manual)
    if mode == "median":
        return np.median(pos, axis=0)
    if mode == "shrinking":
        return shrinking_sphere_center(pos, mass)
    return np.average(pos, axis=0, weights=mass)


def radial_profile(pos, mass, center, rmin, rmax, nbins, logbins):
    r = np.linalg.norm(pos - center, axis=1)

    if logbins:
        edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    else:
        edges = np.linspace(rmin, rmax, nbins + 1)

    shell_mass, _ = np.histogram(r, bins=edges, weights=mass)
    enclosed = np.cumsum(shell_mass)

    return edges[1:], enclosed


def apply_log(ax, mode):
    if mode in ("x", "xy"):
        ax.set_xscale("log")
    if mode in ("y", "xy"):
        ax.set_yscale("log")


def main():
    args = parse_args()

    multi = len(args.files) > 1

    if multi:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, sharex=True,
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}
        )
    else:
        fig, ax1 = plt.subplots()
        ax2 = None

    ref_r = None
    ref_M = None

    for i, snap in enumerate(args.files):
        particles, _ = load_selected_particles(
            snap, args.types, args.pos_scale, args.mass_scale
        )

        pos, mass = combine_particles(particles)
        center = compute_center(pos, mass, args.center_mode, args.center)
        print(f"{snap}: center = {center}")

        r, M = radial_profile(pos, mass, center,
                              args.rmin, args.rmax,
                              args.nbins, args.logbins)

        label = os.path.basename(snap)
        ax1.plot(r, M, label=label)

        # store reference
        if multi and i == 0:
            ref_r = r
            ref_M = M
            ax2.axhline(1.0, color="gray", linestyle="--")

        elif multi:
            M_interp = np.interp(ref_r, r, M)
            ratio = M_interp / ref_M
            ax2.plot(ref_r, ratio)

    apply_log(ax1, args.log)

    ax1.set_ylabel(args.ylabel)
    ax1.legend()

    if multi:
        ax2.set_xlabel(args.xlabel)
        ax2.set_ylabel(r"$M / M_{\rm init}$")
        ax2.set_ylim(0.8, 1.2)
        ax2.grid(alpha=0.3)

    else:
        ax1.set_xlabel(args.xlabel)

    if args.title:
        ax1.set_title(args.title)

    ax1.grid(alpha=0.3)

    fig.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()
