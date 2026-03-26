#!/usr/bin/env python3
"""
Compute and plot enclosed (cumulative) mass profile M(<R) from SWIFT snapshots.

Usage examples:
python3 plotMass.py snapshot_0000.hdf5 --rmax 50 --nbins 100 --log xy --types dm --center-mode shrinking --save-data

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot enclosed mass profile M(<R) from SWIFT snapshots."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="One or more SWIFT HDF5 snapshot files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output figure filename. If omitted, the plot is shown interactively.",
    )
    parser.add_argument(
        "--save-data",
        default=None,
        help="Optional .npz file to save the last computed profile data.",
    )
    parser.add_argument(
        "--rmax",
        type=float,
        default=50.0,
        help="Maximum radius to plot.",
    )
    parser.add_argument(
        "--rmin",
        type=float,
        default=0.0,
        help="Minimum radius for the bin edges. Use a small positive value for log x-axis.",
    )
    parser.add_argument(
        "--nbins",
        type=int,
        default=100,
        help="Number of radial bins.",
    )
    parser.add_argument(
        "--logbins",
        action="store_true",
        help="Use logarithmic radial bins instead of linear bins.",
    )
    parser.add_argument(
        "--log",
        choices=["none", "x", "y", "xy"],
        default="none",
        help="Logarithmic axes.",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=["gas", "stars", "dm"],
        help="Particle types to include: gas stars dm bh ...",
    )
    parser.add_argument(
        "--center-mode",
        choices=["com", "median", "manual", "shrinking"],
        default="com",
        help="How to define the galaxy center.",
    )
    parser.add_argument(
        "--center",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Manual center coordinates. Required if --center-mode manual.",
    )
    parser.add_argument(
        "--periodic",
        action="store_true",
        help="Apply minimum-image wrapping around the center using the box size from the snapshot header.",
    )
    parser.add_argument(
        "--pos-scale",
        type=float,
        default=1.0,
        help="Multiply raw snapshot coordinates by this factor before computing radii.",
    )
    parser.add_argument(
        "--mass-scale",
        type=float,
        default=1.0,
        help="Multiply raw snapshot masses by this factor before computing enclosed mass.",
    )
    parser.add_argument(
        "--xlabel",
        default=r"$R$",
        help="X-axis label.",
    )
    parser.add_argument(
        "--ylabel",
        default=r"$M(<R)$",
        help="Y-axis label.",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Plot title.",
    )
    parser.add_argument(
        "--legend-loc",
        default="best",
        help="Legend location.",
    )
    parser.add_argument(
        "--linewidth",
        type=float,
        default=2.0,
        help="Line width.",
    )
    parser.add_argument(
        "--show-total",
        action="store_true",
        help="If multiple particle types are selected, also draw the total curve in black.",
    )
    return parser.parse_args()


def get_boxsize(handle: h5py.File) -> Optional[float]:
    try:
        boxsize = handle["Header"].attrs["BoxSize"]
        boxsize = np.asarray(boxsize).reshape(-1)
        if boxsize.size == 1:
            return float(boxsize[0])
        return float(boxsize[0])
    except Exception:
        return None


def get_mass_table(handle: h5py.File) -> np.ndarray:
    try:
        table = np.asarray(handle["Header"].attrs["MassTable"], dtype=float)
        return table
    except Exception:
        return np.zeros(6, dtype=float)


def particle_group_name(type_name: str) -> str:
    if type_name not in PARTTYPE_MAP:
        raise ValueError(f"Unknown particle type '{type_name}'.")
    return f"PartType{PARTTYPE_MAP[type_name]}"


def load_particle_set(
    handle: h5py.File,
    type_name: str,
    pos_scale: float,
    mass_scale: float,
) -> Optional[ParticleSet]:
    group_name = particle_group_name(type_name)
    if group_name not in handle:
        return None

    group = handle[group_name]
    if "Coordinates" not in group:
        return None

    pos = np.asarray(group["Coordinates"], dtype=float) * pos_scale

    if "Masses" in group:
        mass = np.asarray(group["Masses"], dtype=float) * mass_scale
    else:
        mass_table = get_mass_table(handle)
        idx = PARTTYPE_MAP[type_name]
        if idx >= len(mass_table) or mass_table[idx] <= 0:
            raise RuntimeError(
                f"{group_name} has no Masses dataset and no constant mass in Header/MassTable."
            )
        mass = np.full(pos.shape[0], mass_table[idx] * mass_scale, dtype=float)

    return ParticleSet(name=type_name, positions=pos, masses=mass)


def load_selected_particles(
    snapshot: str,
    types: List[str],
    pos_scale: float,
    mass_scale: float,
) -> Tuple[List[ParticleSet], Optional[float]]:
    particles: List[ParticleSet] = []
    with h5py.File(snapshot, "r") as handle:
        boxsize = get_boxsize(handle)
        for t in types:
            ps = load_particle_set(handle, t, pos_scale=pos_scale, mass_scale=mass_scale)
            if ps is not None and len(ps.positions) > 0:
                particles.append(ps)
    return particles, boxsize


def combine_particles(particles: List[ParticleSet]) -> Tuple[np.ndarray, np.ndarray]:
    positions = np.vstack([p.positions for p in particles])
    masses = np.concatenate([p.masses for p in particles])
    return positions, masses


def compute_center(
    positions: np.ndarray,
    masses: np.ndarray,
    mode: str,
    manual_center: Optional[np.ndarray],
) -> np.ndarray:
    if mode == "manual":
        if manual_center is None:
            raise ValueError("--center-mode manual requires --center X Y Z.")
        return np.asarray(manual_center, dtype=float)

    if mode == "median":
        return np.median(positions, axis=0)

    if mode == "shrinking":
        return shrinking_sphere_center(positions, masses)

    total_mass = np.sum(masses)
    if not np.isfinite(total_mass) or total_mass <= 0:
        return np.mean(positions, axis=0)
    return np.average(positions, axis=0, weights=masses)

def shrinking_sphere_center(pos, mass, shrink_factor=0.9, min_particles=100):
    center = np.average(pos, axis=0, weights=mass)
    radius = np.max(np.linalg.norm(pos - center, axis=1))

    while True:
        r = np.linalg.norm(pos - center, axis=1)
        mask = r < radius

        if np.sum(mask) < min_particles:
            break

        new_center = np.average(pos[mask], axis=0, weights=mass[mask])

        if np.linalg.norm(new_center - center) < 1e-5:
            break

        center = new_center
        radius *= shrink_factor

    return center

def minimum_image(delta: np.ndarray, boxsize: Optional[float]) -> np.ndarray:
    if boxsize is None:
        return delta
    return delta - boxsize * np.rint(delta / boxsize)


def radial_profile_enclosed_mass(
    positions: np.ndarray,
    masses: np.ndarray,
    center: np.ndarray,
    rmin: float,
    rmax: float,
    nbins: int,
    logbins: bool = False,
    boxsize: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = positions - center[None, :]
    delta = minimum_image(delta, boxsize)
    radii = np.linalg.norm(delta, axis=1)

    if logbins:
        if rmin <= 0:
            raise ValueError("For logarithmic bins, set --rmin to a positive value.")
        edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    else:
        edges = np.linspace(rmin, rmax, nbins + 1)

    shell_mass, _ = np.histogram(radii, bins=edges, weights=masses)
    enclosed_mass = np.cumsum(shell_mass)
    r_plot = edges[1:]
    return r_plot, enclosed_mass, edges


def profile_by_type(
    particles: List[ParticleSet],
    center: np.ndarray,
    rmin: float,
    rmax: float,
    nbins: int,
    logbins: bool,
    boxsize: Optional[float],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    all_positions, all_masses = combine_particles(particles)
    r_plot, total, edges = radial_profile_enclosed_mass(
        all_positions, all_masses, center, rmin, rmax, nbins, logbins=logbins, boxsize=boxsize
    )

    profiles: Dict[str, np.ndarray] = {}
    for p in particles:
        _, enc, _ = radial_profile_enclosed_mass(
            p.positions, p.masses, center, rmin, rmax, nbins, logbins=logbins, boxsize=boxsize
        )
        profiles[p.name] = enc

    return r_plot, profiles, total


def apply_log_scale(ax: plt.Axes, mode: str) -> None:
    if mode in ("x", "xy"):
        ax.set_xscale("log")
    if mode in ("y", "xy"):
        ax.set_yscale("log")


def main() -> None:
    args = parse_args()

    plt.rcParams.update({
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "font.size": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "lines.linewidth": args.linewidth,
        "figure.figsize": (8, 6),
    })

    fig, ax = plt.subplots()

    last_radius = None
    last_total = None
    last_profiles = None

    for snapshot in args.files:
        particles, boxsize = load_selected_particles(
            snapshot,
            types=args.types,
            pos_scale=args.pos_scale,
            mass_scale=args.mass_scale,
        )

        if len(particles) == 0:
            print(f"[WARN] No matching particle types found in {snapshot}. Skipping.")
            continue

        if args.center_mode == "manual":
            center = compute_center(
                *combine_particles(particles),
                mode="manual",
                manual_center=np.asarray(args.center, dtype=float) if args.center is not None else None,
            )
        else:
            pos_all, mass_all = combine_particles(particles)
            center = compute_center(pos_all, mass_all, mode=args.center_mode, manual_center=None)

        r_plot, profiles, total = profile_by_type(
            particles,
            center=center,
            rmin=args.rmin,
            rmax=args.rmax,
            nbins=args.nbins,
            logbins=args.logbins,
            boxsize=boxsize if args.periodic else None,
        )

        label = os.path.basename(snapshot)
        ax.plot(r_plot, total, label=f"{label} (total)")

        if len(args.types) > 1 and args.show_total:
            ax.plot(r_plot, total, color="k", linewidth=args.linewidth + 0.5)

        if len(args.types) > 1:
            for t in args.types:
                if t in profiles:
                    ax.plot(r_plot, profiles[t], linestyle="--", alpha=0.8, label=f"{label} ({t})")

        last_radius = r_plot
        last_total = total
        last_profiles = profiles

        total_mass = total[-1] if len(total) else 0.0
        print(f"{snapshot}: center = {center}, M(<{args.rmax}) = {total_mass:.6e}")

    apply_log_scale(ax, args.log)

    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    if args.title:
        ax.set_title(args.title)
    ax.legend(loc=args.legend_loc)

    fig.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {args.output}")
    else:
        plt.show()

    if args.save_data and last_radius is not None and last_total is not None:
        save_dict = {
            "radius": last_radius,
            "total": last_total,
        }
        if last_profiles is not None:
            for key, value in last_profiles.items():
                save_dict[key] = value
        np.savez(args.save_data, **save_dict)
        print(f"Saved profile data to {args.save_data}")


if __name__ == "__main__":
    main()
