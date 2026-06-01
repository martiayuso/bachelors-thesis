#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pNbody import Nbody

'''
python3 scatter_potential.py snapshot_0100.hdf5 \
    --ptype 1 \
    --center potential \
    --select-radius 2 \
    --rmax 1 \
    --output fig.png
'''


# --------------------------------------------------------
# Argument parser
# --------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Potential + selection plot from SWIFT snapshots using pNbody"
)

parser.add_argument("snapshot", type=str)

parser.add_argument("--output", default="potential_selection.png")

parser.add_argument("--rmax", type=float, default=50.0)

parser.add_argument(
    "--center",
    choices=["potential", "com", "origin"],
    default="potential"
)

parser.add_argument(
    "--select-radius",
    type=float,
    default=2.0,
    help="Selection radius in kpc"
)

parser.add_argument(
    "--ptype",
    type=int,
    nargs="+",
    default=None,
    help="Particle types to keep"
)

parser.add_argument("--s", type=float, default=1.0)
parser.add_argument("--s-selected", type=float, default=5.0)

args = parser.parse_args()


# --------------------------------------------------------
# Load snapshot
# --------------------------------------------------------

nb = Nbody(args.snapshot)

# --------------------------------------------------------
# Potential units
# --------------------------------------------------------

try:
    potential_unit = nb.localsystem_of_units['phi']
except:
    try:
        potential_unit = nb.localsystem_of_units['pot']
    except:
        potential_unit = r"(km/s)$^2$"

print("Potential units:", potential_unit)

# --------------------------------------------------------
# Select particle types
# --------------------------------------------------------

if args.ptype is not None:

    mask = np.isin(nb.tpe, args.ptype)

    pos = nb.pos[mask]
    pot = nb.pot[mask]

else:

    pos = nb.pos
    pot = nb.pot


# --------------------------------------------------------
# Convert to kpc
# --------------------------------------------------------

try:
    pos = nb.Pos(units="kpc")[mask] if args.ptype is not None else nb.Pos(units="kpc")
except:
    pos = pos


# --------------------------------------------------------
# Centering
# --------------------------------------------------------

if args.center == "potential":

    idx = np.argmin(pot)
    center = pos[idx]

elif args.center == "com":

    center = np.mean(pos, axis=0)

else:

    center = np.zeros(3)


pos = pos - center

x = pos[:, 0]
y = pos[:, 1]
z = pos[:, 2]

r = np.sqrt(x*x + y*y + z*z)

print("Particles:", len(pos))
print("Potential min/max:", np.min(pot), np.max(pot))
print("Center:", center)


# --------------------------------------------------------
# Selection
# --------------------------------------------------------

selected = r < args.select_radius

print("Selected particles:", np.sum(selected))


# --------------------------------------------------------
# Plot
# --------------------------------------------------------

# --------------------------------------------------------
# POTENTIAL MAP
# --------------------------------------------------------

fig1, ax1 = plt.subplots(figsize=(6, 6))

sc = ax1.scatter(
    x,
    y,
    c=pot,
    s=args.s,
    cmap="inferno_r",
    alpha=0.8,
    linewidths=0,
    rasterized=True
)

ax1.set_xlim(-args.rmax, args.rmax)
ax1.set_ylim(-args.rmax, args.rmax)

ax1.set_aspect("equal")

ax1.set_xlabel("x [kpc]")
ax1.set_ylabel("y [kpc]")

ax1.set_title("Potential Map")

divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)

cbar = fig1.colorbar(sc, cax=cax)
cbar.set_label(rf"Potential $\Phi$ [{potential_unit}]")

fig1.tight_layout()

potential_name = "potential_map.png"

fig1.savefig(potential_name, dpi=300)

print(f"Saved: {potential_name}")


# --------------------------------------------------------
# SELECTION MAP
# --------------------------------------------------------

fig2, ax2 = plt.subplots(figsize=(6, 6))

# background particles
ax2.scatter(
    x,
    y,
    s=args.s,
    color="black",
    alpha=0.03,
    linewidths=0,
    rasterized=True
)

# selected particles
ax2.scatter(
    x[selected],
    y[selected],
    s=args.s_selected,
    color="red",
    linewidths=0,
    rasterized=True
)

ax2.set_xlim(-args.rmax, args.rmax)
ax2.set_ylim(-args.rmax, args.rmax)

ax2.set_aspect("equal")

ax2.set_xlabel("x [kpc]")
ax2.set_ylabel("y [kpc]")

ax2.set_title("Selected Particles")

fig2.tight_layout()

selection_name = "selection_map.png"

fig2.savefig(selection_name, dpi=300)

print(f"Saved: {selection_name}")


# --------------------------------------------------------
# MULTI-PANEL POTENTIAL MAP (N plots, varying rmax)
# --------------------------------------------------------

N = 9  # number of panels

nrows = int(np.sqrt(N))
ncols = int(np.ceil(N / nrows))

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(12, 12),
    sharex=False,
    sharey=False
)

axes = axes.flatten()

# --------------------------------------------------------
# Different rmax values per panel (log-spaced = nicer zoom)
# --------------------------------------------------------

rmax_values = np.geomspace(1, args.rmax, N)

# --------------------------------------------------------
# Shared color scale across all panels
# --------------------------------------------------------

vmin = np.min(pot)
vmax = np.max(pot)

# --------------------------------------------------------
# Plot each panel
# --------------------------------------------------------

for i in range(N):
    ax = axes[i]

    rmax_i = rmax_values[i]

    sc = ax.scatter(
        x,
        y,
        c=pot,
        s=args.s,
        cmap="inferno_r",
        alpha=0.8,
        linewidths=0,
        rasterized=True,
        vmin=vmin,
        vmax=vmax
    )

    ax.set_xlim(-rmax_i, rmax_i)
    ax.set_ylim(-rmax_i, rmax_i)

    ax.set_aspect("equal")

    ax.set_title(f"rmax = {rmax_i:.2f} kpc", fontsize=10)

    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")

# --------------------------------------------------------
# Hide unused axes (safety)
# --------------------------------------------------------

for j in range(N, len(axes)):
    axes[j].axis("off")

# --------------------------------------------------------
# Global labels
# --------------------------------------------------------

fig.suptitle("Potential Maps at Different Scales", fontsize=14)

# --------------------------------------------------------
# SINGLE SHARED COLORBAR
# --------------------------------------------------------

cbar = fig.colorbar(
    sc,
    ax=axes[:N],
    location="right",
    shrink=0.9,
    pad=0.02
)

cbar.set_label(rf"Potential $\Phi$ [{potential_unit}]")

fig.tight_layout()

# --------------------------------------------------------
# Save
# --------------------------------------------------------

multi_name = "potential_map_multi_rmax.png"
fig.savefig(multi_name, dpi=300)

print(f"Saved: {multi_name}")
