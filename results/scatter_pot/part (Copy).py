#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pNbody import Nbody


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
    default=None
)

parser.add_argument("--s", type=float, default=1.0)
parser.add_argument("--s-selected", type=float, default=5.0)

# NEW: performance control
parser.add_argument("--stride", type=int, default=5, help="Downsampling factor")

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
# GLOBAL DOWN-SAMPLING (IMPORTANT OPTIMIZATION)
# --------------------------------------------------------

stride = args.stride
x_plot = x[::stride]
y_plot = y[::stride]
pot_plot = pot[::stride]




# ========================================================
# 3-PANEL PARTICLE MAP (FULL RESOLUTION, selection style)
# ========================================================

N = 3

fig, axes = plt.subplots(
    1, 3,
    figsize=(15, 5),
    sharex=False,
    sharey=False
)

axes = np.array(axes).flatten()

rmax_values = [5, 10, 50]

# --------------------------------------------------------
# SHELL SELECTION PARAMETERS
# --------------------------------------------------------

dr = 0.0 * args.select_radius  # shell thickness (adjust if needed)

r_inner = args.select_radius - dr
r_outer = args.select_radius + dr

Rproj = np.sqrt(x*x + y*y)

shell = (Rproj > r_inner) & (Rproj < r_outer)

for i in range(N):
    ax = axes[i]

    rmax_i = rmax_values[i]
    
    scale = args.rmax / rmax_i

    s_i = args.s * scale**0.7
    alpha_i = min(0.03 * scale**0.5, 0.2)

    # background particles
    ax.scatter(
        x,
        y,
        s=s_i,
        color="black",
        alpha=alpha_i,
        linewidths=0
    )

    # shell selection
    ax.scatter(
        x[shell],
        y[shell],
        s=args.s_selected * scale**0.5,
        color="red",
        linewidths=0,
        alpha=0.8
    )

    ax.set_xlim(-rmax_i, rmax_i)
    ax.set_ylim(-rmax_i, rmax_i)
    ax.set_aspect("equal")

    ax.set_title(rf"$R_{{\mathrm{{max}}}} = {rmax_i:.1f}\,\mathrm{{kpc}}$")

    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")

fig.suptitle("Particle Distribution at Different Scales", fontsize=14)

fig.tight_layout()
fig.subplots_adjust(bottom=0.125)

fig.savefig("particle_map_3panel.png", dpi=200)
print("Saved: particle_map_3panel.png")
