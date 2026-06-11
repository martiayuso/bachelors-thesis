#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pNbody import Nbody

# ========================================================
# GLOBAL MATPLOTLIB STYLE
# ========================================================

plt.rcParams.update({

    # overall text
    "font.size": 16,

    # axis labels
    "axes.labelsize": 18,

    # subplot titles
    "axes.titlesize": 18,

    # figure title
    "figure.titlesize": 22,

    # tick labels
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,

    # legend
    "legend.fontsize": 14,

    # colorbar tick labels
    "figure.labelsize": 18

})


# ========================================================
# ARGUMENT PARSER
# ========================================================

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


parser.add_argument(
    "--reduction-factor",
    type=float,
    default=3.1,
    help="Global particle reduction factor"
)

args = parser.parse_args()


# ========================================================
# LOAD SNAPSHOT
# ========================================================

nb = Nbody(args.snapshot)


# ========================================================
# POTENTIAL UNITS
# ========================================================

try:
    potential_unit = nb.localsystem_of_units['phi']
except:
    try:
        potential_unit = nb.localsystem_of_units['pot']
    except:
        potential_unit = r"(km/s)$^2$"

print("Potential units:", potential_unit)


# ========================================================
# SELECT PARTICLE TYPES
# ========================================================

if args.ptype is not None:

    mask = np.isin(nb.tpe, args.ptype)

    pos = nb.pos[mask]
    pot = nb.pot[mask]

else:

    pos = nb.pos
    pot = nb.pot


# ========================================================
# CONVERT TO KPC
# ========================================================

try:
    pos = nb.Pos(units="kpc")[mask] if args.ptype is not None else nb.Pos(units="kpc")
except:
    pos = pos


# ========================================================
# CENTERING
# ========================================================

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


# ========================================================
# SELECTION
# ========================================================

selected = r < args.select_radius

print("Selected particles:", np.sum(selected))


# ========================================================
# GLOBAL PARTICLE REDUCTION
# ========================================================

reduction = args.reduction_factor

# fraction of particles to KEEP
keep_fraction = 1.0 / reduction

Ntot = len(x)

rng = np.random.default_rng(42)

keep = rng.random(Ntot) < keep_fraction

x_plot = x[keep]
y_plot = y[keep]
pot_plot = pot[keep]
r_plot = r[keep]

selected_plot = selected[keep]

print("Reduction factor:", reduction)
print("Keeping fraction:", keep_fraction)
print("Reduced particle count:", len(x_plot))



# ========================================================
# 3-PANEL PARTICLE MAP (FULL RESOLUTION, selection style)
# ========================================================

N = 4

fig, axes = plt.subplots(
    2, 2,
    figsize=(15, 15),
    sharex=False,
    sharey=False
)

axes = np.array(axes).flatten()

rmax_values = [5, 10, 20, 50]

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
        x_plot,
        y_plot,
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

fig.suptitle("Particle Distribution at Different Scales", fontsize=20)

fig.tight_layout()
#fig.subplots_adjust(bottom=0.125)

fig.savefig("particle_map_3panel.png", dpi=200)
print("Saved: particle_map_3panel.png")
