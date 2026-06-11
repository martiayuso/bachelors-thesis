#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt

from pNbody import Nbody

# ========================================================
# GLOBAL MATPLOTLIB STYLE
# ========================================================

plt.rcParams.update({

    # dark theme
    "figure.facecolor": "black",
    "axes.facecolor": "black",
    "savefig.facecolor": "black",
    "savefig.edgecolor": "black",

    # text colors
    "text.color": "white",
    "axes.labelcolor": "white",
    "axes.edgecolor": "white",
    "axes.titlecolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",

    # fonts
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "figure.titlesize": 22,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14

})

# ========================================================
# ARGUMENT PARSER
# ========================================================

parser = argparse.ArgumentParser(
    description="Artistic particle distribution plot"
)

parser.add_argument("snapshot", type=str)

parser.add_argument(
    "--output",
    default="particle_map_3panel.png"
)

parser.add_argument(
    "--rmax",
    type=float,
    default=50.0
)

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

parser.add_argument(
    "--s",
    type=float,
    default=1.0
)

parser.add_argument(
    "--s-selected",
    type=float,
    default=5.0
)

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
    potential_unit = nb.localsystem_of_units["phi"]
except:
    try:
        potential_unit = nb.localsystem_of_units["pot"]
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

    if args.ptype is not None:
        pos = nb.Pos(units="kpc")[mask]
    else:
        pos = nb.Pos(units="kpc")

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

keep_fraction = 1.0 / args.reduction_factor

rng = np.random.default_rng(42)

keep = rng.random(len(x)) < keep_fraction

x_plot = x[keep]
y_plot = y[keep]
pot_plot = pot[keep]
r_plot = r[keep]

selected_plot = selected[keep]

print("Reduction factor:", args.reduction_factor)
print("Keeping fraction:", keep_fraction)
print("Reduced particle count:", len(x_plot))

# ========================================================
# SHELL SELECTION
# ========================================================

dr = 0.0 * args.select_radius

r_inner = args.select_radius - dr
r_outer = args.select_radius + dr

Rproj = np.sqrt(x*x + y*y)

shell = (Rproj > r_inner) & (Rproj < r_outer)

# ========================================================
# COLOR MAPPING
# ========================================================

#
# OPTION A: color by radius
#
cvals = r_plot

#
# OPTION B: color by potential
#
# cvals = -pot_plot

norm = plt.Normalize(
    vmin=np.percentile(cvals, 1),
    vmax=np.percentile(cvals, 99)
)

cmap = plt.cm.magma

# ========================================================
# MULTI-SCALE PLOT
# ========================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(15, 15),
    facecolor="black"
)

axes = np.array(axes).flatten()

rmax_values = [5, 10, 20, 50]

for i, ax in enumerate(axes):

    rmax_i = rmax_values[i]

    scale = args.rmax / rmax_i

    s_i = args.s * scale**0.7

    glow_alpha = min(
        0.03 * scale**0.5,
        0.15
    )

    particle_alpha = min(
        0.25 * scale**0.3,
        0.55
    )

    ax.set_facecolor("black")

    # ----------------------------------------------------
    # glow layer
    # ----------------------------------------------------

    ax.scatter(
        x_plot,
        y_plot,
        c=cvals,
        cmap=cmap,
        norm=norm,
        s=s_i * 8,
        alpha=glow_alpha,
        linewidths=0,
        rasterized=True
    )

    # ----------------------------------------------------
    # main particle layer
    # ----------------------------------------------------

    sc = ax.scatter(
        x_plot,
        y_plot,
        c=cvals,
        cmap=cmap,
        norm=norm,
        s=s_i,
        alpha=particle_alpha,
        linewidths=0,
        rasterized=True
    )

    # ----------------------------------------------------
    # highlighted shell
    # ----------------------------------------------------

    ax.scatter(
        x[shell],
        y[shell],
        s=args.s_selected * scale**0.5,
        color="#00E5FF",
        alpha=0.95,
        linewidths=0
    )

    # ----------------------------------------------------
    # axis formatting
    # ----------------------------------------------------

    ax.set_xlim(-rmax_i, rmax_i)
    ax.set_ylim(-rmax_i, rmax_i)

    ax.set_aspect("equal")

    ax.set_title(
        f"Rmax = {rmax_i:.0f} kpc"
    )

    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")

    ax.tick_params(colors="white")

    for spine in ax.spines.values():
        spine.set_color("white")

# ========================================================
# COLORBAR
# ========================================================

cbar = fig.colorbar(
    sc,
    ax=axes.tolist(),
    shrink=0.8,
    pad=0.02
)

cbar.ax.tick_params(colors="white")
cbar.outline.set_edgecolor("white")

cbar.set_label(
    "Radius [kpc]",
    color="white"
)

# ========================================================
# TITLE
# ========================================================

fig.suptitle(
    "Dark Matter Halo Structure",
    fontsize=24,
    color="white"
)

fig.tight_layout()

# ========================================================
# SAVE
# ========================================================

fig.savefig(
    args.output,
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor()
)

print(f"Saved: {args.output}")
