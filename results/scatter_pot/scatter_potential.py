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
# POTENTIAL MAP (single)
# ========================================================

fig1, ax1 = plt.subplots(figsize=(6, 6))

sc = ax1.scatter(
    x_plot,
    y_plot,
    c=pot_plot,
    s=args.s,
    cmap="inferno_r",
    alpha=0.8,
    linewidths=0,
    edgecolors="none"
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

fig1.savefig("potential_map.png", dpi=200)
print("Saved: potential_map.png")


# ========================================================
# SELECTION MAP (single)
# ========================================================

fig2, ax2 = plt.subplots(figsize=(6, 6))

ax2.scatter(
    x_plot,
    y_plot,
    s=args.s,
    color="black",
    alpha=0.03,
    linewidths=0
)

ax2.scatter(
    x_plot[selected[::stride]],
    y_plot[selected[::stride]],
    s=args.s_selected,
    color="red",
    linewidths=0
)

ax2.set_xlim(-args.rmax, args.rmax)
ax2.set_ylim(-args.rmax, args.rmax)
ax2.set_aspect("equal")

ax2.set_xlabel("x [kpc]")
ax2.set_ylabel("y [kpc]")
ax2.set_title("Selected Particles")

fig2.tight_layout()
fig2.savefig("selection_map.png", dpi=200)

print("Saved: selection_map.png")


# ========================================================
# MULTI-PANEL POTENTIAL MAP (9 panels, optimized)
# ========================================================

N = 9
nrows = 3
ncols = 3

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(10, 10),
    sharex=False,
    sharey=False
)

axes = axes.flatten()

rmax_values = np.geomspace(1, args.rmax, N)

vmin = np.min(pot_plot)
vmax = np.max(pot_plot)

for i in range(N):
    ax = axes[i]

    sc = ax.scatter(
        x_plot,
        y_plot,
        c=pot_plot,
        s=args.s,
        cmap="inferno_r",
        alpha=0.8,
        linewidths=0,
        edgecolors="none",
        vmin=vmin,
        vmax=vmax
    )

    rmax_i = rmax_values[i]

    ax.set_xlim(-rmax_i, rmax_i)
    ax.set_ylim(-rmax_i, rmax_i)
    ax.set_aspect("equal")

    ax.set_title(rf"$R_{{\mathrm{{max}}}} = {rmax_i:.1f}\,\mathrm{{kpc}}$")

    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")

for j in range(N, len(axes)):
    axes[j].axis("off")

fig.suptitle("Potential Maps at Different Scales", fontsize=14)

# --------------------------------------------------------
# SINGLE SHARED COLORBAR (FIXED LAYOUT)
# --------------------------------------------------------
'''
# leave space on the right
fig.subplots_adjust(right=0.88)

# create dedicated colorbar axis
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])

cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.set_label(rf"Potential $\Phi$ [{potential_unit}]")
'''

fig.tight_layout()

fig.savefig("potential_map_multi_rmax.png", dpi=200)
print("Saved: potential_map_multi_rmax.png")


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

rmax_values = np.linspace(2, args.rmax, N)

# --------------------------------------------------------
# SHELL SELECTION PARAMETERS
# --------------------------------------------------------

dr = 0.2 * args.select_radius  # shell thickness (adjust if needed)

r_inner = args.select_radius - dr
r_outer = args.select_radius + dr

Rproj = np.sqrt(x*x + y*y)

shell = (Rproj > r_inner) & (Rproj < r_outer)

for i in range(N):
    ax = axes[i]

    rmax_i = rmax_values[i]

    # background particles
    ax.scatter(
        x,
        y,
        s=args.s,
        color="black",
        alpha=0.03,
        linewidths=0
    )

    # shell selection (red)
    ax.scatter(
        x[shell],
        y[shell],
        s=args.s_selected,
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

fig.savefig("particle_map_3panel.png", dpi=200)
print("Saved: particle_map_3panel.png")
