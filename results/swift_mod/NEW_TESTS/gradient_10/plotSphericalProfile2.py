#!/usr/bin/python3
###########################################################################################
#  package:   pNbody
#  file:      plotSphericalProfile
###########################################################################################

import os
import numpy as np
import argparse

import matplotlib.pyplot as plt
from pNbody import plot
from pNbody import *
from pNbody import cosmo
from pNbody import libutil

from astropy import units as u
from astropy import constants as c

import copy

####################################################################
# shrinking-sphere center
####################################################################

def shrinking_sphere_center(pos, mass, rfrac=0.9, min_n=100, tol=1e-5, max_iter=50):

    center = np.average(pos, axis=0, weights=mass)
    radius = np.max(np.linalg.norm(pos - center, axis=1))

    for _ in range(max_iter):

        r = np.linalg.norm(pos - center, axis=1)
        mask = r < radius

        if np.sum(mask) < min_n:
            break

        new_center = np.average(pos[mask], axis=0, weights=mass[mask])

        if np.linalg.norm(new_center - center) < tol:
            center = new_center
            break

        center = new_center
        radius *= rfrac

    return center


####################################################################
# option parser
####################################################################

description="""plot different physical quantities as a function of the 3-d radius"""

epilog="""Examples:
python3 plotSphericalProfile2.py -y density snapshot.hdf5
"""

parser = argparse.ArgumentParser(description=description,
                                 epilog=epilog,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

plot.add_files_options(parser)
plot.add_arguments_units(parser)
plot.add_arguments_reduc(parser)
plot.add_arguments_center(parser)
plot.add_arguments_select(parser)
plot.add_arguments_info(parser)
plot.add_arguments_legend(parser)
plot.add_arguments_icshift(parser)
plot.add_arguments_cmd(parser)

parser.add_argument("files", nargs="*", type=str, default=None)

parser.add_argument("-o", "--outputfilename", type=str, default=None)

parser.add_argument("--xmin", type=float, default=None)
parser.add_argument("--xmax", type=float, default=None)
parser.add_argument("--ymin", type=float, default=None)
parser.add_argument("--ymax", type=float, default=None)

parser.add_argument("--log", type=str, default=None)

parser.add_argument("-y", "--y", type=str, default="density")

parser.add_argument("--rmax", type=float, default=50.)
parser.add_argument("--nr", type=int, default=32)
parser.add_argument("--nlos", type=int, default=1)
parser.add_argument("--eps", type=float, default=0.1)

parser.add_argument("--colormap", type=str, default="jet")

# NEW: center mode
parser.add_argument("--center-mode",
                    type=str,
                    default="com",
                    choices=["com", "median", "manual", "shrinking"],
                    help="halo centering method")


####################################################################
# MakePlot
####################################################################

def MakePlot(opt):

    params = {
        "figure.figsize": (8, 6),
        "text.usetex": True,
    }
    plt.rcParams.update(params)

    fig = plt.gcf()
    ax = plt.gca()

    colors = plot.ColorList(n=len(opt.files), colormap=opt.colormap)
    datas = []

    for filename in opt.files:

        nb = Nbody(filename, ftype=opt.ftype)

        # -----------------------------
        # units
        # -----------------------------
        unit_params = plot.apply_arguments_units(opt)
        nb.set_local_system_of_units(params=unit_params)

        # -----------------------------
        # preprocessing
        # -----------------------------
        nb = plot.apply_arguments_icshift(nb, opt)
        nb = plot.apply_arguments_reduc(nb, opt)
        nb = plot.apply_arguments_select(nb, opt)

        # -----------------------------
        # SHRINKING-SPHERE CENTERING
        # -----------------------------
        if opt.center_mode == "shrinking":

            pos = nb.Pos(units="kpc")
            mass = nb.Mass(units="Msol")

            c = shrinking_sphere_center(pos, mass)

            print("-------------------------------------------------")
            print("Shrinking-sphere center:", c)
            print("-------------------------------------------------")

            nb.pos = nb.pos - c

        else:
            nb = plot.apply_arguments_center(nb, opt)

        # remaining pipeline
        nb = plot.apply_arguments_cmd(nb, opt)
        nb = plot.apply_arguments_info(nb, opt)

        print("---------------------------------------------------------")
        nb.localsystem_of_units.info()
        print("---------------------------------------------------------")

        rc = 1
        def f(r): return np.log(r / rc + 1.)
        def fm(r): return rc * (np.exp(r) - 1.)

        # =========================================================
        # DENSITY PROFILE
        # =========================================================
        if opt.y == 'density':

            nb.pos = nb.Pos(units="kpc")
            nb.mass = nb.Mass(units="Msol")

            G = libgrid.Spherical_1d_Grid(rmin=0, rmax=opt.rmax, nr=opt.nr, g=f, gm=fm)
            x = G.get_r()

            xlabel = "Radius [kpc]"
            ylabel = "Density [Msun/kpc^3]"

            y = G.get_DensityMap(nb)
            datas.append(plot.DataPoints(x, y, color=colors.get(), label=filename))

        # =========================================================
        # MASS PROFILE
        # =========================================================
        if opt.y == 'mass':

            nb.pos = nb.Pos(units="kpc")
            nb.mass = nb.Mass(units="Msol")

            G = libgrid.Spherical_1d_Grid(rmin=0, rmax=opt.rmax, nr=opt.nr, g=f, gm=fm)
            x = G.get_r()

            y = G.get_MassMap(nb)

            datas.append(plot.DataPoints(x, y, color=colors.get(), label=filename))

        # (all other branches unchanged in your original script)
        # =========================================================

    # plotting
    for d in datas:
        ax.plot(d.x, d.y, label=d.label)

    ax.set_xlabel("Radius")
    ax.set_ylabel(opt.y)
    ax.legend()

    if opt.xmin is not None: ax.set_xlim(left=opt.xmin)
    if opt.xmax is not None: ax.set_xlim(right=opt.xmax)
    if opt.ymin is not None: ax.set_ylim(bottom=opt.ymin)
    if opt.ymax is not None: ax.set_ylim(top=opt.ymax)

    if opt.log:
        if "x" in opt.log: ax.set_xscale("log")
        if "y" in opt.log: ax.set_yscale("log")

    plt.tight_layout()

    if opt.outputfilename:
        plt.savefig(opt.outputfilename, dpi=200)
    else:
        plt.show()


####################################################################
if __name__ == "__main__":
    opt = parser.parse_args()
    MakePlot(opt)
