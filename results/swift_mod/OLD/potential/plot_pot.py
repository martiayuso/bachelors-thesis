import numpy as np
import matplotlib.pyplot as plt
from pNbody import Nbody


# ------------------------------------------------------------
# Thesis-quality plotting style
# ------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "legend.fontsize": 18,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 1.0,
})


# ------------------------------------------------------------
# Load snapshot
# ------------------------------------------------------------
snapshot = "fix/centered/snapshot_0010.hdf5"

nb = Nbody(snapshot)

# Particle radii
r = nb.rxyz()

# Particle potentials
phi = nb.pot


# ------------------------------------------------------------
# Analytical NFW potential
# ------------------------------------------------------------
def nfw_potential(r, rho0, rs, G=4.30091e-6):
    """
    Analytical NFW gravitational potential.
    """

    x = r / rs

    return -4.0 * np.pi * G * rho0 * rs**2 * np.log(1.0 + x) / x


# ------------------------------------------------------------
# NFW parameters
# ------------------------------------------------------------
rho0 = 0.0027658297046461644 * 1e10
rs = 0.828177997155062

# Radius grid
r_grid_linear = np.linspace(1e-3, 200, 2000)
r_grid_log = np.logspace(-3, np.log10(50), 2000)

phi_linear = nfw_potential(r_grid_linear, rho0, rs)
phi_log = nfw_potential(r_grid_log, rho0, rs)


# ============================================================
# 1) LINEAR X-AXIS PLOT
# ============================================================
fig, ax = plt.subplots(figsize=(7, 6))

# Scatter
ax.scatter(
    r,
    phi,
    s=2.5,
    alpha=1.0,
    rasterized=True,
    label="Simulation particles"
)

# Analytical curve
ax.plot(
    r_grid_linear,
    phi_linear,
    color="black",
    linewidth=2.0,
    label="Analytical NFW potential"
)

ax.axvline(
    x=50.0,
    color="gray",
    linestyle="--",
    linewidth=1.2,
    alpha=0.0
)

# Formatting
ax.set_xlabel(r"$r\ [{\rm kpc}]$")
ax.set_ylabel(r"$\Phi(r)\ [{\rm km}^2\,{\rm s}^{-2}]$")

ax.set_title(
   # r"Potential distribution for $R_{\rm cut} = 10.0\ \mathrm{kpc}$ at 10 Gyr"
   "Potential distribution of baseline simulation at 10 Gyr"
)

ax.grid(alpha=0.0)

ax.legend(frameon=False)

ax.set_xlim(-5, 150)

plt.tight_layout()

plt.savefig(
    "nfw_potential_linear.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


