import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d
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
# Load snapshots
# ------------------------------------------------------------
snapshot = "snap/centered/rcut_10_mod.hdf5"
snapshot_baseline = "baseline/centered/snapshot_0001.hdf5"

rcut = 10.0

nb = Nbody(snapshot)
nb_base = Nbody(snapshot_baseline)

# Particle radii + potentials (main run)
r = nb.rxyz()
phi = nb.pot

# Baseline reference particles
r_base = nb_base.rxyz()
phi_base = nb_base.pot


# ------------------------------------------------------------
# Parameters used to generate the ICs
# ------------------------------------------------------------
rho0 = 0.0027658297046461644 * 1e10   # Msun / kpc^3
rs = 0.828177997155062                # kpc

r_cutoff = 50.0
power_cutoff = 3.0

G = 4.30091e-6


# ------------------------------------------------------------
# Density profile
# ------------------------------------------------------------
def rho_nfw(r, rho0, rs):
    x = r / rs
    return rho0 / (x * (1.0 + x)**2)


def rho_ic(r, rho0, rs, r_cutoff, power_cutoff):
    return rho_nfw(r, rho0, rs) * np.exp(-(r / r_cutoff)**power_cutoff)


# ------------------------------------------------------------
# Exact potential
# ------------------------------------------------------------
def phi_ic(r, rho0, rs, r_cutoff, power_cutoff, G=4.30091e-6):

    def phi_single(x):

        inner = quad(
            lambda s: s**2 * rho_ic(s, rho0, rs, r_cutoff, power_cutoff),
            0.0,
            x,
            epsabs=1e-8,
            epsrel=1e-8
        )[0]

        outer = quad(
            lambda s: s * rho_ic(s, rho0, rs, r_cutoff, power_cutoff),
            x,
            np.inf,
            epsabs=1e-8,
            epsrel=1e-8
        )[0]

        return -4.0 * np.pi * G * (inner / x + outer)

    return np.vectorize(phi_single)(r)


# ------------------------------------------------------------
# Analytical curve
# ------------------------------------------------------------
r_grid = np.linspace(1e-3, 55.0, 1000)

phi_ic_curve = phi_ic(
    r_grid,
    rho0,
    rs,
    r_cutoff,
    power_cutoff
)


# ------------------------------------------------------------
# IMPORTANT: align using BASELINE snapshot
# ------------------------------------------------------------
mask_base = (r_base > 1.0) & (r_base < 2.0)

phi_ic_base = phi_ic(
    r_base[mask_base],
    rho0,
    rs,
    r_cutoff,
    power_cutoff
)

offset = (
    np.median(phi_base[mask_base])
    - np.median(phi_ic_base)
)

phi_ic_curve += offset


# ------------------------------------------------------------
# Interpolate analytical potential at particle positions
# ------------------------------------------------------------
phi_interp = interp1d(
    r_grid,
    phi_ic_curve,
    kind="cubic",
    bounds_error=False,
    fill_value="extrapolate"
)

phi_ic_particles = phi_interp(r)

ratio = phi / phi_ic_particles


# ------------------------------------------------------------
# Radial binning for ratio panel
# ------------------------------------------------------------
nbins = 50
bins = np.linspace(0.0, 75.0, nbins + 1)
r_mid = 0.5 * (bins[:-1] + bins[1:])

ratio_med, _, _ = binned_statistic(r, ratio, statistic="median", bins=bins)

ratio_p16, _, _ = binned_statistic(
    r, ratio,
    statistic=lambda x: np.percentile(x, 16),
    bins=bins
)

ratio_p84, _, _ = binned_statistic(
    r, ratio,
    statistic=lambda x: np.percentile(x, 84),
    bins=bins
)


# ============================================================
# Figure
# ============================================================
fig, (ax, ax_ratio) = plt.subplots(
    2, 1,
    figsize=(7, 7),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 0.7], "hspace": 0.05}
)


# ------------------------------------------------------------
# Top panel
# ------------------------------------------------------------
ax.scatter(
    r, phi,
    s=2.5,
    alpha=1.0,
    rasterized=True,
    label="Simulation particles"
)

ax.plot(
    r_grid,
    phi_ic_curve,
    color="black",
    linewidth=2.0,
    label="Analytical profile"
)

ax.set_ylabel(r"$\Phi(r)\ [{\rm km}^2\,{\rm s}^{-2}]$")
ax.set_title(fr"Potential distribution for $R_{{\rm cut}}={rcut}\,\mathrm{{kpc}}$ at 100 Myr")
#ax.set_title(fr"Potential distribution of baseline simulation at 100 Myr")

ax.legend(frameon=False, loc="lower right")
ax.set_xlim(-2.0, 75.0)
ax.grid(False)


# ------------------------------------------------------------
# Bottom panel
# ------------------------------------------------------------
ax_ratio.fill_between(
    r_mid,
    ratio_p16,
    ratio_p84,
    alpha=0.3,
    linewidth=0
)

ax_ratio.plot(r_mid, ratio_med, linewidth=2.0)
ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=1.2)

ax_ratio.set_xlabel(r"$r\ [{\rm kpc}]$")
ax_ratio.set_ylabel(r"$\Phi_{\rm sim}/\Phi_{\rm ana}$")
#ax_ratio.set_ylim(0.8, 1.2)
ax_ratio.grid(False)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
plt.tight_layout()
plt.savefig("potential_comparison_ratio.pdf", dpi=300, bbox_inches="tight")
plt.show()
