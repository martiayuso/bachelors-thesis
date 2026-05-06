import numpy as np
import matplotlib.pyplot as plt
import h5py

# ============================================================
# USER INPUT
# ============================================================
snapshot_file = "snapshot_0100.hdf5"

# NFW parameters (must match your ICs / halo!)
rho_s = 1.0e7      # Msun/kpc^3
r_s = 10.0         # kpc
r_vir = 100.0      # kpc

# Candidate cutoff radii (kpc)
cutoffs = [20, 30, 50, 70]

# Inner radius limit (avoid numerical issues)
r_min = 0.05

# Gravitational constant
G = 4.3009e-6  # kpc (km/s)^2 / Msun

# ============================================================
# NFW FUNCTIONS
# ============================================================
def f_nfw(x):
    return np.log1p(x) - x / (1.0 + x)

def m_enclosed_nfw(r):
    x = r / r_s
    return 4.0 * np.pi * rho_s * r_s**3 * f_nfw(x)

def a_nfw(r):
    return G * m_enclosed_nfw(r) / r**2

# ============================================================
# LOAD SWIFT SNAPSHOT
# ============================================================
print("Loading snapshot...")

with h5py.File(snapshot_file, "r") as f:
    coords = f["/PartType1/Coordinates"][:]   # (N,3)

    # Mass handling
    if "/PartType1/Masses" in f:
        masses = f["/PartType1/Masses"][:]
    else:
        mass_value = f["/Header"].attrs["MassTable"][1]
        masses = np.full(coords.shape[0], mass_value)

print(f"Loaded {coords.shape[0]} particles")

# ============================================================
# CENTER HALO
# ============================================================
print("Centering halo...")

# Simple COM centering
center = np.average(coords, axis=0, weights=masses)
coords -= center

# Radii
radii = np.sqrt(np.sum(coords**2, axis=1))

# ============================================================
# SORT + ENCLOSED MASS
# ============================================================
print("Computing enclosed mass profile...")

order = np.argsort(radii)
r_sorted = radii[order]
m_sorted = masses[order]

cum_mass = np.cumsum(m_sorted)

def enclosed_mass_particles(rgrid):
    return np.interp(rgrid, r_sorted, cum_mass,
                     left=0.0, right=cum_mass[-1])

# ============================================================
# RADIAL GRID
# ============================================================
r = np.logspace(np.log10(r_min), np.log10(r_vir), 300)

M_part = enclosed_mass_particles(r)
a_part = G * M_part / r**2

a_analytic = a_nfw(r)

# ============================================================
# PLOT 1: FORCE PROFILE
# ============================================================
plt.figure(figsize=(8,6))
plt.plot(r, a_analytic, label="Analytic NFW", linewidth=2)
plt.plot(r, a_part, '--', label="Particle halo", linewidth=2)

plt.xscale('log')
plt.yscale('log')
plt.xlabel("r [kpc]")
plt.ylabel(r"$|a(r)|$ [(km/s)$^2$/kpc]")
plt.title("Force Profile Comparison")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ============================================================
# HYBRID MODEL FORCE
# ============================================================
def a_hybrid(rgrid, r_cut):
    """
    Hybrid force:
    - Inside r_cut: particle halo
    - Outside r_cut: analytic NFW
    """
    a = np.zeros_like(rgrid)

    for i, ri in enumerate(rgrid):
        if ri < r_cut:
            a[i] = a_part[i]
        else:
            a[i] = a_analytic[i]

    return a

# ============================================================
# PLOT 2: FORCE CONVERGENCE (KEY RESULT)
# ============================================================
plt.figure(figsize=(8,6))

for rc in cutoffs:
    a_h = a_hybrid(r, rc)

    # Only evaluate inside region of interest
    mask = r < rc

    rel_err = np.abs(a_h[mask] - a_analytic[mask]) / a_analytic[mask]

    plt.plot(r[mask], rel_err, label=f"$r_{{cut}}={rc}$ kpc")

plt.xscale('log')
plt.yscale('log')
plt.xlabel("r [kpc]")
plt.ylabel("Relative Force Error")
plt.title("Hybrid Model Convergence (Force-Based)")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ============================================================
# PRINT SUMMARY METRIC
# ============================================================
print("\n=== Cutoff Summary ===")

for rc in cutoffs:
    mask = r < rc
    a_h = a_hybrid(r, rc)
    rel_err = np.abs(a_h[mask] - a_analytic[mask]) / a_analytic[mask]
    print(f"r_cut = {rc:>4} kpc  ->  max error = {np.max(rel_err):.3e}")
