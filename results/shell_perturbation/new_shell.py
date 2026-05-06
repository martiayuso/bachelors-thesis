import numpy as np
import matplotlib.pyplot as plt

# ================================
# 1. Physical Parameters
# ================================
G = 4.3009e-6      # kpc (km/s)^2 / Msun
rho_s = 1.0e7      # characteristic density [Msun/kpc^3]
r_s = 10.0         # scale radius [kpc]
r_vir = 100.0      # virial radius [kpc]
c = r_vir / r_s

# Resolution / inner limit (should match SWIFT softening scale ideally)
r_min = 0.05

# ================================
# 2. NFW Functions
# ================================
def f_nfw(x):
    return np.log1p(x) - x / (1.0 + x)

def m_enclosed(r):
    x = r / r_s
    return 4.0 * np.pi * rho_s * r_s**3 * f_nfw(x)

def rho_nfw(r):
    x = r / r_s
    return rho_s / (x * (1.0 + x)**2)

def phi_inner(r):
    """Potential from mass inside r"""
    r = np.asarray(r)
    return -G * m_enclosed(r) / r

def phi_outer(r):
    """Potential from shells outside r (truncated at r_vir)"""
    x = r / r_s
    return -4.0 * np.pi * G * rho_s * r_s**2 * (1.0 / (1.0 + x) - 1.0 / (1.0 + c))

def phi_total(r):
    return phi_inner(r) + phi_outer(r)

def vcirc(r):
    return np.sqrt(G * m_enclosed(r) / r)

# ================================
# 3. Radial Grid
# ================================
r = np.logspace(np.log10(r_min), np.log10(r_vir), 500)

# ================================
# 4. Mass Profile + Shell Contributions
# ================================
N_shells = 50
edges = np.logspace(np.log10(r_min), np.log10(r_vir), N_shells + 1)
r_mid = np.sqrt(edges[:-1] * edges[1:])

shell_mass = m_enclosed(edges[1:]) - m_enclosed(edges[:-1])
total_mass = m_enclosed(r_vir)
mass_fraction = shell_mass / total_mass

# ---- Plot 1: Mass profile ----
plt.figure(figsize=(8,6))
plt.plot(r, m_enclosed(r), linewidth=2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("r [kpc]")
plt.ylabel("M(<r) [Msun]")
plt.title("NFW Enclosed Mass Profile")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ---- Plot 2: Shell contributions ----
plt.figure(figsize=(8,6))
plt.plot(r_mid, mass_fraction, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("r [kpc]")
plt.ylabel("Shell Mass Fraction")
plt.title("Mass Contribution per Shell")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ================================
# 5. Potential Decomposition
# ================================
phi_in = phi_inner(r)
phi_out = phi_outer(r)
phi_tot = phi_total(r)

plt.figure(figsize=(8,6))
plt.plot(r, np.abs(phi_tot), label="Total")
plt.plot(r, np.abs(phi_in), linestyle="--", label="Inner contribution")
plt.plot(r, np.abs(phi_out), linestyle=":", label="Outer contribution")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("r [kpc]")
plt.ylabel("|Phi(r)| [(km/s)^2]")
plt.title("Potential Decomposition (NFW)")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ================================
# 6. Circular Velocity
# ================================
v_c = vcirc(r)

plt.figure(figsize=(8,6))
plt.plot(r, v_c, linewidth=2)
plt.xscale('log')
plt.xlabel("r [kpc]")
plt.ylabel("V_c [km/s]")
plt.title("Circular Velocity Profile")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ================================
# 7. Hybrid Cutoff Study (KEY RESULT)
# ================================
def phi_hybrid(r, r_cut):
    """
    Hybrid model:
    - Inside r_cut: full NFW
    - Outside r_cut: analytic continuation only
    """
    r = np.asarray(r)
    phi = np.zeros_like(r)

    for i, ri in enumerate(r):
        if ri < r_cut:
            # Keep full potential
            phi[i] = phi_total(ri)
        else:
            # Only analytic outer potential
            phi[i] = phi_outer(ri)

    return phi

# Candidate cutoff radii
cutoffs = [20, 30, 50, 70]

plt.figure(figsize=(8,6))

for rc in cutoffs:
    phi_h = phi_hybrid(r, rc)
    rel_error = np.abs((phi_h - phi_tot) / phi_tot)
    plt.plot(r, rel_error, label=f"r_cut = {rc} kpc")

plt.xscale('log')
plt.yscale('log')
plt.xlabel("r [kpc]")
plt.ylabel("Relative Error in Potential")
plt.title("Hybrid Model Convergence")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()
