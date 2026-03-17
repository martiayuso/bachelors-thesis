import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define Halo & Physical Parameters ---
rho_0 = 1.0e7        # Characteristic density (M_sun / kpc^3)
R_s = 10.0           # Scale radius (kpc)
R_vir = 100.0        # Virial radius (outer edge, kpc)
G = 4.3009e-6        # Gravitational constant in kpc (km/s)^2 / M_sun
m_p = 1.0e5          # Mass of a single dark matter "particle" (M_sun)
v_rel = 200.0        # Typical velocity dispersion/relative velocity (km/s)

def integrated_rho_dr(r, rho_0, R_s):
    """The exact analytical integral of rho(r) dr for an NFW profile."""
    x = r / R_s
    return rho_0 * R_s * (np.log(x / (1.0 + x)) + 1.0 / (1.0 + x))

# --- 2. Define the Shells (Logarithmic)
N_shells = 50
shell_edges = np.logspace(np.log10(0.01), np.log10(R_vir), N_shells + 1)

shell_centers = []
velocity_perturbations = []

# --- 3. Calculate Perturbation per Shell
for i in range(N_shells):
    r_in = shell_edges[i]
    r_out = shell_edges[i+1]
    
    # Midpoint just for the x-axis of the plot
    r_mid = (r_in + r_out) / 2.0  
    shell_centers.append(r_mid)
    
    # Calculate the integral for the shell boundaries
    exact_integral = integrated_rho_dr(r_out, rho_0, R_s) - integrated_rho_dr(r_in, rho_0, R_s)
    
    # Total exact velocity variance contribution from this shell
    constant_term = (16.0 * np.pi * G**2 * m_p) / (v_rel**2)
    delta_V_sq = constant_term * exact_integral
    
    velocity_perturbations.append(delta_V_sq)

shell_centers = np.array(shell_centers)
velocity_perturbations = np.array(velocity_perturbations)

# --- 4. Plotting
plt.figure(figsize=(9, 6))
plt.plot(shell_centers, velocity_perturbations, marker='o', linestyle='-', color='purple', linewidth=2)

plt.axvline(R_s, color='gray', linestyle='--', label=f'Scale Radius $R_s$ ({R_s} kpc)')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Distance from center (Shell Radius $r$) [kpc]')
plt.ylabel(r'Velocity Perturbation Contribution $\Delta V^2$ [(km/s)$^2$]')
plt.title('Analytical Perturbation Contribution of NFW Shells')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)

plt.ylim(min(velocity_perturbations)*0.5, max(velocity_perturbations)*2.0)
plt.show()