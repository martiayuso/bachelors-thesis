import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def plot(ic_file, sim_files, labels):
    params = {
        "text.usetex": True,
        "font.size": 12,
        "axes.labelsize": 16,
        "legend.fontsize": 12,
        "figure.figsize": (10, 8)
    }
    plt.rcParams.update(params)

    # Create figure with two subplots (ratio 3:1)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

    # 1. Load and plot Initial Condition
    ic_data = np.load(ic_file)
    r_ic, rho_ic = ic_data['radius'], ic_data['density']
    
    ax1.plot(r_ic, rho_ic, label="Initial condition", color='tab:blue', linewidth=1.5)
    ax2.axhline(1.0, color='tab:blue', alpha=0.5, linewidth=1)

    # 2. Loop through simulation files
    colors = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    eps_values = [1, 5, 10, 38, 72] # eps in pc to match your sims list
    
    for i, file in enumerate(sim_files):
        data = np.load(file)
        r, rho = data['radius'], data['density']
        current_color = colors[i % len(colors)]
        
        # Plot Density on the main axis
        ax1.plot(r, rho, label=labels[i], color=current_color)
        
        # Interpolate rho onto the r_ic grid
        rho_interp = np.interp(r_ic, r, rho)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = rho_interp / rho_ic
        
        ax2.plot(r_ic, ratio, color=current_color)

        # --- ADD VERTICAL LINES HERE ---
        # 2.8 * epsilon converted from pc to kpc
        v_line_pos = (eps_values[i] * 2.8) / 1000.0
        ax1.axvline(v_line_pos, color=current_color, linestyle='--', alpha=0.6, linewidth=1)
        ax2.axvline(v_line_pos, color=current_color, linestyle='--', alpha=0.6, linewidth=1)

    # Formatting Top Panel
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_ylabel(r'$\rho\ [M_\odot/\rm{kpc}^3]$')
    ax1.set_xlim(1e-2, 10)
    ax1.set_ylim(1e4, 2e10)
    ax1.legend()
    ax1.grid(alpha=0.2)
    ax1.set_title(r"Density Profiles for Different $\epsilon$, at 10 Gyr")    

    # Formatting Bottom Panel
    ax2.set_ylabel(r'$\rho / \rho_{\rm init}$')
    ax2.set_xlabel(r'$r\ [\rm{kpc}]$')
    ax2.set_ylim(0.4, 1.2)
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig("density_profile.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    ic = "profile_snapshot_0000.hdf5.npz"
    sims = [
        "profile_snapshot_1001_1.hdf5.npz",
        "profile_snapshot_1001_5.hdf5.npz", 
        "profile_snapshot_1001_10.hdf5.npz",
        "profile_snapshot_1001_38.hdf5.npz",
        "profile_snapshot_1001_72.hdf5.npz",
    ]
    labels = ["$\epsilon = 1$ pc", "$\epsilon = 5$ pc", "$\epsilon = 10$ pc", "$\epsilon = 38$ pc", "$\epsilon = 72$ pc"]
    
    plot(ic, sims, labels)