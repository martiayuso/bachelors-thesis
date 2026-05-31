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
    # colors to match your reference image
    colors = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    
    for i, file in enumerate(sim_files):
        data = np.load(file)
        r, rho = data['radius'], data['density']
        
        # Plot Density on the main axis
        ax1.plot(r, rho, label=labels[i], color=colors[i % len(colors)])
        
        # Interpolate rho onto the r_ic grid so they match shapes
        # This handles the (150,) vs (350,) problem!
        rho_interp = np.interp(r_ic, r, rho)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = rho_interp / rho_ic
        
        ax2.plot(r_ic, ratio, color=colors[i % len(colors)])

    # Formatting Top Panel
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_ylabel(r'$\rho\ [M_\odot/\rm{kpc}^3]$')
    ax1.set_xlim(1e-2, 10)
    ax1.set_ylim(1e4, 2e10)
    ax1.legend()
    ax1.grid(alpha=0.2)
    ax1.set_title(r"Density Profile Evolution for $\epsilon = 0.01\ \mathrm{kpc}$")    

    # Formatting Bottom Panel
    ax2.set_ylabel(r'$\rho / \rho_{\rm init}$')
    ax2.set_xlabel(r'$r\ [\rm{kpc}]$')
    ax2.set_ylim(0.4, 1.2)
    ax2.grid(alpha=0.2)

    # Add vertical lines for softening
    eps_value = 0.01
    v_line_pos = (eps_value * 2.8)
    ax1.axvline(v_line_pos, color='gray', linestyle='--', alpha=0.6, linewidth=1)
    ax2.axvline(v_line_pos, color='gray', linestyle='--', alpha=0.6, linewidth=1)

    # Reduce outer whitespace
    fig.subplots_adjust(left=0.11, right=0.98, top=0.93, bottom=0.10, hspace=0.05)

    plt.savefig("density_evo_100.png", dpi=300, bbox_inches='tight', pad_inches=0.16)
    plt.show()

if __name__ == "__main__":
    ic = "profile_chunk_0000_0004.hdf5.npz"

    sims = [
        "profile_chunk_0100_0104.hdf5.npz",
        "profile_chunk_0200_0204.hdf5.npz",
        "profile_chunk_0300_0304.hdf5.npz",
        "profile_chunk_0400_0404.hdf5.npz",
        "profile_chunk_0500_0504.hdf5.npz",
        "profile_chunk_0600_0604.hdf5.npz",
        "profile_chunk_0700_0704.hdf5.npz",
        "profile_chunk_0800_0804.hdf5.npz",
        "profile_chunk_0900_0904.hdf5.npz",
        "profile_chunk_0995_0999.hdf5.npz"
    ]
    labels = ["20 Gyr", "40 Gyr", "60 Gyr", "80 Gyr", "100 Gyr"]

    
    plot(ic, sims, labels)
