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
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}
    )

    # 1. Load and plot Initial Condition
    ic_data = np.load(ic_file)
    r_ic, rho_ic = ic_data['radius'], ic_data['density']

    ax1.plot(
        r_ic,
        rho_ic,
        label="Initial condition",
        color='tab:blue',
        linewidth=1.5
    )

    ax2.axhline(
        1.0,
        color='tab:blue',
        alpha=0.5,
        linewidth=1
    )

    # 2. Loop through simulation files
    colors = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

    for i, file in enumerate(sim_files):
        data = np.load(file)
        r, rho = data['radius'], data['density']

        # Plot Density on the main axis
        ax1.plot(
            r,
            rho,
            label=labels[i],
            color=colors[i % len(colors)]
        )

        # Interpolate rho onto the r_ic grid
        rho_interp = np.interp(r_ic, r, rho)

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = rho_interp / rho_ic

        ax2.plot(
            r_ic,
            ratio,
            color=colors[i % len(colors)]
        )


    # Formatting Top Panel
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_ylabel(r'$\rho\ [M_\odot/\rm{kpc}^3]$')

    # Extend x-axis to 100 kpc
    ax1.set_xlim(1e-2, 100)

    ax1.set_ylim(1e2, 2e9)
    ax1.legend()
    ax1.grid(alpha=0.2)

    ax1.set_title(
        r"Density Profile Evolution for $\epsilon = 0.1\ \mathrm{kpc}$"
    )

    # Formatting Bottom Panel
    ax2.set_ylabel(r'$\rho / \rho_{\rm init}$')
    ax2.set_xlabel(r'$r\ [\rm{kpc}]$')

    ax2.set_xscale('log')
    ax2.set_xlim(1e-2, 100)

    ax2.set_ylim(0.4, 1.2)
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig("density_evo_100kpc.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    ic = "profile_snapshot_0000.hdf5.npz"

    sims = [
        "profile_snapshot_0009.hdf5.npz",
        "profile_snapshot_0090_orig.hdf5.npz",
        "profile_speed1.hdf5.npz",
        "profile_speed2.hdf5.npz"
    ]

    labels = [
        "9 Gyr mod", "9 Gyr orig", "speed1", "speed2"
    ]

    plot(ic, sims, labels)
