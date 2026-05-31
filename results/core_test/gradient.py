import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot(ic_file, sim_files, labels):
    params = {
        "text.usetex": True,
        "font.size": 12,
        "axes.labelsize": 16,
        "legend.fontsize": 12,
        "figure.figsize": (10, 8)
    }
    plt.rcParams.update(params)

    # Single figure/axis
    fig, ax = plt.subplots()


    colors = cm.rainbow(np.linspace(0, 1, len(sim_files) + 1))

    # Load initial condition
    ic_data = np.load(ic_file)
    r_ic, rho_ic = ic_data['radius'], ic_data['density']
    rho_ic = 0.01 * rho_ic

    ax.plot(
        r_ic,
        rho_ic,
        label="Initial condition",
        color=colors[0],
        linewidth=2.5
    )

    # Plot simulations
    for i, file in enumerate(sim_files):
        data = np.load(file)
        r, rho = data['radius'], data['density']
        rho = 0.01 * rho

        ax.plot(
            r,
            rho,
            label=labels[i],
            color=colors[i + 1],
            linewidth=2.5
        )

    # Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlim(1e-2, 10)

    ax.set_xlabel(r'$r\ [\rm{kpc}]$')
    ax.set_ylabel(r'$\rho\ [M_\odot/\rm{kpc}^3]$')

    ax.grid(alpha=0.2)

    ax.set_title(
        r"Density Profile Evolution for $\epsilon = 0.01\ \mathrm{kpc}$"
    )

    # Softening line
    eps_value = 0.01
    v_line_pos = eps_value * 2.8

    ax.axvline(
        v_line_pos,
        color='gray',
        linestyle='--',
        alpha=0.6,
        linewidth=1
    )

    ax.legend()

    plt.tight_layout()
    plt.savefig("density_evo_100.png", dpi=300)
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

    labels = [
        "10 Gyr", "20 Gyr", "30 Gyr", "40 Gyr", "50 Gyr",
        "60 Gyr", "70 Gyr", "80 Gyr", "90 Gyr", "100 Gyr"
    ]

    plot(ic, sims, labels)
