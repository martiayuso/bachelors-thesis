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
    rho_ic = 1 * rho_ic

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
        rho = 1 * rho

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
        r"Density Profile Evolution for $\epsilon = 0.1\ \mathrm{kpc}$"
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
        "profile_chunk_0010_0014.hdf5.npz",
        "profile_chunk_0020_0024.hdf5.npz",
        "profile_chunk_0030_0034.hdf5.npz",
        "profile_chunk_0040_0044.hdf5.npz",
        "profile_chunk_0050_0054.hdf5.npz",
        "profile_chunk_0060_0064.hdf5.npz",
        "profile_chunk_0070_0074.hdf5.npz",
        "profile_chunk_0080_0084.hdf5.npz",
        "profile_chunk_0090_0094.hdf5.npz",
        "profile_chunk_0100_0104.hdf5.npz"
    ]

    labels = [
        "1 Gyr", "2 Gyr", "3 Gyr", "4 Gyr", "5 Gyr",
        "6 Gyr", "7 Gyr", "8 Gyr", "9 Gyr", "10 Gyr"
    ]

    plot(ic, sims, labels)
