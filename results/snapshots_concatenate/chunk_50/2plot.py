import numpy as np
import matplotlib.pyplot as plt
import glob
import os
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

    # ------------------------------------------------------------------
    # Figure with two panels
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={
            'height_ratios': [3, 1],
            'hspace': 0.05
        }
    )

    colors = cm.rainbow(np.linspace(0, 1, len(sim_files) + 1))

    # ------------------------------------------------------------------
    # Initial condition
    # ------------------------------------------------------------------
    ic_data = np.load(ic_file)

    r_ic = ic_data['radius']
    rho_ic = ic_data['density']

    rho_ic = 0.02 * rho_ic

    ax1.plot(
        r_ic,
        rho_ic,
        label="Initial condition",
        color=colors[0],
        linewidth=2.5
    )

    ax2.axhline(
        1.0,
        color=colors[0],
        alpha=0.5,
        linewidth=1
    )

    # ------------------------------------------------------------------
    # Simulation snapshots
    # ------------------------------------------------------------------
    for i, file in enumerate(sim_files):

        data = np.load(file)

        r = data['radius']
        rho = data['density']

        rho = 0.02 * rho

        # Density profile
        ax1.plot(
            r,
            rho,
            label=labels[i],
            color=colors[i + 1],
            linewidth=2.0
        )

        # Ratio panel
        rho_interp = np.interp(r_ic, r, rho)

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = rho_interp / rho_ic

        ax2.plot(
            r_ic,
            ratio,
            color=colors[i + 1],
            linewidth=2.0
        )

    # ------------------------------------------------------------------
    # Top panel formatting
    # ------------------------------------------------------------------
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax1.set_ylabel(
        r'$\rho\ [M_\odot/\rm{kpc}^3]$'
    )

    ax1.set_xlim(1e-2, 10)
    ax1.set_ylim(1e4, 1e10)

    ax1.grid(alpha=0.2)

    ax1.set_title(
        r"Density Profile Evolution for $\epsilon = 0.01\ \mathrm{kpc}$"
    )

    ax1.legend()

    # ------------------------------------------------------------------
    # Bottom panel formatting
    # ------------------------------------------------------------------
    ax2.set_ylabel(
        r'$\rho / \rho_{\rm init}$'
    )

    ax2.set_xlabel(
        r'$r\ [\rm{kpc}]$'
    )

    ax2.set_ylim(0.2, 1.2)

    ax2.grid(alpha=0.2)

    # ------------------------------------------------------------------
    # Tick formatting
    # ------------------------------------------------------------------
    for ax in [ax1, ax2]:

        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')

        ax.tick_params(
            axis='both',
            which='major',
            direction='in',
            length=6,
            width=1
        )

        ax.tick_params(
            axis='both',
            which='minor',
            direction='in',
            length=3,
            width=0.8
        )

    # ------------------------------------------------------------------
    # Softening scale
    # ------------------------------------------------------------------
    eps_value = 0.01
    v_line_pos = eps_value * 2.8

    ax1.axvline(
        v_line_pos,
        color='gray',
        linestyle='--',
        alpha=0.6,
        linewidth=1
    )

    ax2.axvline(
        v_line_pos,
        color='gray',
        linestyle='--',
        alpha=0.6,
        linewidth=1
    )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    fig.subplots_adjust(
        left=0.12,
        right=0.97,
        top=0.93,
        bottom=0.11,
        hspace=0.05
    )

    plt.savefig(
        "density_evo_100.png",
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.16
    )

    plt.show()


if __name__ == "__main__":

    ic = "profile_chunk_0000_0049.hdf5.npz"

    sims = [
        "profile_chunk_2000_2049.hdf5.npz",
        "profile_chunk_4000_4049.hdf5.npz",
        "profile_chunk_6000_6049.hdf5.npz",
        "profile_chunk_8000_8049.hdf5.npz",
        "profile_chunk_9950_9999.hdf5.npz"
    ]

    labels = [
        "20 Gyr",
        "40 Gyr",
        "60 Gyr",
        "80 Gyr",
        "100 Gyr"
    ]

    plot(ic, sims, labels)
