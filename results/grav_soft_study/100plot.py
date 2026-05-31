import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot(ic_file, sim_files, labels):

    params = {
        "text.usetex": True,
        "font.size": 20,
        "axes.titlesize": 24,
        "axes.labelsize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
        "lines.linewidth": 2.8,
        "figure.figsize": (9, 14),
        "axes.linewidth": 1.5,
    }
    plt.rcParams.update(params)

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}
    )

    for ax in (ax1, ax2):
        ax.tick_params(axis='both', which='both', width=1.5, length=6)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    # ------------------------------------------------------------------
    # Initial condition
    # ------------------------------------------------------------------
    ic_data = np.load(ic_file)
    r_ic, rho_ic = ic_data['radius'], ic_data['density']

    ax1.plot(
        r_ic,
        rho_ic,
        color='0.4',
        linewidth=2.0
    )

    ax2.axhline(
        1.0,
        color='0.4',
        alpha=0.8,
        linewidth=2.0
    )

    # ------------------------------------------------------------------
    # Colors
    # ------------------------------------------------------------------
    colors = [
        'tab:orange',
        'tab:green',
        'tab:red',
        'tab:purple',
        'tab:brown'
    ]

    eps_values = [1, 5, 10, 50, 100]  # pc

    # ------------------------------------------------------------------
    # Plot simulations
    # ------------------------------------------------------------------
    for i, file in enumerate(sim_files):

        data = np.load(file)
        r, rho = data['radius'], data['density']

        current_color = colors[i]

        # Density profile
        ax1.plot(
            r,
            rho,
            color=current_color,
            linewidth=2
        )

        # Ratio profile
        rho_interp = np.interp(r_ic, r, rho)

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = rho_interp / rho_ic

        ax2.plot(
            r_ic,
            ratio,
            color=current_color,
            linewidth=2
        )

        # Vertical softening lines
        v_line_pos = (eps_values[i] * 2.8) / 1000.0

        ax1.axvline(
            v_line_pos,
            color=current_color,
            linestyle='--',
            alpha=0.7,
            linewidth=1.5
        )

        ax2.axvline(
            v_line_pos,
            color=current_color,
            linestyle='--',
            alpha=0.7,
            linewidth=1.5
        )

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    legend_handles = []
    legend_labels = []

    # IC first
    ic_handle = Line2D([0], [0], color='0.4', lw=3)

    legend_handles.append(ic_handle)
    legend_labels.append("Initial condition")

    # epsilon entries
    for i in range(len(labels)):

        handle = Line2D([0], [0], color=colors[i], lw=3)

        legend_handles.append(handle)
        legend_labels.append(labels[i])

    ax1.legend(
        legend_handles,
        legend_labels,
        loc='upper right',
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor='0.8',
        handlelength=1.0,
        borderpad=0.8,
        labelspacing=0.6
    )

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    ax1.set_ylabel(r'$\rho\ [M_\odot/\rm{kpc}^3]$')

    ax1.set_xlim(1e-2, 10)
    ax1.set_ylim(1e4, 2e10)

    ax1.grid(alpha=0.2, linewidth=0.8)

    ax1.set_title(
        r"Density Profiles for Different $\epsilon$, at 100 Gyr"
    )

    # ------------------------------------------------------------------
    # Bottom panel formatting
    # ------------------------------------------------------------------
    ax2.set_ylabel(r'$\rho / \rho_{\rm init}$')

    ax2.set_xlabel(r'$r\ [\rm{kpc}]$')

    ax2.set_ylim(0.2, 1.6)

    ax2.grid(alpha=0.2, linewidth=0.8)

    # ------------------------------------------------------------------
    # Layout / save
    # ------------------------------------------------------------------
    plt.subplots_adjust(
        left=0.12,
        right=0.98,
        top=0.92,
        bottom=0.10,
        hspace=0.05
    )

    plt.savefig(
        "density_profile.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02
    )

    plt.show()


if __name__ == "__main__":

    ic = "profile_snapshot_0000.hdf5.npz"

    sims = [
        "profile_snapshot_9999_0.001.hdf5.npz",
        "profile_snapshot_9999_0.005.hdf5.npz",
        "profile_snapshot_9999_0.01.hdf5.npz",
        "profile_snapshot_9999_0.05.hdf5.npz",
        "profile_snapshot_9999_0.1.hdf5.npz",
    ]

    labels = [
        r"$\epsilon$ = 1 pc",
        r"$\epsilon$ = 5 pc",
        r"$\epsilon$ = 10 pc",
        r"$\epsilon$ = 50 pc",
        r"$\epsilon$ = 100 pc"
    ]

    plot(ic, sims, labels)
