import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

def plot(ic_file, sim_files, labels):

    params = {
        "text.usetex": True,
        "font.size": 20,
        "axes.titlesize": 26,
        "axes.labelsize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 24,
        "lines.linewidth": 2.8,
        "figure.figsize": (10, 8),
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
        label="Initial condition",
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
    # Color gradients
    # ------------------------------------------------------------------
    blue_colors = plt.cm.Blues(np.linspace(0.45, 0.85, 4))
    red_colors  = plt.cm.Reds(np.linspace(0.45, 0.85, 4))

    # ------------------------------------------------------------------
    # Plot simulations
    # ------------------------------------------------------------------
    for i, file in enumerate(sim_files):

        data = np.load(file)
        r, rho = data['radius'], data['density']

        # First 4 = 10 Gyr
        if i < 4:
            color = blue_colors[i]

        # Last 4 = 100 Gyr
        else:
            color = red_colors[i - 4]

        # Density profile
        ax1.plot(
            r,
            rho,
            color=color,
            linewidth=2.8
        )

        # Ratio profile
        rho_interp = np.interp(r_ic, r, rho)

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = rho_interp / rho_ic

        ax2.plot(
            r_ic,
            ratio,
            color=color,
            linewidth=2.5
        )

    # ------------------------------------------------------------------
    # Main legend:
    # IC first, then blue/red pair for each epsilon value
    # ------------------------------------------------------------------
    legend_handles = []
    legend_labels = []

    # IC entry FIRST
    ic_handle = Line2D([0], [0], color='0.4', lw=3)

    legend_handles.append(ic_handle)
    legend_labels.append("Initial condition")

    # epsilon entries
    for i in range(4):

        blue_line = Line2D([0], [0], color=blue_colors[i], lw=3)
        red_line  = Line2D([0], [0], color=red_colors[i], lw=3)

        legend_handles.append((blue_line, red_line))
        legend_labels.append(labels[i])

    legend1 = ax1.legend(
        legend_handles,
        legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc='upper right',
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor='0.8',
        handlelength=3.0,
        borderpad=0.8,
        labelspacing=0.6
    )
    '''
    # ------------------------------------------------------------------
    # Secondary legend:
    # year-color mapping
    # ------------------------------------------------------------------
    time_handles = [
        Line2D([0], [0], color=blue_colors[2], lw=3),
        Line2D([0], [0], color=red_colors[2], lw=3),
    ]

    time_labels = [
        "10 Gyr",
        "100 Gyr"
    ]

    legend2 = ax1.legend(
        time_handles,
        time_labels,
        loc='upper right',
        bbox_to_anchor=(1.0, 0.585),
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor='0.8',
        handlelength=3.0,
        borderpad=0.8,
        labelspacing=0.6
    )

    # Keep both legends
    ax1.add_artist(legend1)
    '''
    # ------------------------------------------------------------------
    # Top panel formatting
    # ------------------------------------------------------------------
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    ax1.set_ylabel(r'$\rho\ [M_\odot/\rm{kpc}^3]$')

    ax1.set_xlim(1e-2, 10)
    ax1.set_ylim(1e4, 2e10)

    ax1.grid(alpha=0.2, linewidth=0.8)

    ax1.set_title(
        r"$\epsilon_{\mathrm{fmm}}$ Study, Density Profile Evolution "
        r"for $\epsilon = 0.01\ \mathrm{kpc}$"
    )

    # ------------------------------------------------------------------
    # Bottom panel formatting
    # ------------------------------------------------------------------
    ax2.set_ylabel(r'$\rho / \rho_{\rm init}$')
    ax2.set_xlabel(r'$r\ [\rm{kpc}]$')

    ax2.set_ylim(0.2, 1.2)

    ax2.grid(alpha=0.2, linewidth=0.8)

    # ------------------------------------------------------------------
    # Vertical softening line
    # ------------------------------------------------------------------
    eps_value = 0.01
    v_line_pos = eps_value * 2.8

    for ax in (ax1, ax2):
        ax.axvline(
            v_line_pos,
            color='gray',
            linestyle='--',
            alpha=0.7,
            linewidth=1.5
        )

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
        "density_evo_100.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02
    )

    plt.show()


if __name__ == "__main__":

    ic = "profile_snapshot_0000.hdf5.npz"

    sims = [
        "test_3_1000.hdf5.npz",
        "test_1_1000.hdf5.npz",
        "orig_1000.hdf5.npz",
        "test_2_1000.hdf5.npz",
        "test_3_9999.hdf5.npz",
        "test_1_9999.hdf5.npz",
        "orig_9999.hdf5.npz",
        "test_2_9999.hdf5.npz"
    ]

    labels = [
        r"$\epsilon_{\mathrm{fmm}}$ = 0.1",
        r"$\epsilon_{\mathrm{fmm}}$ = 0.01",
        r"$\epsilon_{\mathrm{fmm}}$ = 0.001",
        r"$\epsilon_{\mathrm{fmm}}$ = 0.0001",
    ]

    plot(ic, sims, labels)
