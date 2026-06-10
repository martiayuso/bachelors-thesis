import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# ---------------------------------------------------------------------
# NFW profile
# ---------------------------------------------------------------------
def nfw_profile(r, rho0, rs, alpha=1.0, beta=3.0):

    x = r / rs

    return rho0 / (x**alpha * (1.0 + x)**(beta - alpha))


# ---------------------------------------------------------------------
# Plot one density profile panel
# ---------------------------------------------------------------------
def plot_density_panel(ax, sim_files, labels, r_cutoff, title):

    colors = cm.rainbow(np.linspace(0, 1, len(sim_files) + 1))

    # -----------------------------------------------------------------
    # Analytical profile
    # -----------------------------------------------------------------
    rho0 = 0.0027658297046461644 * 1e10
    rs = 0.828177997155062

    alpha = 1.0
    beta = 3.0

    r_nfw = np.logspace(-3, np.log10(50), 1000)

    rho_nfw = nfw_profile(
        r_nfw,
        rho0,
        rs,
        alpha,
        beta
    )

    # Plot analytical profile
    ax.plot(
        r_nfw,
        rho_nfw,
        color=colors[0],
        linewidth=2.5,
        label="IC"
    )
    
    # Vertical line showing Rcut
    ax.axvline(
        r_cutoff,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=r"$R_{\rm cut}$"
    )

    # -----------------------------------------------------------------
    # Simulation snapshots
    # -----------------------------------------------------------------
    for i, file in enumerate(sim_files):

        data = np.load(file)

        r = data["radius"]
        rho = data["density"]

        rho = 0.2 * rho

        ax.plot(
            r,
            rho,
            color=colors[i + 1],
            linewidth=2.0,
            label=labels[i]
        )

    # -----------------------------------------------------------------
    # Formatting
    # -----------------------------------------------------------------
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(1e-2, 10)
    ax.set_ylim(1e4, 1e10)

    ax.grid(alpha=0.2)

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        length=6
    )

    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=3
    )

    ax.set_title(title)


def make_figure(all_runs, labels):

    plt.rcParams.update({
        "text.usetex": True,
        "font.size": 22,
        "axes.labelsize": 24,
        "axes.titlesize": 24,
        "legend.fontsize": 20,

        # Cleaner ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,

        # Thin axes
        "axes.linewidth": 0.8,
    })

    # -------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(16, 12))

    # Custom grid
    gs = fig.add_gridspec(
        3,
        4,
        left=0.06,
        right=0.95,
        bottom=0.06,
        top=0.97,
        wspace=0.18,
        hspace=0.18
    )

    axes = []

    # -------------------------------------------------------------
    # 12 panels
    # -------------------------------------------------------------
    for row in range(3):
        for col in range(4):

            ax = fig.add_subplot(gs[row, col])

            # FORCE square panels
            ax.set_box_aspect(1)

            axes.append(ax)

    # -------------------------------------------------------------
    # Plot all runs
    # -------------------------------------------------------------
    for i, run in enumerate(all_runs):

        ax = axes[i]

        plot_density_panel(
            ax=ax,
            sim_files=run["files"],
            labels=labels,
            r_cutoff=run["Rcut"],
            title=rf"$R_{{\rm cut}} = {run['Rcut']}\ \rm kpc$"
        )

    # -------------------------------------------------------------
    # Clean tick labels
    # -------------------------------------------------------------
    for i, ax in enumerate(axes):

        row = i // 4
        col = i % 4

        # ---------------------------------------------------------
        # X labels only on bottom row
        # ---------------------------------------------------------
        if row == 2:
            ax.set_xlabel(r"$r\ [{\rm kpc}]$")
        else:
            ax.set_xticklabels([])

        # ---------------------------------------------------------
        # Y labels only on left column
        # ---------------------------------------------------------
        if col == 0:
            ax.set_ylabel(
                r"$\rho\ [M_\odot/{\rm kpc}^3]$"
            )
        else:
            ax.set_yticklabels([])

    # -------------------------------------------------------------
    # Legend in former last-row space
    # -------------------------------------------------------------
    handles, legend_labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.07),
        frameon=False,
        ncol=6,
        handlelength=1.5,
        handletextpad=0.5,
        borderaxespad=0.0
    )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------
    plt.savefig(
        "density_profiles_multi_panel.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":

    labels = [
        "10 Gyr",
        "20 Gyr",
        "30 Gyr",
        "40 Gyr",
        "50 Gyr",
        "60 Gyr",
        "70 Gyr",
        "80 Gyr",
        "90 Gyr",
        "100 Gyr"
    ]

# -------------------------------------------------------------
# Each entry = one subplot
# -------------------------------------------------------------
all_runs = [
    {
        "Rcut": 5.0,
        "files": [
            "test12/profile_chunk_0100_0104.hdf5.npz",
            "test12/profile_chunk_0200_0204.hdf5.npz",
            "test12/profile_chunk_0300_0304.hdf5.npz",
            "test12/profile_chunk_0400_0404.hdf5.npz",
            "test12/profile_chunk_0500_0504.hdf5.npz",
            "test12/profile_chunk_0600_0604.hdf5.npz",
            "test12/profile_chunk_0700_0704.hdf5.npz",
            "test12/profile_chunk_0800_0804.hdf5.npz",
            "test12/profile_chunk_0900_0904.hdf5.npz",
            "test12/profile_chunk_0995_0999.hdf5.npz",
        ]
    },
    {
        "Rcut": 3.0,
        "files": [
            "test/profile_chunk_0100_0104.hdf5.npz",
            "test/profile_chunk_0200_0204.hdf5.npz",
            "test/profile_chunk_0300_0304.hdf5.npz",
            "test/profile_chunk_0400_0404.hdf5.npz",
            "test/profile_chunk_0500_0504.hdf5.npz",
            "test/profile_chunk_0600_0604.hdf5.npz",
            "test/profile_chunk_0700_0704.hdf5.npz",
            "test/profile_chunk_0800_0804.hdf5.npz",
            "test/profile_chunk_0900_0904.hdf5.npz",
            "test/profile_chunk_0995_0999.hdf5.npz",
        ]
    },

    {
        "Rcut": 2.0,
        "files": [
            "test2/profile_chunk_0100_0104.hdf5.npz",
            "test2/profile_chunk_0200_0204.hdf5.npz",
            "test2/profile_chunk_0300_0304.hdf5.npz",
            "test2/profile_chunk_0400_0404.hdf5.npz",
            "test2/profile_chunk_0500_0504.hdf5.npz",
            "test2/profile_chunk_0600_0604.hdf5.npz",
            "test2/profile_chunk_0700_0704.hdf5.npz",
            "test2/profile_chunk_0800_0804.hdf5.npz",
            "test2/profile_chunk_0900_0904.hdf5.npz",
            "test2/profile_chunk_0995_0999.hdf5.npz",
        ]
    },

    {
        "Rcut": 1.0,
        "files": [
            "test3/profile_chunk_0100_0104.hdf5.npz",
            "test3/profile_chunk_0200_0204.hdf5.npz",
            "test3/profile_chunk_0300_0304.hdf5.npz",
            "test3/profile_chunk_0400_0404.hdf5.npz",
            "test3/profile_chunk_0500_0504.hdf5.npz",
            "test3/profile_chunk_0600_0604.hdf5.npz",
            "test3/profile_chunk_0700_0704.hdf5.npz",
            "test3/profile_chunk_0800_0804.hdf5.npz",
            "test3/profile_chunk_0900_0904.hdf5.npz",
            "test3/profile_chunk_0995_0999.hdf5.npz",
        ]
    },

    {
        "Rcut": 0.5,
        "files": [
            "test4/profile_chunk_0100_0104.hdf5.npz",
            "test4/profile_chunk_0200_0204.hdf5.npz",
            "test4/profile_chunk_0300_0304.hdf5.npz",
            "test4/profile_chunk_0400_0404.hdf5.npz",
            "test4/profile_chunk_0500_0504.hdf5.npz",
            "test4/profile_chunk_0600_0604.hdf5.npz",
            "test4/profile_chunk_0700_0704.hdf5.npz",
            "test4/profile_chunk_0800_0804.hdf5.npz",
            "test4/profile_chunk_0900_0904.hdf5.npz",
            "test4/profile_chunk_0995_0999.hdf5.npz",
        ]
    },

    {
        "Rcut": 0.3,
        "files": [
            "test5/profile_chunk_0100_0104.hdf5.npz",
            "test5/profile_chunk_0200_0204.hdf5.npz",
            "test5/profile_chunk_0300_0304.hdf5.npz",
            "test5/profile_chunk_0400_0404.hdf5.npz",
            "test5/profile_chunk_0500_0504.hdf5.npz",
            "test5/profile_chunk_0600_0604.hdf5.npz",
            "test5/profile_chunk_0700_0704.hdf5.npz",
            "test5/profile_chunk_0800_0804.hdf5.npz",
            "test5/profile_chunk_0900_0904.hdf5.npz",
            "test5/profile_chunk_0995_0999.hdf5.npz",
        ]
    },

    {
        "Rcut": 0.2,
        "files": [
            "test6/profile_chunk_0100_0104.hdf5.npz",
            "test6/profile_chunk_0200_0204.hdf5.npz",
            "test6/profile_chunk_0300_0304.hdf5.npz",
            "test6/profile_chunk_0400_0404.hdf5.npz",
            "test6/profile_chunk_0500_0504.hdf5.npz",
            "test6/profile_chunk_0600_0604.hdf5.npz",
            "test6/profile_chunk_0700_0704.hdf5.npz",
            "test6/profile_chunk_0800_0804.hdf5.npz",
            "test6/profile_chunk_0900_0904.hdf5.npz",
            "test6/profile_chunk_0995_0999.hdf5.npz",
        ]
    },

    {
        "Rcut": 0.1,
        "files": [
            "test7/profile_chunk_0100_0104.hdf5.npz",
            "test7/profile_chunk_0200_0204.hdf5.npz",
            "test7/profile_chunk_0300_0304.hdf5.npz",
            "test7/profile_chunk_0400_0404.hdf5.npz",
            "test7/profile_chunk_0500_0504.hdf5.npz",
            "test7/profile_chunk_0600_0604.hdf5.npz",
            "test7/profile_chunk_0700_0704.hdf5.npz",
            "test7/profile_chunk_0800_0804.hdf5.npz",
            "test7/profile_chunk_0900_0904.hdf5.npz",
            "test7/profile_chunk_0995_0999.hdf5.npz",
        ]
    },

    {
        "Rcut": 0.05,
        "files": [
            "test8/profile_chunk_0100_0104.hdf5.npz",
            "test8/profile_chunk_0200_0204.hdf5.npz",
            "test8/profile_chunk_0300_0304.hdf5.npz",
            "test8/profile_chunk_0400_0404.hdf5.npz",
            "test8/profile_chunk_0500_0504.hdf5.npz",
            "test8/profile_chunk_0600_0604.hdf5.npz",
            "test8/profile_chunk_0700_0704.hdf5.npz",
            "test8/profile_chunk_0800_0804.hdf5.npz",
            "test8/profile_chunk_0900_0904.hdf5.npz",
            "test8/profile_chunk_0995_0999.hdf5.npz",
        ]
    },

    {
        "Rcut": 0.03,
        "files": [
            "test9/profile_chunk_0100_0104.hdf5.npz",
            "test9/profile_chunk_0200_0204.hdf5.npz",
            "test9/profile_chunk_0300_0304.hdf5.npz",
            "test9/profile_chunk_0400_0404.hdf5.npz",
            "test9/profile_chunk_0500_0504.hdf5.npz",
            "test9/profile_chunk_0600_0604.hdf5.npz",
            "test9/profile_chunk_0700_0704.hdf5.npz",
            "test9/profile_chunk_0800_0804.hdf5.npz",
            "test9/profile_chunk_0900_0904.hdf5.npz",
            "test9/profile_chunk_0995_0999.hdf5.npz",
        ]
    },

    {
        "Rcut": 0.02,
        "files": [
            "test10/profile_chunk_0100_0104.hdf5.npz",
            "test10/profile_chunk_0200_0204.hdf5.npz",
            "test10/profile_chunk_0300_0304.hdf5.npz",
            "test10/profile_chunk_0400_0404.hdf5.npz",
            "test10/profile_chunk_0500_0504.hdf5.npz",
            "test10/profile_chunk_0600_0604.hdf5.npz",
            "test10/profile_chunk_0700_0704.hdf5.npz",
            "test10/profile_chunk_0800_0804.hdf5.npz",
            "test10/profile_chunk_0900_0904.hdf5.npz",
            "test10/profile_chunk_0995_0999.hdf5.npz",
        ]
    },

    {
        "Rcut": 0.01,
        "files": [
            "test11/profile_chunk_0100_0104.hdf5.npz",
            "test11/profile_chunk_0200_0204.hdf5.npz",
            "test11/profile_chunk_0300_0304.hdf5.npz",
            "test11/profile_chunk_0400_0404.hdf5.npz",
            "test11/profile_chunk_0500_0504.hdf5.npz",
            "test11/profile_chunk_0600_0604.hdf5.npz",
            "test11/profile_chunk_0700_0704.hdf5.npz",
            "test11/profile_chunk_0800_0804.hdf5.npz",
            "test11/profile_chunk_0900_0904.hdf5.npz",
            "test11/profile_chunk_0995_0999.hdf5.npz",
        ]
    },
]

make_figure(all_runs, labels)
