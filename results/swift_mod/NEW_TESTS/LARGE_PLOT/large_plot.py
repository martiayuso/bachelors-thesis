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

    # Apply cutoff
    power_cutoff = 3.0

    rho_nfw *= np.exp(-(r_nfw / r_cutoff)**power_cutoff)

    # Plot analytical profile
    ax.plot(
        r_nfw,
        rho_nfw,
        color=colors[0],
        linewidth=2.5,
        label="IC"
    )

    # -----------------------------------------------------------------
    # Simulation snapshots
    # -----------------------------------------------------------------
    for i, file in enumerate(sim_files):

        data = np.load(file)

        r = data["radius"]
        rho = data["density"]

        rho = 1 * rho

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

    ax.set_xlim(1e-2, 1)
    ax.set_ylim(1e7, 5e9)

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
    fig = plt.figure(figsize=(16, 16))

    # Custom grid
    gs = fig.add_gridspec(
        4,
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
    # First 12 panels
    # -------------------------------------------------------------
    for row in range(3):
        for col in range(4):

            ax = fig.add_subplot(gs[row, col])

            # FORCE square panels
            ax.set_box_aspect(1)

            axes.append(ax)

    # -------------------------------------------------------------
    # Last row centered
    # -------------------------------------------------------------
    ax13 = fig.add_subplot(gs[3, 1])
    ax14 = fig.add_subplot(gs[3, 2])

    ax13.set_box_aspect(1)
    ax14.set_box_aspect(1)

    axes.extend([ax13, ax14])

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
        # X labels only on bottom visible panels
        # ---------------------------------------------------------
        if i >= 8:
            ax.set_xlabel(r"$r\ [{\rm kpc}]$")
        else:
            ax.set_xticklabels([])
            
        # Remove x-axis labels from middle panels in penultimate row
        for i in [9, 10]:
            axes[i].set_xlabel("")

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
    # Legend in empty whitespace
    # -------------------------------------------------------------
    handles, legend_labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        legend_labels,
        loc="center",
        bbox_to_anchor=(0.86, 0.16),
        frameon=False,
        ncol=2,
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
        "1 Gyr",
        "2 Gyr",
        "3 Gyr",
        "4 Gyr",
        "5 Gyr",
        "6 Gyr",
        "7 Gyr",
        "8 Gyr",
        "9 Gyr",
        "10 Gyr"
    ]

# -------------------------------------------------------------
# Each entry = one subplot
# -------------------------------------------------------------
all_runs = [

    {
        "Rcut": 1.0,
        "files": [
            "Rcut1/profile_snapshot_0010.hdf5.npz",
            "Rcut1/profile_snapshot_0020.hdf5.npz",
            "Rcut1/profile_snapshot_0030.hdf5.npz",
            "Rcut1/profile_snapshot_0040.hdf5.npz",
            "Rcut1/profile_snapshot_0050.hdf5.npz",
            "Rcut1/profile_snapshot_0060.hdf5.npz",
            "Rcut1/profile_snapshot_0070.hdf5.npz",
            "Rcut1/profile_snapshot_0080.hdf5.npz",
            "Rcut1/profile_snapshot_0090.hdf5.npz",
            "Rcut1/profile_snapshot_0100.hdf5.npz",
        ]
    },

    {
        "Rcut": 2.0,
        "files": [
            "Rcut2/profile_snapshot_0010.hdf5.npz",
            "Rcut2/profile_snapshot_0020.hdf5.npz",
            "Rcut2/profile_snapshot_0030.hdf5.npz",
            "Rcut2/profile_snapshot_0040.hdf5.npz",
            "Rcut2/profile_snapshot_0050.hdf5.npz",
            "Rcut2/profile_snapshot_0060.hdf5.npz",
            "Rcut2/profile_snapshot_0070.hdf5.npz",
            "Rcut2/profile_snapshot_0080.hdf5.npz",
            "Rcut2/profile_snapshot_0090.hdf5.npz",
            "Rcut2/profile_snapshot_0100.hdf5.npz",
        ]
    },

    {
        "Rcut": 3.0,
        "files": [
            "Rcut3/profile_snapshot_0010.hdf5.npz",
            "Rcut3/profile_snapshot_0020.hdf5.npz",
            "Rcut3/profile_snapshot_0030.hdf5.npz",
            "Rcut3/profile_snapshot_0040.hdf5.npz",
            "Rcut3/profile_snapshot_0050.hdf5.npz",
            "Rcut3/profile_snapshot_0060.hdf5.npz",
            "Rcut3/profile_snapshot_0070.hdf5.npz",
            "Rcut3/profile_snapshot_0080.hdf5.npz",
            "Rcut3/profile_snapshot_0090.hdf5.npz",
            "Rcut3/profile_snapshot_0100.hdf5.npz",
        ]
    },

    {
        "Rcut": 4.0,
        "files": [
            "Rcut4/profile_snapshot_0010.hdf5.npz",
            "Rcut4/profile_snapshot_0020.hdf5.npz",
            "Rcut4/profile_snapshot_0030.hdf5.npz",
            "Rcut4/profile_snapshot_0040.hdf5.npz",
            "Rcut4/profile_snapshot_0050.hdf5.npz",
            "Rcut4/profile_snapshot_0060.hdf5.npz",
            "Rcut4/profile_snapshot_0070.hdf5.npz",
            "Rcut4/profile_snapshot_0080.hdf5.npz",
            "Rcut4/profile_snapshot_0090.hdf5.npz",
            "Rcut4/profile_snapshot_0100.hdf5.npz",
        ]
    },

    {
        "Rcut": 5.0,
        "files": [
            "Rcut5/profile_snapshot_0010.hdf5.npz",
            "Rcut5/profile_snapshot_0020.hdf5.npz",
            "Rcut5/profile_snapshot_0030.hdf5.npz",
            "Rcut5/profile_snapshot_0040.hdf5.npz",
            "Rcut5/profile_snapshot_0050.hdf5.npz",
            "Rcut5/profile_snapshot_0060.hdf5.npz",
            "Rcut5/profile_snapshot_0070.hdf5.npz",
            "Rcut5/profile_snapshot_0080.hdf5.npz",
            "Rcut5/profile_snapshot_0090.hdf5.npz",
            "Rcut5/profile_snapshot_0100.hdf5.npz",
        ]
    },

    {
        "Rcut": 6.0,
        "files": [
            "Rcut6/profile_snapshot_0010.hdf5.npz",
            "Rcut6/profile_snapshot_0020.hdf5.npz",
            "Rcut6/profile_snapshot_0030.hdf5.npz",
            "Rcut6/profile_snapshot_0040.hdf5.npz",
            "Rcut6/profile_snapshot_0050.hdf5.npz",
            "Rcut6/profile_snapshot_0060.hdf5.npz",
            "Rcut6/profile_snapshot_0070.hdf5.npz",
            "Rcut6/profile_snapshot_0080.hdf5.npz",
            "Rcut6/profile_snapshot_0090.hdf5.npz",
            "Rcut6/profile_snapshot_0100.hdf5.npz",
        ]
    },

    {
        "Rcut": 7.0,
        "files": [
            "Rcut7/profile_snapshot_0010.hdf5.npz",
            "Rcut7/profile_snapshot_0020.hdf5.npz",
            "Rcut7/profile_snapshot_0030.hdf5.npz",
            "Rcut7/profile_snapshot_0040.hdf5.npz",
            "Rcut7/profile_snapshot_0050.hdf5.npz",
            "Rcut7/profile_snapshot_0060.hdf5.npz",
            "Rcut7/profile_snapshot_0070.hdf5.npz",
            "Rcut7/profile_snapshot_0080.hdf5.npz",
            "Rcut7/profile_snapshot_0090.hdf5.npz",
            "Rcut7/profile_snapshot_0100.hdf5.npz",
        ]
    },

    {
        "Rcut": 8.0,
        "files": [
            "Rcut8/profile_snapshot_0010.hdf5.npz",
            "Rcut8/profile_snapshot_0020.hdf5.npz",
            "Rcut8/profile_snapshot_0030.hdf5.npz",
            "Rcut8/profile_snapshot_0040.hdf5.npz",
            "Rcut8/profile_snapshot_0050.hdf5.npz",
            "Rcut8/profile_snapshot_0060.hdf5.npz",
            "Rcut8/profile_snapshot_0070.hdf5.npz",
            "Rcut8/profile_snapshot_0080.hdf5.npz",
            "Rcut8/profile_snapshot_0090.hdf5.npz",
            "Rcut8/profile_snapshot_0100.hdf5.npz",
        ]
    },

    {
        "Rcut": 9.0,
        "files": [
            "Rcut9/profile_snapshot_0010.hdf5.npz",
            "Rcut9/profile_snapshot_0020.hdf5.npz",
            "Rcut9/profile_snapshot_0030.hdf5.npz",
            "Rcut9/profile_snapshot_0040.hdf5.npz",
            "Rcut9/profile_snapshot_0050.hdf5.npz",
            "Rcut9/profile_snapshot_0060.hdf5.npz",
            "Rcut9/profile_snapshot_0070.hdf5.npz",
            "Rcut9/profile_snapshot_0080.hdf5.npz",
            "Rcut9/profile_snapshot_0090.hdf5.npz",
            "Rcut9/profile_snapshot_0100.hdf5.npz",
        ]
    },

    {
        "Rcut": 10.0,
        "files": [
            "Rcut10/profile_snapshot_0010.hdf5.npz",
            "Rcut10/profile_snapshot_0020.hdf5.npz",
            "Rcut10/profile_snapshot_0030.hdf5.npz",
            "Rcut10/profile_snapshot_0040.hdf5.npz",
            "Rcut10/profile_snapshot_0050.hdf5.npz",
            "Rcut10/profile_snapshot_0060.hdf5.npz",
            "Rcut10/profile_snapshot_0070.hdf5.npz",
            "Rcut10/profile_snapshot_0080.hdf5.npz",
            "Rcut10/profile_snapshot_0090.hdf5.npz",
            "Rcut10/profile_snapshot_0100.hdf5.npz",
        ]
    },

    {
        "Rcut": 15.0,
        "files": [
            "Rcut15/profile_snapshot_0010.hdf5.npz",
            "Rcut15/profile_snapshot_0020.hdf5.npz",
            "Rcut15/profile_snapshot_0030.hdf5.npz",
            "Rcut15/profile_snapshot_0040.hdf5.npz",
            "Rcut15/profile_snapshot_0050.hdf5.npz",
            "Rcut15/profile_snapshot_0060.hdf5.npz",
            "Rcut15/profile_snapshot_0070.hdf5.npz",
            "Rcut15/profile_snapshot_0080.hdf5.npz",
            "Rcut15/profile_snapshot_0090.hdf5.npz",
            "Rcut15/profile_snapshot_0100.hdf5.npz",
        ]
    },

    {
        "Rcut": 20.0,
        "files": [
            "Rcut20/profile_snapshot_0010.hdf5.npz",
            "Rcut20/profile_snapshot_0020.hdf5.npz",
            "Rcut20/profile_snapshot_0030.hdf5.npz",
            "Rcut20/profile_snapshot_0040.hdf5.npz",
            "Rcut20/profile_snapshot_0050.hdf5.npz",
            "Rcut20/profile_snapshot_0060.hdf5.npz",
            "Rcut20/profile_snapshot_0070.hdf5.npz",
            "Rcut20/profile_snapshot_0080.hdf5.npz",
            "Rcut20/profile_snapshot_0090.hdf5.npz",
            "Rcut20/profile_snapshot_0100.hdf5.npz",
        ]
    },

    {
        "Rcut": 25.0,
        "files": [
            "Rcut25/profile_snapshot_0010.hdf5.npz",
            "Rcut25/profile_snapshot_0020.hdf5.npz",
            "Rcut25/profile_snapshot_0030.hdf5.npz",
            "Rcut25/profile_snapshot_0040.hdf5.npz",
            "Rcut25/profile_snapshot_0050.hdf5.npz",
            "Rcut25/profile_snapshot_0060.hdf5.npz",
            "Rcut25/profile_snapshot_0070.hdf5.npz",
            "Rcut25/profile_snapshot_0080.hdf5.npz",
            "Rcut25/profile_snapshot_0090.hdf5.npz",
            "Rcut25/profile_snapshot_0100.hdf5.npz",
        ]
    },

    {
        "Rcut": 50.0,
        "files": [
            "Rcut50/profile_snapshot_0010.hdf5.npz",
            "Rcut50/profile_snapshot_0020.hdf5.npz",
            "Rcut50/profile_snapshot_0030.hdf5.npz",
            "Rcut50/profile_snapshot_0040.hdf5.npz",
            "Rcut50/profile_snapshot_0050.hdf5.npz",
            "Rcut50/profile_snapshot_0060.hdf5.npz",
            "Rcut50/profile_snapshot_0070.hdf5.npz",
            "Rcut50/profile_snapshot_0080.hdf5.npz",
            "Rcut50/profile_snapshot_0090.hdf5.npz",
            "Rcut50/profile_snapshot_0100.hdf5.npz",
        ]
    },
]

make_figure(all_runs, labels)
