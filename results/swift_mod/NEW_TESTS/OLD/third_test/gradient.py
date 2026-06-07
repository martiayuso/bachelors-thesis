import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def nfw_profile(r, rho0, rs, alpha=1.0, beta=3.0):
    """
    Generalized two-slope profile.

    For alpha=1 and beta=3 this becomes the NFW profile.
    """
    x = r / rs
    return rho0 / (x**alpha * (1.0 + x)**(beta - alpha))


def plot(sim_files, labels):

    params = {
        "text.usetex": True,
        "font.size": 12,
        "axes.labelsize": 16,
        "legend.fontsize": 12,
        "figure.figsize": (10, 8)
    }
    plt.rcParams.update(params)

    fig, ax = plt.subplots()

    colors = cm.rainbow(np.linspace(0, 1, len(sim_files) + 1))

    # ------------------------------------------------------------------
    # Analytical NFW profile parameters
    # ------------------------------------------------------------------
    rho0 = 0.0027658297046461644 * 1e10
    rs = 0.828177997155062
    alpha = 1.0
    beta = 3.0

    # Radius grid
    r_nfw = np.logspace(-3, np.log10(50), 1000)

    # Analytical density profile
    rho_nfw = nfw_profile(r_nfw, rho0, rs, alpha, beta)

    # Optional exponential cutoff
    r_cutoff = 50.0
    power_cutoff = 3.0

    rho_nfw *= np.exp(-(r_nfw / r_cutoff)**power_cutoff)

    ax.plot(
        r_nfw,
        rho_nfw,
        label="Initial condition",
        color=colors[0],
        linewidth=2.5
    )

    # ------------------------------------------------------------------
    # Simulation snapshots
    # ------------------------------------------------------------------
    for i, file in enumerate(sim_files):

        data = np.load(file)

        r = data['radius']
        rho = data['density']

        rho = 0.2 * rho

        ax.plot(
            r,
            rho,
            label=labels[i],
            color=colors[i + 1],
            linewidth=2.5
        )

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------
    ax.set_xscale('log')
    ax.set_yscale('log')
    
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

    ax.set_xlim(1e-2, 10)
    ax.set_ylim(1e4, 1e10)

    ax.set_xlabel(r'$r\ [\rm{kpc}]$')
    ax.set_ylabel(r'$\rho\ [M_\odot/\rm{kpc}^3]$')

    ax.grid(alpha=0.2)

    ax.set_title(
        r"Density Profile Evolution for $\epsilon = 0.01\ \mathrm{kpc}$"
    )

    ax.legend()

    plt.tight_layout()
    plt.savefig("density_evo_100.png", dpi=300)
    plt.show()


if __name__ == "__main__":

    sims = [
            "profile_snap_01.hdf5.npz",
            "profile_snap_02.hdf5.npz",
            "profile_snap_03.hdf5.npz",
            "profile_snap_04.hdf5.npz",
            "profile_snap_05.hdf5.npz"
    ]

    labels = [
        "rcut 1", "rcut 2", "rcut 3", "rcut 4", "rcut 5"
    ]

    plot(sims, labels)
