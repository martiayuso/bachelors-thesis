import numpy as np
import matplotlib.pyplot as plt


def nfw_profile(r, rho0, rs, alpha=1.0, beta=3.0):
    """
    Generalized two-slope profile.
    For alpha=1 and beta=3 this becomes NFW.
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

    # Create figure with two subplots (ratio 3:1)
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}
    )

    # ------------------------------------------------------------------
    # Analytical IC
    # ------------------------------------------------------------------
    rho0 = 0.0027658297046461644 * 1e10
    rs = 0.828177997155062
    alpha = 1.0
    beta = 3.0

    r_ic = np.logspace(-3, np.log10(50), 1000)

    rho_ic = nfw_profile(r_ic, rho0, rs, alpha, beta)

    # Optional cutoff
    r_cutoff = 50.0
    power_cutoff = 3.0

    rho_ic *= np.exp(-(r_ic / r_cutoff)**power_cutoff)

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

    # ------------------------------------------------------------------
    # Simulation snapshots
    # ------------------------------------------------------------------
    colors = [
        'tab:orange',
        'tab:green',
        'tab:red',
        'tab:purple',
        'tab:brown'
    ]

    for i, file in enumerate(sim_files):

        data = np.load(file)

        r = data['radius']
        rho = data['density']

        rho = 0.2 * rho

        # Density panel
        ax1.plot(
            r,
            rho,
            label=labels[i],
            color=colors[i % len(colors)]
        )

        # Ratio panel
        rho_interp = np.interp(r_ic, r, rho)

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = rho_interp / rho_ic

        ax2.plot(
            r_ic,
            ratio,
            color=colors[i % len(colors)]
        )

    # ------------------------------------------------------------------
    # Formatting top panel
    # ------------------------------------------------------------------
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    ax1.set_ylabel(r'$\rho\ [M_\odot/\rm{kpc}^3]$')

    ax1.set_xlim(1e-2, 10)
    ax1.set_ylim(1e4, 1e10)

    ax1.legend()
    ax1.grid(alpha=0.2)

    ax1.set_title(
        r"Density Profile Evolution for $\epsilon = 0.01\ \mathrm{kpc}$ and $R_{\rm cut}=5.0\ \mathrm{kpc}$"
    )

    # ------------------------------------------------------------------
    # Formatting bottom panel
    # ------------------------------------------------------------------
    ax2.set_ylabel(r'$\rho / \rho_{\rm init}$')
    ax2.set_xlabel(r'$r\ [\rm{kpc}]$')

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
    v_line_pos = 2.8 * eps_value

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

    fig.subplots_adjust(
        left=0.11,
        right=0.98,
        top=0.93,
        bottom=0.10,
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

    sims = [
        "profile_chunk_0200_0204.hdf5.npz",
        "profile_chunk_0400_0404.hdf5.npz",
        "profile_chunk_0600_0604.hdf5.npz",
        "profile_chunk_0800_0804.hdf5.npz",
        "profile_chunk_0995_0999.hdf5.npz"
    ]

    labels = [
        "20 Gyr",
        "40 Gyr",
        "60 Gyr",
        "80 Gyr",
        "100 Gyr"
    ]

    plot(sims, labels)
