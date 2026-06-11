import numpy as np
import matplotlib.pyplot as plt


def nfw_profile(r, rho0, rs, alpha=1.0, beta=3.0):
    """
    Generalized two-slope profile.
    For alpha=1 and beta=3 this becomes NFW.
    """
    x = r / rs
    return rho0 / (x**alpha * (1.0 + x)**(beta - alpha))


def get_initial_profile():

    rho0 = 0.0027658297046461644 * 1e10
    rs = 0.828177997155062
    alpha = 1.0
    beta = 3.0

    r_ic = np.logspace(-3, np.log10(50), 2000)

    rho_ic = nfw_profile(
        r_ic,
        rho0,
        rs,
        alpha,
        beta
    )

    # Optional cutoff
    r_cutoff = 50.0
    power_cutoff = 3.0

    rho_ic *= np.exp(-(r_ic / r_cutoff)**power_cutoff)

    return r_ic, rho_ic


def compute_ratio(filename, r_ic, rho_ic):

    data = np.load(filename)

    r = data["radius"]
    rho = 0.2 * data["density"]

    rho_interp = np.interp(r_ic, r, rho)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = rho_interp / rho_ic

    return ratio


def plot_ratio_grid(simulation_groups):

    params = {
        "text.usetex": True,
        "font.size": 18,
        "axes.labelsize": 18,
        "legend.fontsize": 18,
        "figure.figsize": (12, 10),
    }

    plt.rcParams.update(params)

    r_ic, rho_ic = get_initial_profile()

    fig, axes = plt.subplots(
        3,
        2,
        sharex=True,
        sharey=True,
        figsize=(12, 8)
    )

    axes = axes.flatten()
    
    plot_axes = axes[:5]
    legend_ax = axes[5]

    colors = [
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
    ]

    eps_value = 0.01
    softening_scale = 2.8 * eps_value

    for ax, group in zip(plot_axes, simulation_groups):

        for i, filename in enumerate(group["files"]):

            ratio = compute_ratio(
                filename,
                r_ic,
                rho_ic
            )

            ax.plot(
                r_ic,
                ratio,
                color=colors[i],
                lw=1.5,
                label=group["labels"][i]
            )

        ax.axhline(
            1.0,
            color="black",
            ls="--",
            lw=1,
            alpha=0.5
        )

        ax.axvline(
            softening_scale,
            color="gray",
            ls="--",
            lw=1,
            alpha=0.6
        )

        ax.set_xscale("log")
        ax.set_xlim(1e-2, 10)
        ax.set_ylim(0.1, 1.5)

        ax.set_title(group["title"])

        ax.grid(alpha=0.2)

        ax.yaxis.set_ticks_position("both")
        ax.xaxis.set_ticks_position("both")

        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            length=6,
            width=1
        )

        ax.tick_params(
            axis="both",
            which="minor",
            direction="in",
            length=3,
            width=0.8
        )
        
    handles, labels = plot_axes[0].get_legend_handles_labels()

    legend_ax.axis("off")

    legend_ax.legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        fontsize=18
    )

    # Shared labels
    fig.supxlabel(
        r"$r\,[\mathrm{kpc}]$",
        fontsize=18
    )

    fig.supylabel(
        r"$\rho/\rho_{\rm init}$",
        fontsize=18
    )

    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.10,
        top=0.95,
        wspace=0.12,
        hspace=0.32
    )

    plt.savefig(
        "density_ratio_grid.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":

    snapshot_labels = [
        "20 Gyr",
        "40 Gyr",
        "60 Gyr",
        "80 Gyr",
        "100 Gyr"
    ]

    simulation_groups = [

        {
            "title": r"$R_{\rm cut}=5\ \mathrm{kpc}$",
            "labels": snapshot_labels,
            "files": [
                "5/profile_chunk_0200_0204.hdf5.npz",
                "5/profile_chunk_0400_0404.hdf5.npz",
                "5/profile_chunk_0600_0604.hdf5.npz",
                "5/profile_chunk_0800_0804.hdf5.npz",
                "5/profile_chunk_0995_0999.hdf5.npz",
            ]
        },

        {
            "title": r"$R_{\rm cut}=0.5\ \mathrm{kpc}$",
            "labels": snapshot_labels,
            "files": [
                "0.5/profile_chunk_0200_0204.hdf5.npz",
                "0.5/profile_chunk_0400_0404.hdf5.npz",
                "0.5/profile_chunk_0600_0604.hdf5.npz",
                "0.5/profile_chunk_0800_0804.hdf5.npz",
                "0.5/profile_chunk_0995_0999.hdf5.npz",
            ]
        },

        {
            "title": r"$R_{\rm cut}=0.05\ \mathrm{kpc}$",
            "labels": snapshot_labels,
            "files": [
                "0.05/profile_chunk_0200_0204.hdf5.npz",
                "0.05/profile_chunk_0400_0404.hdf5.npz",
                "0.05/profile_chunk_0600_0604.hdf5.npz",
                "0.05/profile_chunk_0800_0804.hdf5.npz",
                "0.05/profile_chunk_0995_0999.hdf5.npz",
            ]
        },

        {
            "title": r"$R_{\rm cut}=0.03\ \mathrm{kpc}$",
            "labels": snapshot_labels,
            "files": [
                "0.03/profile_chunk_0200_0204.hdf5.npz",
                "0.03/profile_chunk_0400_0404.hdf5.npz",
                "0.03/profile_chunk_0600_0604.hdf5.npz",
                "0.03/profile_chunk_0800_0804.hdf5.npz",
                "0.03/profile_chunk_0995_0999.hdf5.npz",
            ]
        },

        {
            "title": r"$R_{\rm cut}=0.02\ \mathrm{kpc}$",
            "labels": snapshot_labels,
            "files": [
                "0.02/profile_chunk_0200_0204.hdf5.npz",
                "0.02/profile_chunk_0400_0404.hdf5.npz",
                "0.02/profile_chunk_0600_0604.hdf5.npz",
                "0.02/profile_chunk_0800_0804.hdf5.npz",
                "0.02/profile_chunk_0995_0999.hdf5.npz",
            ]
        },
    ]

    plot_ratio_grid(simulation_groups)
