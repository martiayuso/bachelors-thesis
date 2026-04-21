import numpy as np
import matplotlib.pyplot as plt

def interp_density_at_radius(r, rho, r_target):
    r = np.asarray(r)
    rho = np.asarray(rho)

    idx = np.argsort(r)
    r = r[idx]
    rho = rho[idx]

    if np.all(r > 0) and np.all(rho > 0) and r_target > 0:
        log_r = np.log10(r)
        log_rho = np.log10(rho)
        log_r_target = np.log10(r_target)
        return 10 ** np.interp(log_r_target, log_r, log_rho)
    else:
        return np.interp(r_target, r, rho)


def compute_time_series(ic_file, sim_files_by_test, r_target=0.01):
    """
    Returns:
    --------
    ratio_matrix : shape (n_tests, n_times)
    rho0 : initial density at r_target
    """
    with np.load(ic_file) as ic_data:
        r_ic = ic_data["radius"]
        rho_ic = ic_data["density"]

    rho0 = interp_density_at_radius(r_ic, rho_ic, r_target)

    n_tests = len(sim_files_by_test)
    n_times = len(sim_files_by_test[0])

    ratio_matrix = np.zeros((n_tests, n_times))

    for i, test_files in enumerate(sim_files_by_test):
        for j, file in enumerate(test_files):
            with np.load(file) as data:
                r = data["radius"]
                rho = data["density"]

            rho_t = interp_density_at_radius(r, rho, r_target)
            ratio_matrix[i, j] = rho_t / rho0

    return ratio_matrix, rho0


def plot_time_evolution(ic_file, sim_files_by_test, test_labels, times,
                        r_target=0.01, output_file="density_time_evo.png"):

    params = {
        "text.usetex": True,
        "font.size": 12,
        "axes.labelsize": 16,
        "legend.fontsize": 12,
        "figure.figsize": (10, 6)
    }
    plt.rcParams.update(params)

    ratio_matrix, rho0 = compute_time_series(ic_file, sim_files_by_test, r_target)

    times = np.asarray(times, dtype=float)

    # Colors similar to your previous style
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    fig, ax = plt.subplots()

    for i, label in enumerate(test_labels):
        ax.plot(times,
                ratio_matrix[i],
                marker='o',
                linewidth=1.8,
                label=label,
                color=colors[i % len(colors)])

    # Reference line (no evolution)
    ax.axhline(1.0, color='black', linestyle='--', alpha=0.6)

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\rho(r=0.01\ \mathrm{kpc}) / \rho_{\rm init}$")

    ax.set_title(r"Density Evolution at $r = 0.01\ \mathrm{kpc}$")

    ax.grid(alpha=0.2)
    ax.legend(title="Parameter")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()


if __name__ == "__main__":
    ic = "profile_snapshot_0000.hdf5.npz"

    # naming convention: {parameter}_{time}.npz
    test_labels = ["0.0001", "0.01"]
    # times = [2, 4, 6, 8, 10, 20, 40, 60, 80, 100]
    times = [2, 4, 6, 8, 10]

    sim_files_by_test = [
        [f"{label}_{t}.npz" for t in times]
        for label in test_labels
    ]

    plot_time_evolution(ic, sim_files_by_test, test_labels, times, r_target=0.01)
