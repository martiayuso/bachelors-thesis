import numpy as np
import matplotlib.pyplot as plt

def interp_density_at_radius(r, rho, r_target):
    r = np.asarray(r)
    rho = np.asarray(rho)

    # Sort by radius in case input is unordered
    idx = np.argsort(r)
    r = r[idx]
    rho = rho[idx]

    # Use log-log interpolation if values are positive
    if np.all(r > 0) and np.all(rho > 0) and r_target > 0:
        log_r = np.log10(r)
        log_rho = np.log10(rho)
        log_r_target = np.log10(r_target)
        return 10 ** np.interp(log_r_target, log_r, log_rho)
    else:
        return np.interp(r_target, r, rho)


def compute_heatmap_matrix(ic_file, sim_files_by_test, r_target=0.01):
    """
    Parameters
    ----------
    ic_file : str
        Initial condition .npz file with keys: 'radius', 'density'
    sim_files_by_test : list[list[str]]
        Outer list = tests/parameter sets
        Inner list = files for that test ordered by time
    r_target : float
        Radius at which to sample density

    Returns
    -------
    ratio_matrix : ndarray, shape (n_tests, n_times)
        rho(r_target, t) / rho_init(r_target)
    rho0 : float
        Initial-condition density at r_target
    """
    with np.load(ic_file) as ic_data:
        r_ic = ic_data["radius"]
        rho_ic = ic_data["density"]

    rho0 = interp_density_at_radius(r_ic, rho_ic, r_target)

    n_tests = len(sim_files_by_test)
    n_times = len(sim_files_by_test[0])

    ratio_matrix = np.full((n_tests, n_times), np.nan)

    for i, test_files in enumerate(sim_files_by_test):
        if len(test_files) != n_times:
            raise ValueError("All tests must have the same number of time snapshots.")
        for j, file in enumerate(test_files):
            with np.load(file) as data:
                r = data["radius"]
                rho = data["density"]
            rho_t = interp_density_at_radius(r, rho, r_target)
            ratio_matrix[i, j] = rho_t / rho0

    return ratio_matrix, rho0


def plot_heatmap(ic_file, sim_files_by_test, test_labels, times, r_target=0.01,
                 output_file="density_heatmap.png"):
    params = {
        "text.usetex": True,
        "font.size": 12,
        "axes.labelsize": 16,
        "legend.fontsize": 12,
        "figure.figsize": (10, 6)
    }
    plt.rcParams.update(params)

    ratio_matrix, rho0 = compute_heatmap_matrix(ic_file, sim_files_by_test, r_target=r_target)

    times = np.asarray(times, dtype=float)
    test_labels = list(test_labels)

    if len(times) != ratio_matrix.shape[1]:
        raise ValueError("Length of 'times' must match number of snapshots per test.")
    if len(test_labels) != ratio_matrix.shape[0]:
        raise ValueError("Length of 'test_labels' must match number of tests.")

    heat = ratio_matrix

    fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * len(test_labels) + 2)))

    im = ax.imshow(
        heat,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
        extent=[times[0], times[-1], -0.5, len(test_labels) - 0.5],
    )

    # Y axis: test labels
    ax.set_yticks(np.arange(len(test_labels)))
    ax.set_yticklabels(test_labels)

    # X axis: time
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Test")
    ax.set_title(rf"Density at $r = {r_target}\ \mathrm{{kpc}}$ relative to initial condition")

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"$\rho / \rho_{\rm init}$")

    ax.grid(False)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()


if __name__ == "__main__":
    ic = "profile_snapshot_0000.hdf5.npz"
    
    # naming convention: {parameter}_{time}.npz
    test_labels = ["0.0001", "0.001", "0.01"]
    times = [20, 40, 60, 80, 100]
    
    sim_files_by_test = [
    [f"{label}_{t}.npz" for t in times]
    for label in test_labels
    ]
    
    plot_heatmap(ic, sim_files_by_test, test_labels, times, r_target=0.01)
