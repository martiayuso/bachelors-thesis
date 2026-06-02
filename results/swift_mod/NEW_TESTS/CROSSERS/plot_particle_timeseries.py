#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


MAX_POINTS_PER_PARTICLE = 300   # subsampling cap
BASE_ALPHA = 0.6                # will be reduced automatically if many particles


def subsample(x, y, max_points):
    """Uniformly subsample (x, y) to at most max_points."""
    n = len(x)
    if n <= max_points:
        return x, y
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return x.iloc[idx], y.iloc[idx]


def main():
    if len(sys.argv) < 2:
        print("Usage: tools/plot_particle_timeseries.py particles_timeseries.csv")
        raise SystemExit(1)

    fn = sys.argv[1]
    df = pd.read_csv(fn)

    os.makedirs("output", exist_ok=True)

    particles = list(df.groupby("id"))
    n_particles = len(particles)

    # Reduce alpha automatically when many particles
    alpha = min(BASE_ALPHA, 5.0 / max(n_particles, 1))

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    for pid, g in particles:
        g = g.sort_values("time")

        t_r, r = subsample(g.time, g.r, MAX_POINTS_PER_PARTICLE)
        t_v, v = subsample(g.time, g.v_mag, MAX_POINTS_PER_PARTICLE)
        t_p, p = subsample(g.time, g.potential, MAX_POINTS_PER_PARTICLE)

        axs[0].plot(t_r, r, alpha=alpha)
        axs[1].plot(t_v, v, alpha=alpha)
        axs[2].plot(t_p, p, alpha=alpha)

    axs[0].set_ylabel("r")
    axs[0].axhline(color="gray", linestyle="--", linewidth=1)

    axs[1].set_ylabel("|v|")
    axs[2].set_ylabel("potential")
    axs[2].set_xlabel("time")

    plt.suptitle(f"All particles (n={n_particles})")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = "output/all_particles_summary.png"
    plt.savefig(out, dpi=200)
    print("Wrote", out)


if __name__ == "__main__":
    main()
