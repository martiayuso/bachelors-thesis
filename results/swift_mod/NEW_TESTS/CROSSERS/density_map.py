#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Usage: tools/plot_particle_timeseries.py particles_timeseries.csv")
        raise SystemExit(1)

    fn = sys.argv[1]
    df = pd.read_csv(fn)

    os.makedirs("output", exist_ok=True)

    # Sort once (important for consistent time range)
    df = df.sort_values("time")

    # Global time range (shared x-axis)
    tmin, tmax = df["time"].min(), df["time"].max()

    # Helper for consistent heatmaps
    def plot_density(ax, x, y, xlabel, ylabel, title, bins=150):
        h = ax.hist2d(
            x,
            y,
            bins=bins,
            range=[[tmin, tmax], [np.nanmin(y), np.nanmax(y)]],
            cmap="magma"
        )
        plt.colorbar(h[3], ax=ax, label="counts")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # r vs time density
    plot_density(
        axs[0],
        df["time"], df["r"],
        "time", "r",
        "Density: r vs time"
    )

    # |v| vs time density
    plot_density(
        axs[1],
        df["time"], df["v_mag"],
        "time", "|v|",
        "Density: |v| vs time"
    )

    # potential vs time density
    plot_density(
        axs[2],
        df["time"], df["potential"],
        "time", "potential",
        "Density: potential vs time"
    )

    plt.suptitle("All particles (density view)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = "output/all_particles_density.png"
    plt.savefig(out, dpi=200)
    print("Wrote", out)


if __name__ == "__main__":
    main()
