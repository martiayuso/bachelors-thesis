#!/usr/bin/env python3
"""
Plot trajectories of selected crossing particles.

For each selected particle, generate a figure with 4 subplots:

    [0] x vs y
    [1] x vs z
    [2] y vs z
    [3] r vs time

The galaxy center is indicated in the spatial projections.

Usage:
    python3 plot_particle_trajectories.py particles_timeseries.csv

Optional:
    python3 plot_particle_trajectories.py particles_timeseries.csv 5
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
DEFAULT_NUM_PARTICLES = 5

OUTPUT_DIR = "output/trajectories"

# Galaxy center
XC = 100.0
YC = 100.0
ZC = 100.0

# Font sizes
TITLE_SIZE = 20
SUBTITLE_SIZE = 16
LABEL_SIZE = 16
TICK_SIZE = 13
LEGEND_SIZE = 16


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------
def select_particle_ids(all_ids, n_select):

    all_ids = np.array(sorted(all_ids))

    if n_select >= len(all_ids):
        return all_ids

    idx = np.linspace(
        0,
        len(all_ids) - 1,
        n_select
    ).astype(int)

    return all_ids[idx]


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 plot_particle_trajectories.py particles_timeseries.csv")
        print("  python3 plot_particle_trajectories.py particles_timeseries.csv N")
        raise SystemExit(1)

    csv_file = sys.argv[1]

    if len(sys.argv) >= 3:
        n_particles = int(sys.argv[2])
    else:
        n_particles = DEFAULT_NUM_PARTICLES

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Reading:", csv_file)

    df = pd.read_csv(csv_file)

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------
    required_cols = [
        "id",
        "time",
        "x",
        "y",
        "z",
        "r"
    ]

    for c in required_cols:
        if c not in df.columns:
            raise RuntimeError(f"Missing required column: {c}")

    particle_ids = sorted(df["id"].unique())

    print(f"Found {len(particle_ids)} particles")

    selected_ids = select_particle_ids(
        particle_ids,
        n_particles
    )

    print("Selected particle IDs:")

    for pid in selected_ids:
        print("  ", int(pid))

    # ---------------------------------------------------------
    # Generate figures
    # ---------------------------------------------------------
    for pid in selected_ids:

        g = df[df["id"] == pid].copy()

        g = g.sort_values("time")

        # Remove NaNs
        mask = (
            np.isfinite(g["x"]) &
            np.isfinite(g["y"]) &
            np.isfinite(g["z"]) &
            np.isfinite(g["r"]) &
            np.isfinite(g["time"])
        )

        g = g[mask]

        if len(g) < 2:
            print(f"Skipping particle {pid}: insufficient data")
            continue

        fig, axs = plt.subplots(
            2,
            2,
            figsize=(13, 11)
        )

        # =====================================================
        # x vs y
        # =====================================================
        ax = axs[0, 0]

        ax.plot(
            g["x"],
            g["y"],
            linewidth=1.5
        )

        ax.scatter(
            XC,
            YC,
            marker="+",
            s=220,
            linewidths=2.5
        )

        ax.scatter(
            g["x"].iloc[0],
            g["y"].iloc[0],
            marker="o",
            s=70
        )

        ax.scatter(
            g["x"].iloc[-1],
            g["y"].iloc[-1],
            marker="x",
            s=70
        )

        ax.set_xlabel("x", fontsize=LABEL_SIZE)
        ax.set_ylabel("y", fontsize=LABEL_SIZE)

        ax.set_title(
            "x vs y",
            fontsize=SUBTITLE_SIZE
        )

        ax.tick_params(
            axis='both',
            labelsize=TICK_SIZE
        )

        ax.axis("equal")

        # =====================================================
        # x vs z
        # =====================================================
        ax = axs[0, 1]

        ax.plot(
            g["x"],
            g["z"],
            linewidth=1.5
        )

        center_marker = ax.scatter(
            XC,
            ZC,
            marker="+",
            s=220,
            linewidths=2.5
        )

        start_marker = ax.scatter(
            g["x"].iloc[0],
            g["z"].iloc[0],
            marker="o",
            s=70
        )

        end_marker = ax.scatter(
            g["x"].iloc[-1],
            g["z"].iloc[-1],
            marker="x",
            s=70
        )

        ax.set_xlabel("x", fontsize=LABEL_SIZE)
        ax.set_ylabel("z", fontsize=LABEL_SIZE)

        ax.set_title(
            "x vs z",
            fontsize=SUBTITLE_SIZE
        )

        ax.tick_params(
            axis='both',
            labelsize=TICK_SIZE
        )

        ax.axis("equal")

        # =====================================================
        # y vs z
        # =====================================================
        ax = axs[1, 0]

        ax.plot(
            g["y"],
            g["z"],
            linewidth=1.5
        )

        ax.scatter(
            YC,
            ZC,
            marker="+",
            s=220,
            linewidths=2.5
        )

        ax.scatter(
            g["y"].iloc[0],
            g["z"].iloc[0],
            marker="o",
            s=70
        )

        ax.scatter(
            g["y"].iloc[-1],
            g["z"].iloc[-1],
            marker="x",
            s=70
        )

        ax.set_xlabel("y", fontsize=LABEL_SIZE)
        ax.set_ylabel("z", fontsize=LABEL_SIZE)

        ax.set_title(
            "y vs z",
            fontsize=SUBTITLE_SIZE
        )

        ax.tick_params(
            axis='both',
            labelsize=TICK_SIZE
        )

        ax.axis("equal")

        # =====================================================
        # r vs time
        # =====================================================
        ax = axs[1, 1]

        ax.plot(
            g["time"],
            g["r"],
            linewidth=1.5
        )

        ax.scatter(
            g["time"].iloc[0],
            g["r"].iloc[0],
            marker="o",
            s=70
        )

        ax.scatter(
            g["time"].iloc[-1],
            g["r"].iloc[-1],
            marker="x",
            s=70
        )

        ax.set_xlabel(
            "time",
            fontsize=LABEL_SIZE
        )

        ax.set_ylabel(
            "r",
            fontsize=LABEL_SIZE
        )

        ax.set_title(
            "r vs time",
            fontsize=SUBTITLE_SIZE
        )

        ax.tick_params(
            axis='both',
            labelsize=TICK_SIZE
        )
        
        fig.legend(
            [
                start_marker,
                end_marker,
                center_marker
            ],
            [
                "Start",
                "End",
                "Galaxy center"
            ],
            fontsize=LEGEND_SIZE,
            frameon=False,          # removes legend box
            ncol=4,                 # horizontal legend
            loc="upper center",
            bbox_to_anchor=(0.68, 0.965)
        )

        # =====================================================
        # Finalize
        # =====================================================
        plt.suptitle(
            f"Particle {int(pid)} Trajectory",
            fontsize=TITLE_SIZE,
            x=0.28,
            y=0.95
        )

        # Reduced spacing between title and plots
        plt.tight_layout(
            rect=[0, 0, 1, 0.96]
        )

        out = os.path.join(
            OUTPUT_DIR,
            f"particle_{int(pid)}_trajectory.png"
        )

        plt.savefig(
            out,
            dpi=200
        )

        plt.close(fig)

        print("Wrote:", out)


if __name__ == "__main__":
    main()
