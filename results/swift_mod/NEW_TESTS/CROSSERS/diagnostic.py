#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter



R_CUT = 1.0
BINS = 800


# -----------------------------
# Safe histogram helper
# -----------------------------
def safe_hist2d(ax, x, y, title, xlabel, ylabel):
    x = np.asarray(x)
    y = np.asarray(y)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        ax.set_title(title + " (no data)")
        return

    # Compute histogram manually
    H, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=BINS
    )

    # Smooth the histogram
    H = gaussian_filter(H, sigma=2.5)
    
    H = np.clip(H, 1, None)

    # Render smoothly
    im = ax.imshow(
        H.T,
        origin='lower',
        aspect='auto',
        extent=[
            xedges[0], xedges[-1],
            yedges[0], yedges[-1]
        ],
        cmap='magma',
        norm=LogNorm(),
        interpolation='bilinear'
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.colorbar(im, ax=ax, label='counts')


# -----------------------------
# Main
# -----------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: diagnostic.py particles_timeseries.csv")
        raise SystemExit(1)

    fn = sys.argv[1]
    df = pd.read_csv(fn)

    os.makedirs("output", exist_ok=True)

    print(f"Loaded {df['id'].nunique()} particles, {len(df)} rows")

    # ============================================================
    # PANEL 1: TIME-BASED EVOLUTION (ensemble view)
    # ============================================================
    fig1, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    safe_hist2d(
        axs[0],
        df["time"],
        df["r"],
        "Radius vs time (all crossers)",
        "time",
        "r"
    )

    safe_hist2d(
        axs[1],
        df["time"],
        df["v_mag"],
        "|v| vs time (all crossers)",
        "time",
        "|v|"
    )

    if "potential" in df.columns:
        safe_hist2d(
            axs[2],
            df["time"],
            df["potential"],
            "Potential vs time",
            "time",
            "Φ"
        )
    else:
        axs[2].set_title("Potential missing")

    plt.suptitle("time evolution")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig1.savefig("output/time_evolution.png", dpi=200)


    # ============================================================
    # PANEL 2: Rcut-centered physics (MOST IMPORTANT)
    # ============================================================
    df["dr"] = df["r"] - R_CUT

    fig2, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    safe_hist2d(
        axs[0],
        df["dr"],
        df["r"],
        "Radius structure around Rcut",
        "r - Rcut",
        "r"
    )

    safe_hist2d(
        axs[1],
        df["dr"],
        df["v_mag"],
        "Velocity vs distance to Rcut",
        "r - Rcut",
        "|v|"
    )

    if "potential" in df.columns:
        safe_hist2d(
            axs[2],
            df["dr"],
            df["potential"],
            "Potential vs distance to Rcut",
            "r - Rcut",
            "Φ"
        )
    else:
        axs[2].set_title("Potential missing")

    plt.suptitle("SWIFT hybrid diagnostics: Rcut boundary structure")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig2.savefig("output/rcut_structure.png", dpi=200)


    # ============================================================
    # PANEL 3: PHASE SPACE (physics consistency check)
    # ============================================================
    fig3, ax3 = plt.subplots(1, 1, figsize=(7, 6))

    safe_hist2d(
        ax3,
        df["r"],
        df["v_mag"],
        "|v| vs r phase space",
        "r",
        "|v|"
    )

    ax3.axvline(R_CUT, color="cyan", linestyle="--", linewidth=1)

    fig3.tight_layout()
    fig3.savefig("output/phase_space.png", dpi=200)


    # ============================================================
    # PANEL 4: INSIDE vs OUTSIDE velocity sanity check
    # ============================================================
    inside = df[df["r"] < R_CUT]["v_mag"]
    outside = df[df["r"] >= R_CUT]["v_mag"]

    fig4, ax4 = plt.subplots(1, 1, figsize=(7, 5))

    inside = inside[np.isfinite(inside)]
    outside = outside[np.isfinite(outside)]

    if len(inside) > 10:
        ax4.hist(inside, bins=80, alpha=0.5, label="inside r<Rcut")
    if len(outside) > 10:
        ax4.hist(outside, bins=80, alpha=0.5, label="outside r>Rcut")

    ax4.set_title("Velocity distribution: inside vs outside Rcut")
    ax4.set_xlabel("|v|")
    ax4.legend()

    fig4.tight_layout()
    fig4.savefig("output/velocity_inside_outside.png", dpi=200)


    print("Wrote:")
    print(" - output/time_evolution.png")
    print(" - output/rcut_structure.png")
    print(" - output/phase_space.png")
    print(" - output/velocity_inside_outside.png")


if __name__ == "__main__":
    main()
