import numpy as np
import matplotlib.pyplot as plt


def draw_shell_sphere(ax, n_shells: int, r_max: float = 1.0):
    """
    Draw concentric wireframe spheres on a 3D axis.
    """
    radii = np.linspace(r_max / n_shells, r_max, n_shells)

    # Sphere mesh
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 22)

    for k, r in enumerate(radii, start=1):
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones_like(u), np.cos(v))

        # Blue wireframes
        alpha = 0.18 + 0.12 * (k / n_shells)
        lw = 0.4 if n_shells > 8 else 0.5

        ax.plot_wireframe(
            x, y, z,
            rstride=1,
            cstride=1,
            color=(0.0, 0.35, 1.0, alpha),
            linewidth=lw
        )

    # Clean rendering
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_zlim(-r_max, r_max)

    ax.set_axis_off()
    ax.view_init(elev=18, azim=30)


def make_figure():
    ns = [2, 4, 8]

    fig = plt.figure(figsize=(12, 5), facecolor="white")

    for i, n in enumerate(ns, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")

        draw_shell_sphere(ax, n_shells=n)

        # Only display N labels
        ax.text2D(
            0.5,
            0.91,
            f"N={n}",
            transform=ax.transAxes,
            fontsize=20,
            ha="center",
            fontfamily="serif",
            color="black"
        )

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=-0.18, hspace=0.0)

    return fig


if __name__ == "__main__":
    fig = make_figure()

    plt.savefig(
        "blue_shell_spheres.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.0,
        facecolor="white"
    )

    plt.show()