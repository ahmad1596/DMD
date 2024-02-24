import numpy as np
from matplotlib import pyplot as plt

def main():
    um = 1e-6
    Nx = 1000
    n_silica = 1.45
    n_air = 1.0  
    r_core = 25.5 * um
    r_clad = 34.0 * um  
    r_total = r_core + r_clad
    x = np.linspace(8 * um, 20 * um, Nx)
    y = x.copy()
    x_mesh, y_mesh = np.meshgrid(x, y)
    r_mesh = np.sqrt(x_mesh**2 + y_mesh**2)
    n = np.ones([Nx, Nx], dtype=float)
    n *= n_silica
    n[r_mesh < r_total] = n_silica
    n[r_mesh < (r_total - r_clad)] = 1
    n[r_mesh > r_total] = 1

    # Ellipses for glass
    glass_ellipses = [
        {"center": (20 * um, 0), "major_axis": 6 * um, "minor_axis": 3 * um, "angle": 0},
        {"center": (-20 * um, 0), "major_axis": 6 * um, "minor_axis": 3 * um, "angle": 0},
        {"center": (0, 20 * um), "major_axis": 6 * um, "minor_axis": 3 * um, "angle": np.pi/2},
        {"center": (0, -20 * um), "major_axis": 6 * um, "minor_axis": 3 * um, "angle": np.pi/2},
        {"center": (14 * um, 14 * um), "major_axis": 6 * um, "minor_axis": 3 * um, "angle": -np.pi/4},
        {"center": (-14 * um, -14 * um), "major_axis": 6 * um, "minor_axis": 3 * um, "angle": -np.pi/4},
        {"center": (14 * um, -14 * um), "major_axis": 6 * um, "minor_axis": 3 * um, "angle": np.pi/4},
        {"center": (-14 * um, 14 * um), "major_axis": 6 * um, "minor_axis": 3 * um, "angle": np.pi/4},
    ]

    # Ellipses for air
    air_ellipses = [
        {"center": (20 * um, 0), "major_axis": 5 * um, "minor_axis": 2 * um, "angle": 0},
        {"center": (-20 * um, 0), "major_axis": 5 * um, "minor_axis": 2 * um, "angle": 0},
        {"center": (0, 20 * um), "major_axis": 5 * um, "minor_axis": 2 * um, "angle": np.pi/2},
        {"center": (0, -20 * um), "major_axis": 5 * um, "minor_axis": 2 * um, "angle": np.pi/2},
        {"center": (14 * um, 14 * um), "major_axis": 5 * um, "minor_axis": 2 * um, "angle": -np.pi/4},
        {"center": (-14 * um, -14 * um), "major_axis": 5 * um, "minor_axis": 2 * um, "angle": -np.pi/4},
        {"center": (14 * um, -14 * um), "major_axis": 5 * um, "minor_axis": 2 * um, "angle": np.pi/4},
        {"center": (-14 * um, 14 * um), "major_axis": 5 * um, "minor_axis": 2 * um, "angle": np.pi/4},
    ]

    for ellipse_params in glass_ellipses + air_ellipses:
        center = ellipse_params["center"]
        major_axis = ellipse_params["major_axis"]
        minor_axis = ellipse_params["minor_axis"]
        angle = ellipse_params["angle"]
        x_rotated = (x_mesh - center[0]) * np.cos(angle) - (y_mesh - center[1]) * np.sin(angle)
        y_rotated = (x_mesh - center[0]) * np.sin(angle) + (y_mesh - center[1]) * np.cos(angle)
        ellipse_mask = (
            (x_rotated / major_axis) ** 2
            + (y_rotated / minor_axis) ** 2
        ) < 1
        n[ellipse_mask] = n_silica if ellipse_params in glass_ellipses else n_air

    fig, fillplot = plt.subplots(1, 1)
    fig.set_size_inches(8, 6)
    fig.set_dpi(640)
    contourf_ = fillplot.contourf(x / um, x / um, n, 100)
    fig.colorbar(contourf_).set_label(label='Refractive Index', labelpad=12,
                                       fontsize=14, weight='bold')
    plt.axis('square')
    fillplot.set_xlabel('\u03bcm', fontsize=14, fontweight="bold")
    fillplot.set_ylabel('\u03bcm', fontsize=14, fontweight="bold")
    plt.show()

if __name__ == "__main__":
    main()
