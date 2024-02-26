import numpy as np
from matplotlib import pyplot as plt
import time
from ModeSolverFD import ModeSolverFD

# Set up problem
um = 1e-6
lambda_ = 0.65*um
k0 = 2*np.pi/lambda_
beta = k0
Nx = 41
NoModes = 2
n_silica = 1.45
n_air = 1.0  
r_core = 25.5 * um
r_clad = 34.0 * um  
r_total = r_core + r_clad
x = np.linspace(-(26) * um, (26) * um, Nx)
y = x.copy()
x_mesh, y_mesh = np.meshgrid(x, y)
r_mesh = np.sqrt(x_mesh**2 + y_mesh**2)
n = np.ones([Nx, Nx], dtype=int)
n = n*n_silica
n[r_mesh < r_total] = n_silica 
n[r_mesh < (r_total - r_clad)] = 1
n[r_mesh > r_total] = 1

# Ellipses for glass
glass_ellipses = [
    {"center": (-19.35 * um, 4.72 * um), "major_axis": 11.28 * um, "minor_axis": 10.28 * um, "angle": 166.29},
    {"center": (-10.30 * um, 17.82 * um), "major_axis": 9.85 * um, "minor_axis": 9.82 * um, "angle": 120.03},
    {"center": (4.80 * um, 19.92 * um), "major_axis": 10.42 * um, "minor_axis": 9.49 * um, "angle": 76.45},
    {"center": (17.46 * um, 10.08 * um), "major_axis": 10.70 * um, "minor_axis": 9.91 * um, "angle": 30.00},
    {"center": (19.58 * um, -4.59 * um), "major_axis": 10.80 * um, "minor_axis": 10.09 * um, "angle": -13.19},
    {"center": (10.15 * um, -17.15 * um), "major_axis": 11.20 * um, "minor_axis": 10.09 * um, "angle": -59.38},
    {"center": (-4.73 * um, -19.40 * um), "major_axis": 11.36 * um, "minor_axis": 9.97 * um, "angle": -103.70},
    {"center": (-16.85 * um, -11.15 * um), "major_axis": 10.70 * um, "minor_axis": 10.60 * um, "angle": -147.51},
]

# Ellipses for air
air_ellipses = [
    {"center": (-19.35 * um, 4.72 * um), "major_axis": 10.88 * um, "minor_axis": 9.88 * um, "angle": 166.29},
    {"center": (-10.30 * um, 17.82 * um), "major_axis": 9.45 * um, "minor_axis": 9.42 * um, "angle": 120.03},
    {"center": (4.80 * um, 19.92 * um), "major_axis": 10.02 * um, "minor_axis": 9.09 * um, "angle": 76.45},
    {"center": (17.46 * um, 10.08 * um), "major_axis": 10.30 * um, "minor_axis": 9.51 * um, "angle": 30.00},
    {"center": (19.58 * um, -4.59 * um), "major_axis": 10.40 * um, "minor_axis": 9.69 * um, "angle": -13.19},
    {"center": (10.15 * um, -17.15 * um), "major_axis": 10.80 * um, "minor_axis": 9.69 * um, "angle": -59.38},
    {"center": (-4.73 * um, -19.40 * um), "major_axis": 10.96 * um, "minor_axis": 9.57 * um, "angle": -103.70},
    {"center": (-16.85 * um, -11.15 * um), "major_axis": 10.30 * um, "minor_axis": 10.20 * um, "angle": -147.51},
]

for ellipse_params in glass_ellipses + air_ellipses:
    ellipse_params["major_axis"] /= 2
    ellipse_params["minor_axis"] /= 2
    center = ellipse_params["center"]
    major_axis = ellipse_params["major_axis"]
    minor_axis = ellipse_params["minor_axis"]
    angle = np.deg2rad(ellipse_params["angle"]) 
    x_rotated = (x_mesh - center[0]) * np.cos(angle) + (y_mesh - center[1]) * np.sin(angle)
    y_rotated = (y_mesh - center[1]) * np.cos(angle) - (x_mesh - center[0]) * np.sin(angle)
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

dx = x[1] - x[0]
lambda_in_nm = lambda_ * 1e9
dx_in_nm = dx * 1e9
lambda_dx_ratio = lambda_ / dx

print("\nlambda_: {:.2f} nm".format(lambda_in_nm))
print("dx: {:.2f} nm".format(dx_in_nm))
print("lambda_/dx: {:.2f}".format(lambda_dx_ratio))
    
# Call FD solver
t = time.time()
RetVal, RetVal_Ex, RetVal_Ey, RetVal_Ez, RetVal_Hx, RetVal_Hy, \
RetVal_Hz, RetVal_Eabs, RetVal_Habs = ModeSolverFD(dx, n, lambda_, beta, NoModes)
elapsed = time.time()-t
print(elapsed)
# Plot modes
RetVal['beta'] = np.diag(RetVal['beta'])
for i in range(0, NoModes):
    fig, fillplot = plt.subplots(1, 1)
    fig.set_size_inches(8, 6)
    fig.set_dpi(600)
    contourf_ = fillplot.contourf(x / um, x / um, RetVal_Eabs[i], 100)
    fig.colorbar(contourf_).set_label(label='E_abs', labelpad=12, fontsize=14, weight='bold')
    plt.axis('square')
    real_neff = np.real(RetVal['beta'][i])/k0
    imag_neff = np.imag(RetVal['beta'][i])/k0
    fillplot.set_title('Effective Index: {:.6g}{:.6g}j'.format(real_neff, imag_neff),
                       pad=20, fontsize=14, fontweight="bold")
    fillplot.set_xlabel('\u03bcm', fontsize=14, fontweight="bold")
    fillplot.set_ylabel('\u03bcm', fontsize=14, fontweight="bold")
    plt.show()


