import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

# Parameters
wavelength = 1.55  # Wavelength in micrometers
dx = 0.01          # Spatial step size in micrometers
dy = 0.01          # Spatial step size in micrometers
n_modes = 5        # Number of eigenmodes to compute
core_radius = 0.5  # Core radius in micrometers
core_index = 1.5   # Refractive index of the core
cladding_index = 1.45  # Refractive index of the cladding

# Define refractive index profile of the waveguide
def define_waveguide(n_x, n_y, core_radius, core_index, cladding_index):
    x = np.linspace(-1, 1, n_x)
    y = np.linspace(-1, 1, n_y)
    X, Y = np.meshgrid(x, y)

    # Define refractive index profile
    refractive_index = np.ones_like(X) * cladding_index  # Cladding
    refractive_index[np.sqrt(X**2 + Y**2) <= core_radius] = core_index  # Core

    return refractive_index, x, y

# Finite Difference Method to discretize Helmholtz equation
def fdtd(waveguide, dx, dy, wavelength):
    k0 = 2 * np.pi / wavelength
    eps_r = waveguide**2   # Square of refractive index

    # Construct sparse matrix for 2D finite difference approximation of Helmholtz equation
    nx, ny = waveguide.shape
    n = nx * ny

    data = np.zeros(7*n)
    rows = np.zeros(7*n)
    cols = np.zeros(7*n)

    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j
            data[7*idx] = -4 / dx**2 - 4 / dy**2 + k0**2 * eps_r[i, j]
            rows[7*idx] = idx
            cols[7*idx] = idx
            if i > 0:
                data[7*idx + 1] = 1 / dx**2
                rows[7*idx + 1] = idx
                cols[7*idx + 1] = idx - ny
            if i < nx - 1:
                data[7*idx + 2] = 1 / dx**2
                rows[7*idx + 2] = idx
                cols[7*idx + 2] = idx + ny
            if j > 0:
                data[7*idx + 3] = 1 / dy**2
                rows[7*idx + 3] = idx
                cols[7*idx + 3] = idx - 1
            if j < ny - 1:
                data[7*idx + 4] = 1 / dy**2
                rows[7*idx + 4] = idx
                cols[7*idx + 4] = idx + 1

    laplacian_sparse = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    # Solve eigenvalue problem
    eigvals, eigvecs = eigsh(-laplacian_sparse, k=n_modes, sigma=0, which='LM')
    eigvals /= (k0**2)

    return eigvals, eigvecs

# Main function
def main():
    # Define waveguide structure
    nx = 100  # Number of grid points along x
    ny = 100  # Number of grid points along y
    waveguide, x, y = define_waveguide(nx, ny, core_radius, core_index, cladding_index)

    # Plot refractive index profile
    plt.figure(dpi=600)
    plt.imshow(waveguide, cmap='viridis', origin='lower', extent=[x[0], x[-1], y[0], y[-1]])
    plt.colorbar(label='Refractive Index')
    plt.title('2D Refractive Index Profile')
    plt.xlabel('X (μm)')
    plt.ylabel('Y (μm)')
    plt.show()

    # Compute eigenmodes
    eigenvalues, eigenmodes = fdtd(waveguide, dx, dy, wavelength)

    # Print and plot results
    for i in range(n_modes):
        plt.figure(dpi=600)
        plt.imshow(np.abs(eigenmodes[:, i].reshape(nx, ny)), cmap='viridis', origin='lower', extent=[x[0], x[-1], y[0], y[-1]])
        plt.title(f"Mode {i+1}")
        plt.colorbar(label='Electric Field Amplitude')
        plt.xlabel('X (μm)')
        plt.ylabel('Y (μm)')
        plt.show()

if __name__ == "__main__":
    main()
