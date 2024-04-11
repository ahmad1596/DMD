import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import time

nx = 100
ny = 100
wavelength = 0.55
k0 = 2 * np.pi / wavelength
guess_k = k0 
n_modes = 2

pml_thickness = 0.1
pml_sigma_max = 1.5

min_x = -1
max_x = 1
min_y = -1
max_y = 1

def define_waveguide(nx, ny):
    core_radius = 0.5
    core_index = 1.0
    cladding_index = 1.45
    x = np.linspace(min_x, max_x, nx)
    y = np.linspace(min_y, max_y, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)
    refractive_index = np.ones_like(X) * cladding_index
    refractive_index[np.sqrt(X**2 + Y**2) <= core_radius] = core_index
    return refractive_index, dx, dy, x, y

def plot_refractive_index_profile(waveguide, x, y):
    plt.figure(figsize=(8, 6), dpi=600)
    plt.imshow(waveguide, cmap='viridis', origin='lower', extent=[min_x - pml_thickness, max_x + pml_thickness, min_y - pml_thickness, max_y + pml_thickness])
    plt.colorbar(label='Refractive Index')
    plt.title('2D Refractive Index Profile with PML')
    plt.xlabel('X (μm)')
    plt.ylabel('Y (μm)')

def plot_pml_regions(nx_pml, ny_pml):
    plt.fill_between([min_x - pml_thickness, min_x], min_y - pml_thickness, max_y + pml_thickness, color='gray', alpha=0.5)
    plt.fill_between([max_x, max_x + pml_thickness], min_y - pml_thickness, max_y + pml_thickness, color='gray', alpha=0.5)
    plt.fill_betweenx([min_y - pml_thickness, min_y], min_x - pml_thickness, max_x + pml_thickness, color='gray', alpha=0.5)
    plt.fill_betweenx([max_y, max_y + pml_thickness], min_x - pml_thickness, max_x + pml_thickness, color='gray', alpha=0.5)

    plt.text(min_x - pml_thickness * 0.5, 0, 'PML', color='gray', ha='center', va='center', rotation='vertical')
    plt.text(max_x + pml_thickness * 0.5, 0, 'PML', color='gray', ha='center', va='center', rotation='vertical')
    plt.text(0, min_y - pml_thickness * 0.5, 'PML', color='gray', ha='center', va='center')
    plt.text(0, max_y + pml_thickness * 0.5, 'PML', color='gray', ha='center', va='center')

def plot_eigenmodes(eigenmodes, nx, ny, nx_pml, ny_pml):
    for i in range(n_modes):
        plt.figure(figsize=(8, 6), dpi=600)
        plt.imshow(np.abs(eigenmodes[:, i].reshape(nx, ny)), cmap='viridis', origin='lower', extent=[min_x - pml_thickness, max_x + pml_thickness, min_y - pml_thickness, max_y + pml_thickness])
        plt.colorbar(label='Electric Field Amplitude')
        plt.title(f"Mode {i+1}")
        plt.xlabel('X (μm)')
        plt.ylabel('Y (μm)')
        plot_pml_regions(nx_pml, ny_pml)
    plt.show()

def fdfd(waveguide, dx, dy, wavelength):
    k0 = 2 * np.pi / wavelength
    eps_r = waveguide**2
    nx, ny = waveguide.shape
    nx_pml, ny_pml = int(pml_thickness / dx), int(pml_thickness / dy)
    n = nx * ny
    data = np.zeros(9 * n, dtype=complex)
    rows = np.zeros(9 * n)
    cols = np.zeros(9 * n)
    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j
            data[9 * idx] = -4 / dx**2 - 4 / dy**2 + k0**2 * eps_r[i, j]
            rows[9 * idx] = idx
            cols[9 * idx] = idx
            if i > 0:
                data[9 * idx + 1] = 1 / dx**2
                rows[9 * idx + 1] = idx
                cols[9 * idx + 1] = idx - ny
            if i < nx - 1:
                data[9 * idx + 2] = 1 / dx**2
                rows[9 * idx + 2] = idx
                cols[9 * idx + 2] = idx + ny
            if j > 0:
                data[9 * idx + 3] = 1 / dy**2
                rows[9 * idx + 3] = idx
                cols[9 * idx + 3] = idx - 1
            if j < ny - 1:
                data[9 * idx + 4] = 1 / dy**2
                rows[9 * idx + 4] = idx
                cols[9 * idx + 4] = idx + 1
            if i < nx_pml:
                sigma_x = pml_sigma_max * ((nx_pml - i) / nx_pml)**2
                pml_term = sigma_x * 1j / (2 * k0)
                data[9 * idx] += pml_term
                data[9 * idx + 5] = -pml_term
            elif i >= nx - nx_pml:
                sigma_x = pml_sigma_max * ((i - (nx - nx_pml - 1)) / nx_pml)**2
                pml_term = sigma_x * 1j / (2 * k0)
                data[9 * idx] += pml_term
                data[9 * idx + 5] = -pml_term
            if j < ny_pml:
                sigma_y = pml_sigma_max * ((ny_pml - j) / ny_pml)**2
                pml_term = sigma_y * 1j / (2 * k0)
                data[9 * idx] += pml_term
                data[9 * idx + 7] = -pml_term
            elif j >= ny - ny_pml:
                sigma_y = pml_sigma_max * ((j - (ny - ny_pml - 1)) / ny_pml)**2
                pml_term = sigma_y * 1j / (2 * k0)
                data[9 * idx] += pml_term
                data[9 * idx + 7] = -pml_term
    laplacian_sparse = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    eigvals, eigvecs = eigsh(-laplacian_sparse, k=n_modes, sigma=guess_k, which='LM')
    eigvals /= (k0**2)
    return eigvals, eigvecs

def print_coordinates_and_pml(dx, dy, wavelength, pml_thickness):
    nx_pml = int(np.ceil(pml_thickness / dx))
    ny_pml = int(np.ceil(pml_thickness / dy))
    print("Minimum coordinates (µm):", min_x - pml_thickness, min_y - pml_thickness)
    print("Maximum coordinates (µm):", max_x + pml_thickness, max_y + pml_thickness)
    print(f"Wavelength (µm): {wavelength}")
    print(f"Grid spacing (dx, dy): {dx:.3f} µm, {dy:.3f} µm")
    grid_points_per_wavelength_x = round(wavelength / dx)
    grid_points_per_wavelength_y = round(wavelength / dy)
    print(f"Grid points per wavelength (in x-direction): {grid_points_per_wavelength_x}")
    print(f"Grid points per wavelength (in y-direction): {grid_points_per_wavelength_y}")
    print("PML regions:")
    print(f"X-direction PML: {nx_pml} grid points from {min_x - pml_thickness} to {min_x} µm and {nx_pml} grid points from {max_x} µm to {max_x + pml_thickness} µm")
    print(f"Y-direction PML: {ny_pml} grid points from {min_y - pml_thickness} to {min_y} µm and {ny_pml} grid points from {max_y} µm to {max_y + pml_thickness} µm")
    return nx_pml, ny_pml

def sort_eigenvalues_and_eigenvectors(eigenvalues, eigenmodes):
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenmodes = eigenmodes[:, sorted_indices]
    return eigenvalues, eigenmodes

def print_propagation_constants(eigenvalues, k0):
    print("Propagation constants for each mode:")
    for i, eigenvalue in enumerate(eigenvalues):
        propagation_constant = np.sqrt(eigenvalue) / k0
        print(f"Mode {i+1} (\u03BCm\u207B\u00B9): {propagation_constant:.10f}")

def main():
    start_time = time.time()
    waveguide, dx, dy, x, y = define_waveguide(nx, ny)
    nx_pml, ny_pml = print_coordinates_and_pml(dx, dy, wavelength, pml_thickness)
    plot_refractive_index_profile(waveguide, x, y)
    plot_pml_regions(nx_pml, ny_pml)
    eigenvalues, eigenmodes = fdfd(waveguide, dx, dy, wavelength)
    eigenvalues, eigenmodes = sort_eigenvalues_and_eigenvectors(eigenvalues, eigenmodes)
    print_propagation_constants(eigenvalues, k0)
    plot_eigenmodes(eigenmodes, nx, ny, nx_pml, ny_pml)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Simulation took {elapsed_time:.2f} seconds")
if __name__ == "__main__":
    main()
