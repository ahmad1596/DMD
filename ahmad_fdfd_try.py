import numpy as np
from scipy.sparse import spdiags, kron, eye, csr_matrix
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
import time

def construct_pml_profile(pml_width, sigma_max, Nx, Ny):
    pml_indices_x = np.arange(pml_width)[:, np.newaxis]
    pml_indices_y = np.arange(pml_width)[np.newaxis, :]
    sigma_x = np.zeros((Nx, Ny))
    sigma_y = np.zeros((Nx, Ny))
    sigma_x[:pml_width, :] = sigma_max * ((pml_indices_x + 0.5) / pml_width) ** 2
    sigma_x[-pml_width:, :] = sigma_max * ((pml_indices_x[::-1] + 0.5) / pml_width) ** 2
    sigma_y[:, :pml_width] = sigma_max * ((pml_indices_y + 0.5) / pml_width) ** 2
    sigma_y[:, -pml_width:] = sigma_max * ((pml_indices_y[::-1] + 0.5) / pml_width) ** 2
    return sigma_x, sigma_y

def plot_refractive_index_profile(x, y, n, sigma_x, sigma_y, pml_width, dx, dy):
    plt.figure(figsize=(8, 6), dpi=640)
    plt.imshow(n.T, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='viridis')
    plt.title('Refractive Index Profile with PML')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')

    # Add PML regions
    pml_color = 'gray'
    plt.fill_between([x.min(), x.min() + pml_width*dx], y.min(), y.max(), color=pml_color, alpha=0.2, label='PML')
    plt.fill_between([x.max() - pml_width*dx, x.max()], y.min(), y.max(), color=pml_color, alpha=0.2)
    plt.fill_between(x, y.min(), y.min() + pml_width*dy, color=pml_color, alpha=0.2)
    plt.fill_between(x, y.max() - pml_width*dy, y.max(), color=pml_color, alpha=0.2)

    # Add colorbar for refractive index
    plt.colorbar()
    plt.legend()

    plt.show(block=False)

def compute_loss(eigenvalues):
    kappa = np.imag(eigenvalues)
    wavelength = 2 * np.pi / np.real(eigenvalues)  # Assuming eigenvalues are the propagation constants
    return -0.2 * np.log10(np.exp(-2 * np.pi * kappa / wavelength))

def main():
    lam_start = 450e-9
    lam_end = 750e-9
    lam_interval = 300e-9
    num_modes = 2
    core_radius = 1e-6
    cladding_radius = 2e-6
    n_core = 1.45
    n_clad = 1.0
    Nx = 100
    Ny = 100
    dx = 1e-8
    dy = 1e-8
    eps0 = 8.854e-12

    pml_width = 10
    sigma_max = 1.5 / (np.pi * np.sqrt(eps0))

    x = np.linspace(-core_radius - dx, core_radius + dx, Nx)
    y = np.linspace(-core_radius - dy, core_radius + dy, Ny)
    X, Y = np.meshgrid(x, y)

    n = np.ones((Nx, Ny)) * n_clad
    index_core = (X ** 2 + Y ** 2 <= cladding_radius ** 2) & (X ** 2 + Y ** 2 >= core_radius ** 2)
    n[index_core] = n_core

    sigma_x, sigma_y = construct_pml_profile(pml_width, sigma_max, Nx, Ny)

    print("\nConstructed PML profiles.")
    plot_refractive_index_profile(x, y, n, sigma_x, sigma_y, pml_width, dx, dy)

    loss_data = np.zeros((0, num_modes))  

    for lam in np.arange(lam_start, lam_end + lam_interval, lam_interval):
        k0 = 2*np.pi/lam
        beta = k0

        print(f"\nWavelength: {lam*1e9} nm")
        print(f"Meshing: {Nx}x{Ny} cells")

        # Convert grid cell sizes to micrometers
        dx_um = dx * 1e6
        dy_um = dy * 1e6

        print(f"Grid cell size: {dx_um:.2f} µm x {dy_um:.2f} µm")

        # Print core, cladding, and computational domain dimensions in micrometers
        core_diameter_um = (cladding_radius - core_radius) * 1e6
        cladding_thickness_um = core_radius * 1e6

        print(f"Core diameter: {core_diameter_um:.2f} µm")
        print(f"Cladding thickness: {cladding_thickness_um:.2f} µm")
        print(f"PML width: {pml_width} cells")

        # Construct the derivative operators
        Dx = spdiags([-np.ones(Nx), np.ones(Nx)], [-1, 1], Nx, Nx) / dx
        Dy = spdiags([-np.ones(Ny), np.ones(Ny)], [-1, 1], Ny, Ny) / dy

        Lx = kron(Dx, eye(Ny, format='csr'))
        Ly = kron(eye(Nx, format='csr'), Dy)

        Sx = csr_matrix((sigma_x.flatten(), (np.arange(Nx * Ny), np.arange(Nx * Ny))), shape=(Nx * Ny, Nx * Ny))
        Sy = csr_matrix((sigma_y.flatten(), (np.arange(Nx * Ny), np.arange(Nx * Ny))), shape=(Nx * Ny, Nx * Ny))

        L = Lx @ Sy + Sx @ Ly

        eps_r = n ** 2
        A = L + spdiags(eps_r.flatten(), 0, Nx * Ny, Nx * Ny) / eps0

        print("Constructed operator matrices.")

        eigenvalues, eigenvectors = eigs(A, k=num_modes, sigma=beta)

        print("Computed eigenvalues and eigenvectors.")

        guided_modes = eigenvectors.T

        plt.figure(figsize=(10, 6), dpi=640)
        for i in range(num_modes):
            start_time = time.time()
            plt.subplot(2, (num_modes+1) // 2, i + 1)
            plt.imshow(np.abs(guided_modes[i].reshape(Nx, Ny)) ** 2, cmap='hot', extent=[x.min(), x.max(), y.min(), y.max()])
            plt.title(f'Mode {i + 1}')
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            plt.colorbar()
            end_time = time.time()
            print(f"Time taken to simulate Mode {i + 1}: {end_time - start_time:.4f} seconds")
        plt.tight_layout()
        plt.show()

        print("Plotted guided modes.")

        loss = compute_loss(eigenvalues)
        loss_data = np.vstack((loss_data, loss))

        cells_per_wavelength = lam / dx
        print(f"Number of cells per wavelength: {cells_per_wavelength:.2f}")

    plt.figure(figsize=(8, 6), dpi=640)
    for mode_idx in range(num_modes):
        wavelengths = np.arange(lam_start, lam_end + lam_interval, lam_interval)
        plt.plot(wavelengths, loss_data[:, mode_idx], label=f"Mode {mode_idx + 1}")
    plt.xlabel('Wavelength (m)')
    plt.ylabel('Loss (dB/cm)')
    plt.title('Loss vs. Wavelength for Different Mode Numbers')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
