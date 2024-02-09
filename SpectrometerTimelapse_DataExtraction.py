# Author: Ahmad Azizan (aaaba2@cam.ac.uk)
import h5py
import matplotlib.pyplot as plt
import numpy as np

def plot_spectrum(wavelengths, intensities, x_label, y_label, title, legend_label=None):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    ax.plot(wavelengths, intensities, label=legend_label)
    ax.set_xlabel(x_label, fontsize=14, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=16, fontweight="bold")
    plt.show()

def plot_background_spectrum(filename):
    with h5py.File(filename, "r") as file:
        wavelengths = file["wavelengths"][:]
        intensities = file["intensities"][:]
    plot_spectrum(wavelengths, intensities, "Wavelength (nm)", "Intensity (a.u.)",
                  "Background Intensity vs. Wavelength", "Background Intensity")

def plot_averaged_spectrum(filename, fiber_type="No Fiber"):
    with h5py.File(filename, "r") as file:
        wavelengths = file["wavelengths"][:]
        averaged_intensities = file["averaged_intensities"][:]
    wavenumbers = 1 / (wavelengths * 1e-2)
    plot_spectrum(wavelengths, averaged_intensities, "Wavelength (nm)", "Averaged Intensity (a.u.)",
                  f"Averaged Intensity ({fiber_type}) vs. Wavelength", f"Averaged Intensity ({fiber_type})")
    plot_spectrum(wavenumbers, averaged_intensities, "Wavenumber (cm$^{-1}$)", "Averaged Intensity (a.u.)",
                  f"Averaged Intensity ({fiber_type}) vs. Wavenumber", f"Averaged Intensity ({fiber_type})")

def plot_effective_fiber_spectrum(filename):
    with h5py.File(filename, "r") as file:
        wavelengths = file["wavelengths"][:]
        effective_fiber_spectrum = file["effective_fiber_spectrum"][:]
    plot_spectrum(wavelengths, effective_fiber_spectrum, "Wavelength (nm)", "Averaged Intensity (a.u.)",
                  "Effective Fiber Spectrum vs. Wavelength", "Effective Fiber Spectrum")

def plot_normalized_transmission(filename):
    with h5py.File(filename, "r") as file:
        wavelengths = file["wavelengths"][:]
        normalized_transmission = file["effective_fiber_transmission"][:]
    plot_spectrum(wavelengths, normalized_transmission, "Wavelength (nm)", "Normalized Transmission (%)",
                  "Normalized Transmission vs. Wavelength", "Normalized Transmission")

def plot_individual_spectra_normalized(filename_individual_spectra, filename_averaged_spectrum_without_fiber):
    with h5py.File(filename_individual_spectra, "r") as file_individual:
        num_spectra = len(file_individual.keys())
        with h5py.File(filename_averaged_spectrum_without_fiber, "r") as file_averaged:
            averaged_intensities = file_averaged["averaged_intensities"][:]
        fig_wavelengths, ax_wavelengths = plt.subplots(figsize=(8, 6), dpi=600)
        ax_wavelengths.set_xlabel("Wavelength (nm)", fontsize=14, fontweight="bold")
        ax_wavelengths.set_ylabel("Normalized Intensity (%)", fontsize=14, fontweight="bold")
        ax_wavelengths.set_title("Normalized Individual Spectra", fontsize=16, fontweight="bold")
        ax_wavelengths.tick_params(axis="both", which="major", labelsize=12, direction="in")
        ax_wavelengths.grid(color="gray", linestyle="--", linewidth=0.5)
        fig_wavenumbers, ax_wavenumbers = plt.subplots(figsize=(8, 6), dpi=600)
        ax_wavenumbers.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=14, fontweight="bold")
        ax_wavenumbers.set_ylabel("Normalized Intensity (%)", fontsize=14, fontweight="bold")
        ax_wavenumbers.set_title("Normalized Individual Spectra", fontsize=16, fontweight="bold")
        ax_wavenumbers.tick_params(axis="both", which="major", labelsize=12, direction="in")
        ax_wavenumbers.grid(color="gray", linestyle="--", linewidth=0.5)
        for i in range(num_spectra):
            group_name = f"Spectrum_{i+1:03d}"
            group = file_individual[group_name]
            wavelengths = group["wavelengths"][:]
            intensities = group["intensities"][:]
            wavenumbers = 1 / (wavelengths * 1e-2)
            normalized_intensity = 100 * np.abs(intensities[1] - averaged_intensities) / max(np.abs(intensities[1] - averaged_intensities))
            ax_wavelengths.plot(wavelengths, normalized_intensity, label=f"Spectrum {i+1}")
            ax_wavenumbers.plot(wavenumbers, normalized_intensity, label=f"Spectrum {i+1}")
        ax_wavelengths.legend(loc="upper right", fontsize=8)
        ax_wavenumbers.legend(loc="upper right", fontsize=8)
        plt.show()

def extract_and_plot_effective_intensity_vs_time(filename, target_wavelength, filename_averaged_spectrum_without_fiber):
    with h5py.File(filename, "r") as file:
        num_spectra = len(file.keys())
        with h5py.File(filename_averaged_spectrum_without_fiber, "r") as file_averaged:
            averaged_intensities = file_averaged["averaged_intensities"][:]
        extracted_intensities = []
        timestamps = []
        for i in range(num_spectra):
            group_name = f"Spectrum_{i+1:03d}"
            group = file[group_name]
            wavelengths = group["wavelengths"][:]
            intensities = group["intensities"][:]
            timestamp = group["timestamp"][()]
            wavelength_index = (np.abs(wavelengths - target_wavelength)).argmin()
            intensity_at_wavelength = np.abs(intensities[1, wavelength_index] - averaged_intensities[wavelength_index])
            if i > 0:
                extracted_intensities.append(intensity_at_wavelength)
                timestamps.append(timestamp)
        time_elapsed_ms = [(timestamp - timestamps[0]) * 1000 for timestamp in timestamps]
        plot_spectrum(time_elapsed_ms, extracted_intensities, "Time Elapsed (ms)",
                      f"Effective Intensity at {target_wavelength} nm",
                      f"Effective Intensity vs Time at {target_wavelength} nm")

def extract_and_plot_intensity_vs_time_range(filename, start_wavelength, end_wavelength, num_points,
                                             filename_averaged_spectrum_without_fiber):
    wavelength_range = np.linspace(start_wavelength, end_wavelength, num_points)
    with h5py.File(filename, "r") as file:
        num_spectra = len(file.keys())
        with h5py.File(filename_averaged_spectrum_without_fiber, "r") as file_averaged:
            averaged_intensities = file_averaged["averaged_intensities"][:]
        timestamps = []
        wavelengths = file["Spectrum_001/wavelengths"][:]
        wavelength_indices = [np.abs(wavelengths - wavelength).argmin() for wavelength in wavelength_range]
        extracted_effective_intensities = []
        extracted_normalized_intensities = []
        for i in range(num_spectra):
            group_name = f"Spectrum_{i+1:03d}"
            group = file[group_name]
            intensities = group["intensities"][:]
            timestamp = group["timestamp"][()]
            intensities_at_range = intensities[1, wavelength_indices]
            effective_intensities_at_range = np.abs(intensities_at_range - averaged_intensities[wavelength_indices])
            normalized_intensity_at_range = 100 * effective_intensities_at_range / np.max(effective_intensities_at_range)
            if i > 0:
                extracted_effective_intensities.append(effective_intensities_at_range)
                extracted_normalized_intensities.append(normalized_intensity_at_range)
                timestamps.append(timestamp)
        time_elapsed_ms = [(timestamp - timestamps[0]) * 1000 for timestamp in timestamps]

        cmap = plt.get_cmap("tab10")
        fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=600)
        ax1.set_xlabel("Time Elapsed (ms)", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Effective Intensity", fontsize=14, fontweight="bold")
        ax1.set_title("Effective Intensity vs Time for Wavelength Range", fontsize=16, fontweight="bold")
        ax1.tick_params(axis="both", which="major", labelsize=12, direction="in")
        ax1.grid(color="gray", linestyle="--", linewidth=0.5)
        for idx, wavelength in enumerate(wavelength_range):
            ax1.plot(time_elapsed_ms, [intensity[idx] for intensity in extracted_effective_intensities],
                     label=f"{wavelength:.1f} nm", color=cmap(idx))
        ax1.legend(loc="upper right", fontsize=8)

        fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=600)
        ax2.set_xlabel("Time Elapsed (ms)", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Normalized Intensity (%)", fontsize=14, fontweight="bold")
        ax2.set_title("Normalized Intensity vs Time for Wavelength Range", fontsize=16, fontweight="bold")
        ax2.tick_params(axis="both", which="major", labelsize=12, direction="in")
        ax2.grid(color="gray", linestyle="--", linewidth=0.5)
        for idx, wavelength in enumerate(wavelength_range):
            ax2.plot(time_elapsed_ms, [intensity[idx] for intensity in extracted_normalized_intensities],
                     label=f"{wavelength:.1f} nm", color=cmap(idx))
        ax2.legend(loc="upper right", fontsize=8)

        plt.show()

def main():
    try:
        filename_background = "./.h5_files/averaged_background_spectrum_20ms.h5"
        print("Plotting Background Spectrum...")
        plot_background_spectrum(filename_background)
        print("Background Spectrum plotted.")

        filename_averaged_spectra_without_fiber = "./.h5_files/averaged_spectrum_without_fiber_20ms.h5"
        print("Plotting Averaged Spectrum without Fiber...")
        plot_averaged_spectrum(filename_averaged_spectra_without_fiber, fiber_type="No Fiber")
        print("Averaged Spectrum without Fiber plotted.")

        filename_averaged_spectra_with_fiber = "./.h5_files/2024-02-09_averaged_spectrum_with_fiber_20ms.h5"
        print("Plotting Averaged Spectrum with Fiber...")
        plot_averaged_spectrum(filename_averaged_spectra_with_fiber, fiber_type="With Fiber")
        print("Averaged Spectrum with Fiber plotted.")

        filename_effective_fiber_spectrum = "./.h5_files/2024-02-09_effective_fiber_spectrum_20ms.h5"
        print("Plotting Effective Fiber Spectrum...")
        plot_effective_fiber_spectrum(filename_effective_fiber_spectrum)
        print("Effective Fiber Spectrum plotted.")

        filename_normalized_transmission = "./.h5_files/2024-02-09_effective_fiber_transmission_20ms.h5"
        print("Plotting Normalized Transmission...")
        plot_normalized_transmission(filename_normalized_transmission)
        print("Normalized Transmission plotted.")

        filename_individual_spectra_with_fiber = "./.h5_files/2024-02-09_spectrum_with_fiber_20ms.h5"
        print("Plotting Individual Spectra with Fiber...")
        plot_individual_spectra_normalized(filename_individual_spectra_with_fiber,
                                           filename_averaged_spectra_without_fiber)
        print("Individual Spectra with Fiber plotted.")

        target_wavelength = 450
        print(f"Extracting and Plotting Intensity vs. Time for Wavelength {target_wavelength} nm...")
        extract_and_plot_effective_intensity_vs_time(filename_individual_spectra_with_fiber,
                                                     target_wavelength, filename_averaged_spectra_without_fiber)
        print(f"Intensity vs. Time for Wavelength {target_wavelength} nm plotted.")

        start_wavelength = 445
        end_wavelength = 455
        num_points = 10
        print(f"Extracting and Plotting Intensity vs. Time for Wavelength Range {start_wavelength} nm to "
              f"{end_wavelength} nm...")
        extract_and_plot_intensity_vs_time_range(filename_individual_spectra_with_fiber, start_wavelength,
                                                  end_wavelength, num_points, filename_averaged_spectra_without_fiber)
        print(f"Intensity vs. Time for Wavelength Range {start_wavelength} nm to {end_wavelength} nm plotted.")

    except Exception as e:
        print("An error occurred:", str(e))

if __name__ == "__main__":
    main()
