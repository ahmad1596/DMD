# Author: Ahmad Azizan (aaaba2@cam.ac.uk)
import h5py
import matplotlib.pyplot as plt
import numpy as np

def plot_background_spectrum(filename):
    with h5py.File(filename, "r") as file:
       wavelengths = file["wavelengths"][:]
       intensities = file["intensities"][:]          
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    fig.set_dpi(600)
    ax.plot(wavelengths, intensities, label="Background Intensity")
    ax.set_xlabel("Wavelength (nm)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Intensity (a.u.)", fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=8)
    plt.title("Background Intensity vs. Wavelength", fontsize=16, fontweight="bold")
    plt.show()

def plot_averaged_spectrum_without_fiber(filename):
    with h5py.File(filename, "r") as file:
        wavelengths = file["wavelengths"][:]
        averaged_intensities = file["averaged_intensities"][:]
    wavenumbers = 1 / (wavelengths * 1e-2)
    fig_wavelengths, ax_wavelengths = plt.subplots()
    fig_wavelengths.set_size_inches(8, 6)
    fig_wavelengths.set_dpi(600)
    ax_wavelengths.plot(wavelengths, averaged_intensities, label="Average Intensity (No Fiber)")
    ax_wavelengths.set_xlabel("Wavelength (nm)", fontsize=14, fontweight="bold")
    ax_wavelengths.set_ylabel("Average Intensity (a.u.)", fontsize=14, fontweight="bold")
    ax_wavelengths.tick_params(axis="both", which="major", labelsize=12, direction="in")
    ax_wavelengths.grid(color="gray", linestyle="--", linewidth=0.5)
    ax_wavelengths.legend(loc="upper right", fontsize=8)
    ax_wavelengths.set_title("Average Intensity (No Fiber) vs. Wavelength", fontsize=16, fontweight="bold")
    fig_wavenumbers, ax_wavenumbers = plt.subplots()
    fig_wavenumbers.set_size_inches(8, 6)
    fig_wavenumbers.set_dpi(600)
    ax_wavenumbers.plot(wavenumbers, averaged_intensities, label="Average Intensity (No Fiber)")
    ax_wavenumbers.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=14, fontweight="bold")
    ax_wavenumbers.set_ylabel("Average Intensity (a.u.)", fontsize=14, fontweight="bold")
    ax_wavenumbers.tick_params(axis="both", which="major", labelsize=12, direction="in")
    ax_wavenumbers.grid(color="gray", linestyle="--", linewidth=0.5)
    ax_wavenumbers.legend(loc="upper right", fontsize=8)
    ax_wavenumbers.set_title("Average Intensity (No Fiber) vs. Wavenumber", fontsize=16, fontweight="bold")
    plt.show()


def plot_averaged_spectrum_with_fiber(filename):
    with h5py.File(filename, "r") as file:
        wavelengths = file["wavelengths"][:]
        averaged_intensities = file["averaged_intensities"][:]
    wavenumbers = 1 / (wavelengths * 1e-2)
    fig_wavelengths, ax_wavelengths = plt.subplots()
    fig_wavelengths.set_size_inches(8, 6)
    fig_wavelengths.set_dpi(600)
    ax_wavelengths.plot(wavelengths, averaged_intensities, label="Average Intensity (With Fiber)")
    ax_wavelengths.set_xlabel("Wavelength (nm)", fontsize=14, fontweight="bold")
    ax_wavelengths.set_ylabel("Average Intensity (a.u.)", fontsize=14, fontweight="bold")
    ax_wavelengths.tick_params(axis="both", which="major", labelsize=12, direction="in")
    ax_wavelengths.grid(color="gray", linestyle="--", linewidth=0.5)
    ax_wavelengths.legend(loc="upper right", fontsize=8)
    ax_wavelengths.set_title("Average Intensity (With Fiber) vs. Wavelength", fontsize=16, fontweight="bold")
    fig_wavenumbers, ax_wavenumbers = plt.subplots()
    fig_wavenumbers.set_size_inches(8, 6)
    fig_wavenumbers.set_dpi(600)
    ax_wavenumbers.plot(wavenumbers, averaged_intensities, label="Average Intensity (With Fiber)")
    ax_wavenumbers.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=14, fontweight="bold")
    ax_wavenumbers.set_ylabel("Average Intensity (a.u.)", fontsize=14, fontweight="bold")
    ax_wavenumbers.tick_params(axis="both", which="major", labelsize=12, direction="in")
    ax_wavenumbers.grid(color="gray", linestyle="--", linewidth=0.5)
    ax_wavenumbers.legend(loc="upper right", fontsize=8)
    ax_wavenumbers.set_title("Average Intensity (With Fiber) vs. Wavenumber", fontsize=16, fontweight="bold")
    plt.show()
    
def plot_normalized_averaged_spectrum_fiber(filename):
    with h5py.File(filename, "r") as file:
        wavelengths = file["wavelengths"][:]
        normalized_averaged_intensities = file["normalized_intensities"][:]
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    fig.set_dpi(600)
    ax.plot(wavelengths, normalized_averaged_intensities, label="Normalized Averaged Intensity (With Fiber/Without Fiber)")
    ax.set_xlabel("Wavelength (nm)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Normalized Averaged Intensity", fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=8)
    plt.title("Normalized Averaged Intensity vs. Wavelength (With Fiber/Without Fiber)", fontsize=16, fontweight="bold")
    plt.show()
      
def plot_normalized_power_spectrum(filename):
    with h5py.File(filename, "r") as file:
        wavelengths = file["wavelengths"][:]
        normalized_power_spectrum = file["normalized_power_spectrum"][:]
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    fig.set_dpi(600)
    ax.plot(wavelengths, normalized_power_spectrum, label="Normalized Power Spectrum (%)")
    ax.set_xlabel("Wavelength (nm)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Normalized Power Spectrum (%)", fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=8)
    plt.title("Normalized Power Spectrum vs. Wavelength", fontsize=16, fontweight="bold")
    plt.show()


def plot_individual_spectra_normalized(filename_individual_spectra, filename_averaged_spectrum_without_fiber):
    with h5py.File(filename_individual_spectra, "r") as file_individual:
        num_spectra = len(file_individual.keys())
        with h5py.File(filename_averaged_spectrum_without_fiber, "r") as file_averaged:
            averaged_intensities = file_averaged["averaged_intensities"][:] 
        fig_wavelengths, ax_wavelengths = plt.subplots(figsize=(8, 6))
        fig_wavelengths.set_dpi(600)
        ax_wavelengths.set_xlabel("Wavelength (nm)", fontsize=14, fontweight="bold")
        ax_wavelengths.set_ylabel("Normalized Intensity", fontsize=14, fontweight="bold")
        ax_wavelengths.set_title("Normalized Individual Spectra", fontsize=16, fontweight="bold")
        ax_wavelengths.tick_params(axis="both", which="major", labelsize=12, direction="in")
        ax_wavelengths.grid(color="gray", linestyle="--", linewidth=0.5)
        fig_wavenumbers, ax_wavenumbers = plt.subplots(figsize=(8, 6))
        fig_wavenumbers.set_dpi(600)
        ax_wavenumbers.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=14, fontweight="bold")
        ax_wavenumbers.set_ylabel("Normalized Intensity", fontsize=14, fontweight="bold")
        ax_wavenumbers.set_title("Normalized Individual Spectra", fontsize=16, fontweight="bold")
        ax_wavenumbers.tick_params(axis="both", which="major", labelsize=12, direction="in")
        ax_wavenumbers.grid(color="gray", linestyle="--", linewidth=0.5)
        for i in range(num_spectra):
            group_name = f"Spectrum_{i+1:03d}"
            group = file_individual[group_name]
            wavelengths = group["wavelengths"][:]
            intensities = group["intensities"][:]
            wavenumbers = 1 / (wavelengths * 1e-2) 
            normalized_intensity = intensities[1] / averaged_intensities
            ax_wavelengths.plot(wavelengths, normalized_intensity, label=f"Spectrum {i+1}")
            ax_wavenumbers.plot(wavenumbers, normalized_intensity, label=f"Spectrum {i+1}")
        ax_wavelengths.legend(loc="upper right", fontsize=8)
        ax_wavenumbers.legend(loc="upper right", fontsize=8)
        plt.show()
  
def extract_and_plot_normalized_intensity_vs_time(filename, target_wavelength, filename_averaged_spectrum_without_fiber):
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
            wavelength_index = (abs(wavelengths - target_wavelength)).argmin()
            intensity_at_wavelength = intensities[1, wavelength_index] / averaged_intensities[wavelength_index]
            if i > 0:
                extracted_intensities.append(intensity_at_wavelength)
                timestamps.append(timestamp)       
        time_elapsed_ms = [(timestamp - timestamps[0]) * 1000 for timestamp in timestamps]
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.set_dpi(600)
        ax.set_xlabel("Time Elapsed (ms)", fontsize=14, fontweight="bold")
        ax.set_ylabel(f"Normalized Intensity at {target_wavelength} nm", fontsize=14, fontweight="bold")
        ax.set_title(f"Normalized Intensity vs Time at {target_wavelength} nm", fontsize=16, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
        ax.grid(color="gray", linestyle="--", linewidth=0.5)
        ax.plot(time_elapsed_ms, extracted_intensities, marker='o', linestyle='-', markersize=3)
        plt.show()

def extract_and_plot_normalized_intensity_vs_time_range(filename, start_wavelength, end_wavelength, num_points, filename_averaged_spectrum_without_fiber):
    wavelength_range = np.linspace(start_wavelength, end_wavelength, num_points)
    with h5py.File(filename, "r") as file:
        num_spectra = len(file.keys())
        with h5py.File(filename_averaged_spectrum_without_fiber, "r") as file_averaged:
            averaged_intensities = file_averaged["averaged_intensities"][:]
        extracted_intensities = []
        timestamps = []
        wavelengths = file["Spectrum_001/wavelengths"][:] 
        wavelength_indices = [np.abs(wavelengths - wavelength).argmin() for wavelength in wavelength_range]
        for i in range(num_spectra):
            group_name = f"Spectrum_{i+1:03d}"
            group = file[group_name]
            intensities = group["intensities"][:]
            timestamp = group["timestamp"][()]
            intensities_at_range = intensities[1, wavelength_indices]
            normalized_intensities_at_range = intensities_at_range / averaged_intensities[wavelength_indices]
            if i > 0:
                extracted_intensities.append(normalized_intensities_at_range)
                timestamps.append(timestamp) 
        time_elapsed_ms = [(timestamp - timestamps[0]) * 1000 for timestamp in timestamps]
        cmap = plt.get_cmap("tab10")
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.set_dpi(600)
        ax.set_xlabel("Time Elapsed (ms)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Normalized Intensity", fontsize=14, fontweight="bold")
        ax.set_title("Normalized Intensity vs Time for Wavelength Range", fontsize=16, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
        ax.grid(color="gray", linestyle="--", linewidth=0.5)
        for idx, wavelength in enumerate(wavelength_range):
            ax.plot(time_elapsed_ms, [intensity[idx] for intensity in extracted_intensities], label=f"{wavelength:.1f} nm", color=cmap(idx))
        ax.legend(loc="upper right", fontsize=8)
        plt.show()
        
def main():
    try:
        filename_background = "./.h5_files/averaged_background_spectrum_10ms.h5"
        print("Plotting Background Spectrum...")
        plot_background_spectrum(filename_background) 
        print("Background Spectrum plotted.")

        filename_averaged_spectra_without_fiber = "./.h5_files/averaged_spectrum_without_fiber_10ms.h5"
        print("Plotting Averaged Spectrum without Fiber...")
        plot_averaged_spectrum_without_fiber(filename_averaged_spectra_without_fiber)
        print("Averaged Spectrum without Fiber plotted.")
       
        filename_averaged_spectra_with_fiber = "./.h5_files/2024-02-05_averaged_spectrum_with_fiber_10ms.h5"
        print("Plotting Averaged Spectrum with Fiber...")
        plot_averaged_spectrum_with_fiber(filename_averaged_spectra_with_fiber)
        print("Averaged Spectrum with Fiber plotted.")
        
        filename_normalized_averaged_spectrum_fiber = "./.h5_files/2024-02-05_normalized_averaged_spectrum_10ms.h5"
        print("Plotting Normalized Averaged Spectrum with Fiber...")
        plot_normalized_averaged_spectrum_fiber(filename_normalized_averaged_spectrum_fiber)
        print("Normalized Averaged Spectrum with Fiber plotted.")
        
        filename_normalized_power_spectrum = "./.h5_files/2024-02-05_normalized_power_spectrum_10ms.h5"
        print("Plotting Normalized Power Spectrum...")
        plot_normalized_power_spectrum(filename_normalized_power_spectrum)
        print("Normalized Power Spectrum plotted.")

        filename_individual_spectra_with_fiber = "./.h5_files/2024-02-05_spectrum_with_fiber_10ms.h5"
        print("Plotting Individual Spectra with Fiber...")
        plot_individual_spectra_normalized(filename_individual_spectra_with_fiber, filename_averaged_spectra_without_fiber)
        print("Individual Spectra with Fiber plotted.")
        
        target_wavelength = 532
        print(f"Extracting and Plotting Intensity vs. Time for Wavelength {target_wavelength} nm...")
        extract_and_plot_normalized_intensity_vs_time(filename_individual_spectra_with_fiber, target_wavelength, filename_averaged_spectra_without_fiber)
        print(f"Intensity vs. Time for Wavelength {target_wavelength} nm plotted.")
        
        start_wavelength = 540
        end_wavelength = 530
        num_points = 10
        print(f"Extracting and Plotting Intensity vs. Time for Wavelength Range {start_wavelength} nm to {end_wavelength} nm...")
        extract_and_plot_normalized_intensity_vs_time_range(filename_individual_spectra_with_fiber, start_wavelength, end_wavelength, num_points, filename_averaged_spectra_without_fiber)
        print(f"Intensity vs. Time for Wavelength Range {start_wavelength} nm to {end_wavelength} nm plotted.")
        
    except Exception as e:
        print("An error occurred:", str(e))
    
if __name__ == "__main__":
    main()
