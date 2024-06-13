import os
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
from seabreeze.spectrometers import list_devices, Spectrometer

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def file_exists(file_path):
    exists = os.path.exists(file_path)
    print(f"File exists at {file_path}: {exists}")
    return exists

def save_data_to_hdf5(file_path, data_dict):
    with h5py.File(file_path, "w") as file:
        for key, value in data_dict.items():
            file.create_dataset(key, data=value)
    print(f"Saved data to HDF5 file: {file_path}")

def initialize_data_and_spectrometer(integration_time_ms):
    data_directory = "./.h5_files"
    create_directory_if_not_exists(data_directory)
    spectrometer = find_and_initialize_spectrometer()
    print("Data directory and spectrometer initialized.")
    return data_directory, spectrometer

def find_and_initialize_spectrometer():
    devices = list_devices()
    print(f"Devices found: {devices}")
    if not devices:
        print("No spectrometer device found.")
        return None
    spectrometer = Spectrometer(devices[0])
    print(f"Spectrometer initialized: {spectrometer.model}")
    return spectrometer

def get_background_measurement_settings():
    bg_time_interval_seconds = 1
    bg_number_of_spectra = 10  
    bg_integration_time_ms = 10
    bg_total_duration_seconds = bg_number_of_spectra * (bg_time_interval_seconds + bg_integration_time_ms / 1000)
    bg_time_background = bg_total_duration_seconds
    print(f"Background Measurement settings: Time Interval = {bg_time_interval_seconds} seconds, Number of Spectra = {bg_number_of_spectra}, Integration Time = {bg_integration_time_ms} ms")
    print(f"Total measurement duration for background: {bg_total_duration_seconds:.2f} seconds")
    return bg_time_interval_seconds, bg_number_of_spectra, bg_integration_time_ms, bg_total_duration_seconds, bg_time_background

def get_spectra_measurement_settings():
    spectra_time_interval_seconds = 1
    spectra_number_of_spectra = 10
    spectra_integration_time_ms = 10
    spectra_total_duration_seconds = spectra_number_of_spectra * (spectra_time_interval_seconds + spectra_integration_time_ms / 1000)
    print(f"Spectra Measurement settings: Time Interval = {spectra_time_interval_seconds} seconds, Number of Spectra = {spectra_number_of_spectra}, Integration Time = {spectra_integration_time_ms} ms")
    print(f"Total measurement duration for spectra: {spectra_total_duration_seconds:.2f} seconds")
    return spectra_time_interval_seconds, spectra_number_of_spectra, spectra_integration_time_ms, spectra_total_duration_seconds

def record_background_spectrum(spectrometer, bg_time_interval_seconds, bg_integration_time_ms, bg_time_background):
    spectra = []
    timestamps = []
    spectrometer.integration_time_micros(bg_integration_time_ms * 1000)
    start_time = time.time()
    print("Recording background spectrum...")
    while (time.time() - start_time) <= bg_time_background:
        spectrum_data = spectrometer.spectrum(correct_dark_counts=True)
        wavelengths = spectrometer.wavelengths()
        spectra.append((wavelengths, spectrum_data))
        timestamps.append(time.time())
        time.sleep(bg_time_interval_seconds)
    print(f"Recorded {len(spectra)} background spectra.")
    avg_background = calculate_average_spectrum(spectra)
    return spectra, timestamps, avg_background

def record_spectra(spectrometer, spectra_time_interval_seconds, spectra_number_of_spectra, spectra_integration_time_ms, spectra_total_duration_seconds):
    spectra = []
    timestamps = []
    spectrometer.integration_time_micros(spectra_integration_time_ms * 1000)
    print("Press Enter when ready to start recording spectra...")
    input()  # Wait for user to press Enter
    print("Recording spectra...")
    start_time = time.time()
    for i in range(spectra_number_of_spectra):
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time > spectra_total_duration_seconds:
            break
        print(f"Elapsed time: {elapsed_time:.2f} seconds, Recording spectrum {i+1}/{spectra_number_of_spectra}")
        spectrum_data = spectrometer.spectrum(correct_dark_counts=True)
        wavelengths = spectrometer.wavelengths()
        spectra.append((wavelengths, spectrum_data))
        timestamps.append(current_time)
        time.sleep(spectra_time_interval_seconds)
    print(f"Recorded {len(spectra)} spectra.")
    return spectra, timestamps

def save_all_spectra(filename, wavelengths, spectra, timestamps):
    data_dict = {
        "wavelengths": wavelengths,
        "spectra_data": np.array([spectrum[1] for spectrum in spectra]),
        "timestamps": np.array(timestamps)
    }
    save_data_to_hdf5(filename, data_dict)
    print(f"All spectra saved to: {filename}")

def plot_spectrum(wavelengths, spectrum_data, fig_title):
    print(f"Plotting spectrum with wavelengths shape {wavelengths.shape} and spectrum_data shape {spectrum_data.shape}")
    if spectrum_data.ndim == 2:
        spectrum_data = spectrum_data[1]
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=600)
    ax.plot(wavelengths, spectrum_data, marker='o', markersize=2, linestyle='-', color='b')
    ax.set_xlabel('Wavelength')
    ax.set_ylabel('Intensity')
    ax.set_title(fig_title)
    plt.tight_layout()
    plt.show()

def close_spectrometer(spectrometer):
    if spectrometer:
        spectrometer.close()
        print("Spectrometer closed.")

def calculate_average_spectrum(spectra_list):
    num_spectra = len(spectra_list)
    if num_spectra == 0:
        return None
    sum_intensities = np.zeros_like(spectra_list[0][1])
    for wavelengths, intensities in spectra_list:
        sum_intensities += intensities
    avg_spectrum = sum_intensities / num_spectra
    print(f"Calculated average spectrum from {num_spectra} spectra.")
    return avg_spectrum

def main():
    try:
        print("Initializing data directory and spectrometer...")
        data_directory, spectrometer = initialize_data_and_spectrometer(20)

        if not spectrometer:
            print("Exiting due to spectrometer initialization failure.")
            return

        print("Recording background spectrum...")
        bg_time_interval_seconds, bg_number_of_spectra, bg_integration_time_ms, bg_total_duration_seconds, bg_time_background = get_background_measurement_settings()
        background_spectra, background_timestamps, avg_background = record_background_spectrum(spectrometer, bg_time_interval_seconds, bg_integration_time_ms, bg_time_background)

        print("Recording spectra...")
        spectra_time_interval_seconds, spectra_number_of_spectra, spectra_integration_time_ms, spectra_total_duration_seconds = get_spectra_measurement_settings()
        spectra, timestamps = record_spectra(spectrometer, spectra_time_interval_seconds, spectra_number_of_spectra, spectra_integration_time_ms, spectra_total_duration_seconds)

        print("Enter a name for the HDF5 file (without extension):")
        file_name = input().strip()
        if not file_name:
            file_name = "spectra_data"
        filename = os.path.join(data_directory, f"{file_name}.h5")

        print("Saving all spectra to one file...")
        wavelengths = spectra[0][0] if spectra else np.array([])
        save_all_spectra(filename, wavelengths, spectra, timestamps)

        print("Plotting individual spectra...")
        for i, (wavelengths, spectrum_data) in enumerate(spectra):
            fig_title = f"Individual Spectrum {i+1}"
            plot_spectrum(wavelengths, spectrum_data, fig_title)

    except Exception as e:
        print("An error occurred:", str(e))
    finally:
        print("Closing spectrometer...")
        close_spectrometer(spectrometer)

if __name__ == "__main__":
    main()
