# Author: Ahmad Azizan (aaaba2@cam.ac.uk)
import os
import time
import h5py
from seabreeze.spectrometers import list_devices, Spectrometer
import matplotlib.pyplot as plt

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def file_exists(file_path):
    return os.path.exists(file_path)

def save_data_to_hdf5(file_path, data_dict):
    with h5py.File(file_path, "w") as file:
        for key, value in data_dict.items():
            file.create_dataset(key, data=value)

def initialize_data_and_spectrometer():
    data_directory = "./.h5 files"
    create_directory_if_not_exists(data_directory)
    background_file_path = os.path.join(data_directory, "background_spectrum.h5") 
    spectrometer = find_and_initialize_spectrometer()
    return data_directory, background_file_path, spectrometer
            
def find_and_initialize_spectrometer():
    devices = list_devices()
    if not devices:
        print("No spectrometer device found.")
        return None
    return Spectrometer(devices[0])

def get_measurement_settings():
    time_interval_seconds = 0
    number_of_spectra = 415
    integration_time_ms = 3
    total_duration_seconds = number_of_spectra * (time_interval_seconds + integration_time_ms / 1000)
    time_background = 10
    print(total_duration_seconds)
    return time_interval_seconds, number_of_spectra, integration_time_ms, total_duration_seconds, time_background
   
def save_background_spectrum(background):
    if not os.path.exists("./.h5 files"):
        os.makedirs("./.h5 files")
    with h5py.File("./.h5 files/background_spectrum.h5", "w") as file:
        file.create_dataset("wavelengths", data=background[0])
        file.create_dataset("intensities", data=background[1])
        
def load_background_spectrum():
    if os.path.exists("./.h5 files/background_spectrum.h5"):
        with h5py.File("./.h5 files/background_spectrum.h5", "r") as file:
            wavelengths = file["wavelengths"][:]
            intensities = file["intensities"][:]
            return wavelengths, intensities
    else:
        return None
    
def record_spectra_background(spectrometer, time_interval_seconds, integration_time_ms, total_duration_seconds):
    spectra = []
    timestamps = []  
    integration_time_micros = integration_time_ms * 1000  
    spectrometer.integration_time_micros(integration_time_micros)
    start_time = time.time()
    while (time.time() - start_time) <= total_duration_seconds:
        spectrum_data = spectrometer.spectrum(correct_dark_counts=True)  
        wavelengths = spectrometer.wavelengths()
        spectra.append((wavelengths, spectrum_data))
        time.sleep(time_interval_seconds)
        timestamps.append(time.time())  
    return spectra, timestamps

def record_spectra(spectrometer, time_interval_seconds, integration_time_ms, total_duration_seconds, avg_background):
    spectra = []
    timestamps = []
    integration_time_micros = integration_time_ms * 1000  
    spectrometer.integration_time_micros(integration_time_micros)
    start_time = time.perf_counter()  
    time_per_spectrum = []  
    spectrum_number = 1 
    while (time.perf_counter() - start_time) <= total_duration_seconds:
        spectrum_start_time = time.perf_counter()
        spectrum_data = spectrometer.spectrum(correct_dark_counts=True)
        spectrum_data = spectrum_data - avg_background
        wavelengths = spectrometer.wavelengths()
        time.sleep(time_interval_seconds)
        spectra.append((wavelengths, spectrum_data))
        timestamps.append(time.perf_counter())  
        spectrum_end_time = time.perf_counter()  
        time_taken_ms = (spectrum_end_time - spectrum_start_time) * 1000  
        time_per_spectrum.append(time_taken_ms)
        print(f"Spectrum {spectrum_number} recorded in {time_taken_ms:.2f} ms")
        spectrum_number += 1
    return spectra, timestamps, time_per_spectrum

def calculate_average_spectra(spectra_list):
    num_spectra = len(spectra_list)
    if num_spectra == 0:
        return None
    sum_intensities = [0] * len(spectra_list[0][1])
    for wavelengths, intensities in spectra_list:
        for i, intensity in enumerate(intensities):
            sum_intensities[i] += intensity
    avg_intensities = [intensity_sum / num_spectra for intensity_sum in sum_intensities]
    return avg_intensities

def check_and_handle_background_spectrum(background_file_path, spectrometer, time_interval_seconds, integration_time_ms, time_background):
    if not file_exists(background_file_path):
        input("Block the laser beam and press Enter when ready to read the background...")
        background_spectra, _ = record_spectra_background(spectrometer, time_interval_seconds, integration_time_ms, time_background)
        avg_background = calculate_average_spectra([spectrum[1] for spectrum in background_spectra])
        save_data_to_hdf5(background_file_path, {"wavelengths": spectrometer.wavelengths(), "intensities": avg_background})
        print("Background reading complete.")
        input("Unblock the laser beam and press Enter when ready to start live view...")
        background_wavelengths = spectrometer.wavelengths()
    else:
        print("Using the existing background spectrum.")
        background_wavelengths, avg_background = load_background_spectrum()
    return background_wavelengths, avg_background
        
def generate_filenames(data_directory, current_date):
    count = 0
    filename = os.path.join(data_directory, f"{current_date}.h5")
    averaged_filename = os.path.join(data_directory, f"averaged_{current_date}.h5")
    while file_exists(filename) or file_exists(averaged_filename):
        count += 1
        filename = os.path.join(data_directory, f"{current_date}({count}).h5")
        averaged_filename = os.path.join(data_directory, f"averaged_{current_date}({count}).h5")
    return filename, averaged_filename

def save_data_to_files(filename, averaged_filename, wavelengths, averaged_intensities, spectra, timestamps):
    save_data_to_hdf5(averaged_filename, {"wavelengths": wavelengths, "averaged_intensities": averaged_intensities})
    with h5py.File(filename, "w") as file:
        for i, (wavelengths, intensity_values) in enumerate(spectra):
            group_name = f"Spectrum_{i + 1:03d}"
            group = file.create_group(group_name)
            group.create_dataset("wavelengths", data=wavelengths)
            group.create_dataset("intensities", data=intensity_values)
            group.create_dataset("timestamp", data=timestamps[i])

def process_and_save_data(data_directory, spectra, timestamps):
    averaged_intensities = calculate_average_spectra([spectrum[1] for spectrum in spectra])
    wavelengths = spectra[0][0]  
    current_date = time.strftime("%Y-%m-%d")    
    filename, averaged_filename = generate_filenames(data_directory, current_date)  
    save_data_to_files(filename, averaged_filename, wavelengths, averaged_intensities, spectra, timestamps)  
    print("Averaged spectrum saved to", averaged_filename)
    print("All {} spectra saved to {}".format(len(spectra), filename))
    
def close_spectrometer(spectrometer):
    if spectrometer:
        spectrometer.close()
        
def plot_time_per_spectrum(time_per_spectrum):
    time_per_spectrum = time_per_spectrum[1:]
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.set_dpi(600)
    ax.set_xlabel("Spectrum Number", fontsize=14, fontweight="bold")
    ax.set_ylabel("Time Taken (milliseconds)", fontsize=14, fontweight="bold")
    ax.set_title("Time Taken to Record Each Spectrum", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    mean_time = sum(time_per_spectrum) / len(time_per_spectrum)
    median_time = sorted(time_per_spectrum)[len(time_per_spectrum) // 2] 
    std_deviation = (sum((x - mean_time) ** 2 for x in time_per_spectrum) / len(time_per_spectrum)) ** 0.5
    ax.errorbar(
        range(1, len(time_per_spectrum) + 1),
        time_per_spectrum,
        yerr=std_deviation,
        fmt='o',
        markersize=3,
        capsize=3,
        label='Time Taken',
    )

    ax.axhline(y=mean_time, color='blue', linestyle='-', label=f'Mean Time: {mean_time:.2f} ms')
    ax.axhline(y=median_time, color='green', linestyle='-', label=f'Median Time: {median_time:.2f} ms')
    ax.text(0.66, 0.80, f"Total Spectra: {len(time_per_spectrum)} spectra", transform=ax.transAxes, fontsize=10, color='blue')
    ax.text(0.66, 0.76, f"Total Time Taken: {sum(time_per_spectrum):.2f} ms", transform=ax.transAxes, fontsize=10, color='blue')
    ax.legend(loc='upper right', fontsize=10)
    plt.show()


def main():
    try:
        data_directory, background_file_path, spectrometer = initialize_data_and_spectrometer()
        time_interval_seconds, number_of_spectra, integration_time_ms, total_duration_seconds, time_background = get_measurement_settings()
        background_wavelengths, avg_background = check_and_handle_background_spectrum(background_file_path, spectrometer, time_interval_seconds, integration_time_ms, time_background)
        print("Start recording spectra...")
        spectra, timestamps, time_per_spectrum = record_spectra(spectrometer, time_interval_seconds, integration_time_ms, total_duration_seconds, avg_background)
        process_and_save_data(data_directory, spectra, timestamps)
        plot_time_per_spectrum(time_per_spectrum)   
    except Exception as e:
        print("An error occurred:", str(e))
    finally:
        close_spectrometer(spectrometer)

if __name__ == "__main__":
    main()
