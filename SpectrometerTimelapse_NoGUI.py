# Author: Ahmad Azizan (aaaba2@cam.ac.uk)
import os
import time
import h5py
from seabreeze.spectrometers import list_devices, Spectrometer
import numpy as np

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
    time_interval_seconds = 1
    number_of_spectra = 10
    integration_time_ms = 10
    total_duration_seconds = number_of_spectra * (time_interval_seconds + integration_time_ms / 1000)
    time_background = total_duration_seconds
    print(f"Total measurement duration: {total_duration_seconds:.2f} seconds")
    return time_interval_seconds, number_of_spectra, integration_time_ms, total_duration_seconds, time_background

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
    start_time = time.time()
    while (time.time() - start_time) <= total_duration_seconds:
        spectrum_data = spectrometer.spectrum(correct_dark_counts=True)
        wavelengths = spectrometer.wavelengths()
        spectrum_data = spectrum_data - avg_background
        spectra.append((wavelengths, spectrum_data))
        time.sleep(time_interval_seconds)
        timestamps.append(time.time())
    return spectra, timestamps

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
        background_spectra, background_timestamps = record_spectra_background(spectrometer, time_interval_seconds, integration_time_ms, time_background)
        avg_background = calculate_average_spectra([spectrum[1] for spectrum in background_spectra])
        save_data_to_hdf5(background_file_path, {"wavelengths": spectrometer.wavelengths(), "intensities": avg_background})
        print("Background reading complete.")
        input("Unblock the laser beam and press Enter when ready to start live view...")
        background_wavelengths = spectrometer.wavelengths()
    else:
        print("Using the existing background spectrum.")
        background_wavelengths, avg_background = load_background_spectrum()
    return background_wavelengths, avg_background

def record_or_load_spectrum_without_fiber(spectrometer, data_directory, time_interval_seconds, integration_time_ms, time_background):
    count = 0
    current_date = time.strftime('%Y-%m-%d')
    while True:
        spectrum_without_fiber_filename, spectrum_without_fiber_averaged_filename = generate_filenames(data_directory, current_date, with_fiber=False, count=count)
        if not file_exists(spectrum_without_fiber_filename):
            break
        count += 1
    if not file_exists(spectrum_without_fiber_averaged_filename):
        input(f"Remove the fiber, block the laser beam, and press Enter when ready to record the spectrum without fiber (Count={count + 1})...")
        background_without_fiber_spectra, background_without_fiber_timestamps = record_spectra_background(spectrometer, time_interval_seconds, integration_time_ms, time_background)
        avg_background_without_fiber = calculate_average_spectra([spectrum[1] for spectrum in background_without_fiber_spectra])
        print(f"Integration time for background without fiber recording: {integration_time_ms} ms")
        spectrum_without_fiber, spectrum_without_fiber_timestamps = record_spectra(spectrometer, time_interval_seconds, integration_time_ms, time_background, avg_background_without_fiber)
        avg_spectrum_without_fiber = calculate_average_spectra([spectrum[1] for spectrum in spectrum_without_fiber])
        save_data_to_hdf5(spectrum_without_fiber_averaged_filename, {"wavelengths": spectrometer.wavelengths(), "intensities": avg_spectrum_without_fiber})
        print(f"Spectrum without fiber recorded and saved with Count={count + 1}")
    else:
        print(f"Using the existing spectrum without fiber (Count={count + 1}).")
    input("Replace the fiber, unblock the laser beam, and press Enter when ready to continue with fiber spectra...")
    return spectrum_without_fiber_filename, avg_spectrum_without_fiber

def generate_filenames(data_directory, with_fiber=True, count=0):
    current_date = time.strftime('%Y-%m-%d')
    file_type = "with_fiber" if with_fiber else "without_fiber"
    if count > 0:
        filename = os.path.join(data_directory, f"{current_date}_spectrum_{file_type}({count}).h5")
        averaged_filename = os.path.join(data_directory, f"{current_date}_averaged_spectrum_{file_type}({count}).h5")
    else:
        filename = os.path.join(data_directory, f"{current_date}_spectrum_{file_type}.h5")
        averaged_filename = os.path.join(data_directory, f"{current_date}_averaged_spectrum_{file_type}.h5")
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
    filename, averaged_filename = generate_filenames(data_directory, current_date, with_fiber=True)
    spectrum_without_fiber_filename, spectrum_without_fiber_averaged_filename = generate_filenames(data_directory, current_date, with_fiber=False)
    save_data_to_files(filename, averaged_filename, wavelengths, averaged_intensities, spectra, timestamps)
    print("Averaged spectrum with fiber saved to", averaged_filename)
    print("All {} spectra with fiber saved to {}".format(len(spectra), filename))

def close_spectrometer(spectrometer):
    if spectrometer:
        spectrometer.close()
        
def normalize_spectrum_fiber(data_directory, wavelengths, spectrum_with_fiber_filename, spectrum_without_fiber_filename):
    with h5py.File(spectrum_with_fiber_filename, "r") as file:
        spectrum_with_fiber_groups = file.keys()
        spectrum_with_fiber_intensities = [file[group_name + "/intensities"][:] for group_name in spectrum_with_fiber_groups]
    with h5py.File(spectrum_without_fiber_filename, "r") as file:
        spectrum_without_fiber_groups = file.keys()
        spectrum_without_fiber_intensities = [file[group_name + "/intensities"][:] for group_name in spectrum_without_fiber_groups]
    for spec_with, spec_without in zip(spectrum_with_fiber_intensities, spectrum_without_fiber_intensities):
        if len(spec_with) != len(spec_without):
            raise ValueError("Spectra with and without fiber have different lengths.")
    normalized_intensities_list = [spec_with / spec_without for spec_with, spec_without in zip(spectrum_with_fiber_intensities, spectrum_without_fiber_intensities)]
    current_date = time.strftime('%Y-%m-%d')
    for i, normalized_intensities in enumerate(normalized_intensities_list):
        normalized_filename = os.path.join(data_directory, f"{current_date}_normalized_spectrum_fiber_{i + 1}.h5")
        save_data_to_hdf5(normalized_filename, {"wavelengths": wavelengths, "normalized_intensities": normalized_intensities})
        print(f"Normalized spectrum with fiber {i + 1} saved to", normalized_filename)
    averaged_normalized_intensities = np.mean(normalized_intensities_list, axis=0)
    averaged_normalized_filename = os.path.join(data_directory, f"{current_date}_averaged_normalized_spectrum_fiber.h5")
    save_data_to_hdf5(averaged_normalized_filename, {"wavelengths": wavelengths, "averaged_normalized_intensities": averaged_normalized_intensities})
    print("Averaged normalized spectrum with fiber saved to", averaged_normalized_filename)

def normalize_power():
    power_measurement_wavelength = 550
    input_power_mW = float(input("Enter the input power in milliwatts: "))
    output_power_uW = float(input("Enter the output power in microwatts: "))
    input_power_uW = input_power_mW * 1000  # Convert milliwatts to microwatts
    power_percentage = (output_power_uW * 100) / input_power_uW
    print(f"Wavelength at {power_measurement_wavelength} nm")
    print(f"Normalized Power Percentage: {power_percentage:.2f}%")
    return power_percentage

def calculate_normalized_spectral_power_transmission(data_directory, wavelengths, power_percentage):
    current_date = time.strftime('%Y-%m-%d')
    averaged_normalized_filename = os.path.join(data_directory, f"{current_date}_averaged_normalized_spectrum_fiber.h5")
    with h5py.File(averaged_normalized_filename, "r") as file:
        averaged_normalized_intensities = file["averaged_normalized_intensities"][:]
    normalized_spectral_power_transmission = (averaged_normalized_intensities * power_percentage) / np.max(averaged_normalized_intensities)
    normalized_power_transmission_filename = os.path.join(data_directory, f"{current_date}_normalized_power_transmission.h5")
    save_data_to_hdf5(normalized_power_transmission_filename, {"wavelengths": wavelengths, "normalized_spectral_power_transmission": normalized_spectral_power_transmission})
    print("Normalized Spectral Power Transmission saved to", normalized_power_transmission_filename)
    return normalized_power_transmission_filename


def main():
    try:
        data_directory, background_file_path, spectrometer = initialize_data_and_spectrometer()
        time_interval_seconds, number_of_spectra, integration_time_ms, total_duration_seconds, time_background = get_measurement_settings()
        wavelengths, avg_spectrum_without_fiber = record_or_load_spectrum_without_fiber(spectrometer, data_directory, time_interval_seconds, integration_time_ms, time_background)
        background_wavelengths, avg_background = check_and_handle_background_spectrum(background_file_path, spectrometer, time_interval_seconds, integration_time_ms, time_background)
        filename, filename_averaged = generate_filenames(data_directory, with_fiber=True)
        spectrum_without_fiber_filename, spectrum_without_fiber_averaged_filename = generate_filenames(data_directory, with_fiber=False)
        print("Start recording spectra...")
        spectra, timestamps = record_spectra(spectrometer, time_interval_seconds, integration_time_ms, total_duration_seconds, avg_background)
        process_and_save_data(data_directory, spectra, timestamps)
        normalize_spectrum_fiber(data_directory, wavelengths, filename, spectrum_without_fiber_filename)
        power_percentage = normalize_power()
        calculate_normalized_spectral_power_transmission(data_directory, wavelengths, power_percentage)
    except Exception as e:
        print("An error occurred:", str(e))
    finally:
        close_spectrometer(spectrometer)

if __name__ == "__main__":
    main()
