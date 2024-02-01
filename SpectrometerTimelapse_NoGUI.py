# Author: Ahmad Azizan (aaaba2@cam.ac.uk)
import os
import time
import h5py
from seabreeze.spectrometers import list_devices, Spectrometer
import numpy as np

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
    background_file_path = os.path.join(data_directory, f"averaged_background_spectrum_{integration_time_ms}ms.h5")
    spectrometer = find_and_initialize_spectrometer()
    print("Data directory and spectrometer initialized.")
    return data_directory, background_file_path, spectrometer

def find_and_initialize_spectrometer():
    devices = list_devices()
    if not devices:
        print("No spectrometer device found.")
        return None
    spectrometer = Spectrometer(devices[0])
    print(f"Spectrometer initialized: {spectrometer.model}")
    return spectrometer

def get_measurement_settings():
    time_interval_seconds = 1
    number_of_spectra = 1
    integration_time_ms = 10
    total_duration_seconds = number_of_spectra * (time_interval_seconds + integration_time_ms / 1000)
    time_background = total_duration_seconds
    print(f"Measurement settings: Time Interval = {time_interval_seconds} seconds, Number of Spectra = {number_of_spectra}, Integration Time = {integration_time_ms} ms")
    print(f"Total measurement duration: {total_duration_seconds:.2f} seconds")
    return time_interval_seconds, number_of_spectra, integration_time_ms, total_duration_seconds, time_background

def load_background_spectrum(file_path):
    if file_exists(file_path):
        with h5py.File(file_path, "r") as file:
            wavelengths = file["wavelengths"][:]
            intensities = file["intensities"][:]
            print(f"Loaded background spectrum from: {file_path}")
            return wavelengths, intensities
    else:
        print("Background spectrum file not found.")
        return None

def record_spectra_background(spectrometer, time_interval_seconds, integration_time_ms, total_duration_seconds):
    spectra = []
    timestamps = []
    integration_time_micros = integration_time_ms * 1000
    spectrometer.integration_time_micros(integration_time_micros)
    start_time = time.time()
    print("Recording background spectra...")
    while (time.time() - start_time) <= total_duration_seconds:
        spectrum_data = spectrometer.spectrum(correct_dark_counts=True)
        wavelengths = spectrometer.wavelengths()
        spectra.append((wavelengths, spectrum_data))
        time.sleep(time_interval_seconds)
        timestamps.append(time.time())
    print(f"Recorded {len(spectra)} background spectra.")
    return spectra, timestamps

def record_spectra(spectrometer, time_interval_seconds, integration_time_ms, total_duration_seconds, avg_background):
    spectra = []
    timestamps = []
    integration_time_micros = integration_time_ms * 1000
    spectrometer.integration_time_micros(integration_time_micros)
    start_time = time.time()
    print("Recording spectra with background subtraction...")
    while (time.time() - start_time) <= total_duration_seconds:
        spectrum_data = spectrometer.spectrum(correct_dark_counts=True)
        wavelengths = spectrometer.wavelengths()
        spectrum_data = spectrum_data - avg_background
        spectra.append((wavelengths, spectrum_data))
        time.sleep(time_interval_seconds)
        timestamps.append(time.time())
    print(f"Recorded {len(spectra)} spectra with background subtraction.")
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
    print(f"Calculated average intensities from {num_spectra} spectra.")
    return avg_intensities

def check_and_handle_background_spectrum(background_file_path, spectrometer, time_interval_seconds, integration_time_ms, time_background):
    if not file_exists(background_file_path):
        input("Make sure the laser is OFF and press Enter when ready to read the background...")
        background_spectra, background_timestamps = record_spectra_background(spectrometer, time_interval_seconds, integration_time_ms, time_background)
        avg_background = calculate_average_spectra([spectrum[1] for spectrum in background_spectra])
        save_data_to_hdf5(background_file_path, {"wavelengths": spectrometer.wavelengths(), "intensities": avg_background})
        print("Background reading complete. Saved background spectrum.")
        input("Turn ON the laser and press Enter when ready to start live view...")
        background_wavelengths = spectrometer.wavelengths()
    else:
        print("Using the existing background spectrum.")
        background_wavelengths, avg_background = load_background_spectrum(background_file_path)
    return background_wavelengths, avg_background

def record_or_load_spectrum_without_fiber(spectrometer, data_directory, time_interval_seconds, integration_time_ms, time_background):
    avg_spectrum_without_fiber = None
    try:
        spectrum_without_fiber_filename, spectrum_without_fiber_averaged_filename = generate_filenames_without_fiber(data_directory, integration_time_ms)
        if not file_exists(spectrum_without_fiber_averaged_filename):
            input(f"Remove the fiber, block the laser beam, and press Enter when ready to record the spectrum without fiber (Integration Time={integration_time_ms} ms)...")
            background_without_fiber_spectra, background_without_fiber_timestamps = record_spectra_background(spectrometer, time_interval_seconds, integration_time_ms, time_background)
            avg_background_without_fiber = calculate_average_spectra([spectrum[1] for spectrum in background_without_fiber_spectra])
            print(f"Integration time for background without fiber recording: {integration_time_ms} ms")
            spectrum_without_fiber, spectrum_without_fiber_timestamps = record_spectra(spectrometer, time_interval_seconds, integration_time_ms, time_background, avg_background_without_fiber)
            avg_spectrum_without_fiber = calculate_average_spectra([spectrum[1] for spectrum in spectrum_without_fiber])
            save_data_to_hdf5(spectrum_without_fiber_averaged_filename, {"wavelengths": spectrometer.wavelengths(), "averaged_intensities": avg_spectrum_without_fiber})
            print(f"Spectrum without fiber recorded and saved with Integration Time={integration_time_ms} ms")
        else:
            print(f"Using the existing spectrum without fiber (Integration Time={integration_time_ms} ms.)")
        input("Replace the fiber, unblock the laser beam, and press Enter when ready to continue with fiber spectra...")
    except Exception as e:
        print("An error occurred:", str(e))
    return spectrum_without_fiber_filename, avg_spectrum_without_fiber

def generate_filenames_with_fiber(data_directory, integration_time_ms, count=0):
    current_date = time.strftime('%Y-%m-%d')
    base_filename = os.path.join(data_directory, f"{current_date}_spectrum_with_fiber_{integration_time_ms}ms")
    averaged_filename = os.path.join(data_directory, f"{current_date}_averaged_spectrum_with_fiber_{integration_time_ms}ms")
    while file_exists(f"{averaged_filename}.h5"):
        count += 1
        base_filename = os.path.join(data_directory, f"{current_date}_spectrum_with_fiber_{integration_time_ms}ms_({count})")
        averaged_filename = os.path.join(data_directory, f"{current_date}_averaged_spectrum_with_fiber_{integration_time_ms}ms_({count})")
    base_filename += ".h5"
    averaged_filename += ".h5"
    print(f"Generated filenames with fiber: {base_filename}, {averaged_filename}")
    return base_filename, averaged_filename

def generate_filenames_without_fiber(data_directory, integration_time_ms):
    filename = os.path.join(data_directory, f"spectrum_without_fiber_{integration_time_ms}ms.h5")
    averaged_filename = os.path.join(data_directory, f"averaged_spectrum_without_fiber_{integration_time_ms}ms.h5")
    print(f"Generated filenames without fiber: {filename}, {averaged_filename}")
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
    print(f"Saved data to files: {averaged_filename}, {filename}")

def process_and_save_data(data_directory, spectra, timestamps, integration_time_ms):
    averaged_intensities = calculate_average_spectra([spectrum[1] for spectrum in spectra])
    wavelengths = spectra[0][0]
    filename, averaged_filename = generate_filenames_with_fiber(data_directory, integration_time_ms)
    spectrum_without_fiber_filename, spectrum_without_fiber_averaged_filename = generate_filenames_without_fiber(data_directory, integration_time_ms)
    save_data_to_files(filename, averaged_filename, wavelengths, averaged_intensities, spectra, timestamps)
    print("Averaged spectrum with fiber saved to", averaged_filename)
    print("All {} spectra with fiber saved to {}".format(len(spectra), filename))

def close_spectrometer(spectrometer):
    if spectrometer:
        spectrometer.close()
        print("Spectrometer closed.")

def calculate_and_save_normalized_spectrum(spectrum_with_fiber_averaged_filename, spectrum_without_fiber_averaged_filename):
    with h5py.File(spectrum_with_fiber_averaged_filename, "r") as file_fiber, \
         h5py.File(spectrum_without_fiber_averaged_filename, "r") as file_no_fiber:
        wavelengths_fiber = file_fiber["wavelengths"][:]
        intensities_fiber = file_fiber["averaged_intensities"][:]
        intensities_no_fiber = file_no_fiber["averaged_intensities"][:]
        normalized_intensities = intensities_fiber / intensities_no_fiber
        normalized_averaged_spectrum_filename = spectrum_with_fiber_averaged_filename.replace("averaged_spectrum_with_fiber", "normalized_averaged_spectrum")        
        save_data_to_hdf5(normalized_averaged_spectrum_filename, {"wavelengths": wavelengths_fiber, "normalized_intensities": normalized_intensities})
        print("Normalized averaged spectrum saved to:", normalized_averaged_spectrum_filename)
    return normalized_averaged_spectrum_filename
    
def normalize_power():
    power_measurement_wavelength = 532
    input_power_uW = float(input("Enter the input power in microwatts: "))
    background_input_power_uW = float(input("Enter the background input power in microwatts: "))
    output_power_uW = float(input("Enter the output power in microwatts: "))
    background_output_power_uW = float(input("Enter the background output power in microwatts: "))
    normalized_input_power_uW = input_power_uW - background_input_power_uW
    normalized_output_power_uW = output_power_uW - background_output_power_uW
    power_percentage = (normalized_output_power_uW * 100) / normalized_input_power_uW
    print(f"Wavelength at {power_measurement_wavelength} nm")
    print(f"Normalized Power Percentage: {power_percentage:.2f}%")
    return power_percentage

def calculate_and_save_normalized_power_spectrum(normalized_averaged_spectrum_filename, power_percentage):
    with h5py.File(normalized_averaged_spectrum_filename, "r") as file_normalized:
        wavelengths_fiber = file_normalized["wavelengths"][:]
        normalized_intensities = file_normalized["normalized_intensities"][:]  

    normalized_power_spectrum = (normalized_intensities * power_percentage) / np.max(normalized_intensities)
    normalized_power_spectrum_filename = normalized_averaged_spectrum_filename.replace("normalized_averaged_spectrum", "normalized_power_spectrum")
    save_data_to_hdf5(normalized_power_spectrum_filename, {"wavelengths": wavelengths_fiber, "normalized_power_spectrum": normalized_power_spectrum})
    print("Normalized power spectrum saved to:", normalized_power_spectrum_filename)


def main():
    try:
        
        print("Initializing data directory and spectrometer...")
        time_interval_seconds, number_of_spectra, integration_time_ms, total_duration_seconds, time_background = get_measurement_settings()
        data_directory, background_file_path, spectrometer = initialize_data_and_spectrometer(integration_time_ms)

        print("Checking and handling background spectrum...")
        background_wavelengths, avg_background = check_and_handle_background_spectrum(background_file_path, spectrometer, time_interval_seconds, integration_time_ms, time_background)

        print("Recording or loading spectrum without fiber...")
        wavelengths, avg_spectrum_without_fiber = record_or_load_spectrum_without_fiber(spectrometer, data_directory, time_interval_seconds, integration_time_ms, time_background)
        spectrum_with_fiber_filename, spectrum_with_fiber_averaged_filename = generate_filenames_with_fiber(data_directory, integration_time_ms)
        spectrum_without_fiber_filename, spectrum_without_fiber_averaged_filename = generate_filenames_without_fiber(data_directory, integration_time_ms)

        print("Start recording spectra...")
        spectra, timestamps = record_spectra(spectrometer, time_interval_seconds, integration_time_ms, total_duration_seconds, avg_background)

        print("Processing and saving data...")
        process_and_save_data(data_directory, spectra, timestamps, integration_time_ms)
        
        print("Calculating and saving normalized spectrum...")
        normalized_averaged_spectrum_filename = calculate_and_save_normalized_spectrum(spectrum_with_fiber_averaged_filename, spectrum_without_fiber_averaged_filename)
       
        print("Calculating normalized power percentage...")
        power_percentage = normalize_power()
        
        print("Calculating and saving normalized power spectrum...")
        calculate_and_save_normalized_power_spectrum(normalized_averaged_spectrum_filename, power_percentage)
       
        
    except Exception as e:
        print("An error occurred:", str(e))
    finally:
        print("Closing spectrometer...")
        close_spectrometer(spectrometer)

if __name__ == "__main__":
    main()
