# Author: Ahmad Azizan (aaaba2@cam.ac.uk)
import os
import time
import h5py
from seabreeze.spectrometers import list_devices, Spectrometer

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
    number_of_spectra = 100
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


def main():
    try:
        data_directory, background_file_path, spectrometer = initialize_data_and_spectrometer()
        time_interval_seconds, number_of_spectra, integration_time_ms, total_duration_seconds, time_background = get_measurement_settings()
        background_wavelengths, avg_background = check_and_handle_background_spectrum(background_file_path, spectrometer, time_interval_seconds, integration_time_ms, time_background)
        print("Start recording spectra...")
        spectra, timestamps = record_spectra(spectrometer, time_interval_seconds, integration_time_ms, total_duration_seconds, avg_background)
        process_and_save_data(data_directory, spectra, timestamps)
    except Exception as e:
        print("An error occurred:", str(e))
    finally:
        close_spectrometer(spectrometer)

if __name__ == "__main__":
    main()
