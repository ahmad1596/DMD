import os
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
from seabreeze.spectrometers import list_devices, Spectrometer
import serial
import threading
import concurrent.futures
from datetime import datetime

# Utility functions
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
            file.create_dataset(key, value=value)
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

def record_spectra_with_bg_subtraction(spectrometer, spectra_time_interval_seconds, spectra_number_of_spectra, spectra_integration_time_ms, avg_background):
    spectra = []
    timestamps = []
    spectrometer.integration_time_micros(spectra_integration_time_ms * 1000)
    print("Recording spectra with background subtraction...")
    start_time = time.time()
    for i in range(spectra_number_of_spectra):
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time > spectra_time_interval_seconds * spectra_number_of_spectra:
            break
        print(f"Elapsed time: {elapsed_time:.2f} seconds, Recording spectrum {i+1}/{spectra_number_of_spectra}")
        spectrum_data = spectrometer.spectrum(correct_dark_counts=True) - avg_background
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

# Temperature control functions
def setup_serial_port(com_port='COM4', baud_rate=115200):
    try:
        ser = serial.Serial(com_port, baud_rate)
        if ser.is_open:
            print(f"Serial connection to {com_port} established at {baud_rate} baud.")
            return ser
    except serial.SerialException as e:
        print(f"Error: {e}")
        return None

def get_output_file_path(base_path):
    i = 0
    while True:
        if i == 0:
            output_file_path = f"{base_path}.txt"
        else:
            output_file_path = f"{base_path}({i}).txt"
        if not os.path.exists(output_file_path):
            return output_file_path
        i += 1

def read_and_append_data(ser, target_temperature, stop_temperature, numerator, start_time, output_file_path, fiber_temperature_data, ambient_temperature_data):
    while True:
        data = ser.readline().decode('utf-8').strip()
        print(data)
        elapsed_time = (time.time() - start_time) / 60  # Convert elapsed time to minutes
        if data.startswith('T1;'):
            temperature = float(data.split(';')[2])
            fiber_temperature_data.append((elapsed_time, temperature))  # Append data to fiber list
            with open(output_file_path, 'a') as txt_file:
                txt_file.write(f"{data}, {elapsed_time:.2f}\n")

            if numerator > 0:
                if temperature >= stop_temperature:
                    print(f"Fibre Temperature has reached the stop temperature ({stop_temperature}°C).")
                    return
                if temperature >= target_temperature:
                    print(f"Fibre Temperature has reached the target temperature ({target_temperature}°C).")
                    return
            else:
                if temperature <= stop_temperature:
                    print(f"Fibre Temperature has reached the stop temperature ({stop_temperature}°C).")
                    return
                if temperature <= target_temperature:
                    print(f"Fibre Temperature has reached the target temperature ({target_temperature}°C).")
                    return

        elif data.startswith('T2;'):
            temperature = float(data.split(';')[2])
            ambient_temperature_data.append((elapsed_time, temperature))  # Append data to ambient list
            with open(output_file_path, 'a') as txt_file:
                txt_file.write(f"{data}, {elapsed_time:.2f}\n")

def define_cycle_settings():
    cycle_settings = [
        (10, 10, 20, 0),
        (1, 20, 0, 0)
    ]
    num_cycles = len(cycle_settings)
    return cycle_settings, num_cycles

def configure_cycle(ser, numerator, target_temperature, stop_temperature, start_time, output_file_path, fiber_temperature_data, ambient_temperature_data):
    print("\nPower Percentage Configuration...")
    power_percentage = int((numerator / 255) * 100)
    ser.write(f'const {numerator}\n'.encode('utf-8'))
    time.sleep(1)
    print(f"Output power percentage set to {power_percentage}%")
    print("\nTemperature Controller Configuration...")
    ser.write(f'settemp 1 {target_temperature}\n'.encode('utf-8'))
    ser.write(f'settemp 2 {target_temperature}\n'.encode('utf-8'))
    ser.write(b'start\n')
    ser.write(b'reg\n')
    read_thread = threading.Thread(target=read_and_append_data, args=(ser, target_temperature, stop_temperature, numerator, start_time, output_file_path, fiber_temperature_data, ambient_temperature_data))
    read_thread.start()
    read_thread.join()

def disconnect_serial_port(ser):
    ser.write(b'regoff\n')
    time.sleep(1)
    ser.write(b'stop\n')
    time.sleep(1)
    ser.write(b'off\n')
    ser.close()
    print("Serial connection is disconnected.")

def plot_temperature_data(fiber_temperature_data, ambient_temperature_data):
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)

    fiber_time_values, fiber_temperature_values = zip(*fiber_temperature_data)
    ambient_time_values, ambient_temperature_values = zip(*ambient_temperature_data)
    
    ax.set_xlabel("Time Elapsed (minutes)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Temperature (°C)", fontsize=14, fontweight="bold")
    ax.set_title("Temperature vs. Time", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    
    line1, = ax.plot(fiber_time_values, fiber_temperature_values, label='Fibre Temperature', 
                     marker='o', linestyle='-', color=cmap(0), markersize=2)
    line2, = ax.plot(ambient_time_values, ambient_temperature_values, label='Ambient Temperature', 
                     marker='o', linestyle='-', color=cmap(1), markersize=2)
    
    ax.legend(handles=[line1, line2], loc="center right", fontsize=10)
    plt.show()

def plot_concurrency_check(spectra_timestamps, fiber_temperature_timestamps):
    # Convert timestamps to datetime objects
    spectra_times = [datetime.fromtimestamp(ts) for ts in spectra_timestamps]
    fiber_temp_times = [datetime.fromtimestamp(ts) for ts in fiber_temperature_timestamps]
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=600)
    ax2 = ax1.twinx()

    # Plot number of spectra recorded over time
    spectra_counts = range(1, len(spectra_times) + 1)
    ax1.plot(spectra_times, spectra_counts, 'g-', label="Spectra Count")
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Number of Spectra', color='g')
    ax1.tick_params(axis='y', labelcolor='g')

    # Plot number of temperature readings over time
    temp_counts = range(1, len(fiber_temp_times) + 1)
    ax2.plot(fiber_temp_times, temp_counts, 'b-', label="Temperature Data Count")
    ax2.set_ylabel('Number of Temperature Data', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    # Format x-axis ticks as HH:MM:SS
    date_format = '%H:%M:%S'
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: datetime.utcfromtimestamp(x).strftime(date_format)))

    plt.title('Concurrency Check: Number of Data Points Recorded Over Time')
    fig.tight_layout()
    plt.show()

# Main function
def main():
    try:
        # 1. Initialize spectrometer
        print("Initializing spectrometer...")
        spectrometer = find_and_initialize_spectrometer()
        if not spectrometer:
            print("Exiting due to spectrometer initialization failure.")
            return

        # 2. Initialize temperature stage
        print("Initializing temperature stage...")
        ser = setup_serial_port()
        if ser is None:
            return
        cycle_settings, num_cycles = define_cycle_settings()
        base_output_file = r'C:\Users\DELL\Documents\optofluidics-master\optofluidics-master\Python\tempcontroller\output_start'
        output_file_path = get_output_file_path(base_output_file)
        fiber_temperature_data = []  # List to store fiber temperature data
        ambient_temperature_data = []  # List to store ambient temperature data

        # 3. Record background spectra
        print("Recording background spectra...")
        bg_time_interval_seconds, bg_number_of_spectra, bg_integration_time_ms, bg_total_duration_seconds, bg_time_background = get_background_measurement_settings()
        background_spectra, background_timestamps, avg_background = record_background_spectrum(spectrometer, bg_time_interval_seconds, bg_integration_time_ms, bg_time_background)

        # Get spectra measurement settings
        spectra_time_interval_seconds, spectra_number_of_spectra, spectra_integration_time_ms, spectra_total_duration_seconds = get_spectra_measurement_settings()

        # 5. Start concurrent tasks for recording spectra with background subtraction and temperature ramp
        with concurrent.futures.ThreadPoolExecutor() as executor:
            start_time = time.time()
            print("Starting concurrent tasks for recording spectra and temperature ramp...")
            spectra_future = executor.submit(record_spectra_with_bg_subtraction, spectrometer, spectra_time_interval_seconds, spectra_number_of_spectra, spectra_integration_time_ms, avg_background)

            for cycle in range(num_cycles):
                numerator, target_temperature, stop_temperature, wait_time = cycle_settings[cycle]
                print(f"\nCycle {cycle + 1}")
                executor.submit(configure_cycle, ser, numerator, target_temperature, stop_temperature, start_time, output_file_path, fiber_temperature_data, ambient_temperature_data)
                print(f"Waiting for {wait_time} seconds before starting the next cycle...")
                time.sleep(wait_time)  # Wait for the specified time before the next cycle

            spectra, spectra_timestamps = spectra_future.result()

        # 6. Disconnect spectrometer
        print("Disconnecting spectrometer...")
        close_spectrometer(spectrometer)

        # 7. Disconnect serial port
        print("Disconnecting serial port...")
        disconnect_serial_port(ser)

        # Extract fiber temperature timestamps
        fiber_temperature_timestamps = [timestamp for timestamp, temp in fiber_temperature_data]

        # 8. Plot temperature data
        print("Plotting temperature data...")
        plot_temperature_data(fiber_temperature_data, ambient_temperature_data)

        # 9. Plot spectra
        print("Plotting spectra...")
        for i, (wavelengths, spectrum_data) in enumerate(spectra):
            fig_title = f"Individual Spectrum {i+1}"
            plot_spectrum(wavelengths, spectrum_data, fig_title)

        # 10. Plot concurrency check
        print("Plotting concurrency check...")
        plot_concurrency_check(spectra_timestamps, fiber_temperature_timestamps)

    except Exception as e:
        print("An error occurred:", str(e))

if __name__ == "__main__":
    main()
