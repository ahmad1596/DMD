import os
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
from seabreeze.spectrometers import list_devices, Spectrometer
import serial
import threading
import concurrent.futures
from datetime import datetime, timedelta
import matplotlib.dates as mdates

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
        (255, 24, 24, 0),  # (numerator, target_temperature, stop_temperature, wait_time)
        (-255, 20, 23, 0),
        (255, 24, 24, 0),
        (-255, 20, 23, 0)
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
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.set_dpi(600)

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
    bg_time_interval_seconds = 0.5
    bg_number_of_spectra = 2
    bg_integration_time_ms = 10
    bg_total_duration_seconds = bg_number_of_spectra * (bg_time_interval_seconds + bg_integration_time_ms / 1000)
    bg_time_background = bg_total_duration_seconds
    print(f"Background Measurement settings: Time Interval = {bg_time_interval_seconds} seconds, Number of Spectra = {bg_number_of_spectra}, Integration Time = {bg_integration_time_ms} ms")
    print(f"Total measurement duration for background: {bg_total_duration_seconds:.2f} seconds")
    return bg_time_interval_seconds, bg_number_of_spectra, bg_integration_time_ms, bg_total_duration_seconds, bg_time_background

def get_spectra_measurement_settings():
    spectra_time_interval_seconds = 1
    spectra_number_of_spectra = 50
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
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity (a.u)')
    ax.set_title(fig_title)
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
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

def plot_concurrency_check(spectra_timestamps, fiber_temperature_timestamps):
    # Convert timestamps to datetime objects
    spectra_times = [datetime.fromtimestamp(ts) for ts in spectra_timestamps]
    fiber_temp_times = [datetime.fromtimestamp(ts) for ts in fiber_temperature_timestamps]
    # Sort timestamps
    spectra_times.sort()
    fiber_temp_times.sort()
    # Calculate time intervals for spectra (in seconds)
    if spectra_times:
        spectra_time_intervals = [(t - spectra_times[0]).total_seconds() for t in spectra_times]
        spectra_counts = np.arange(1, len(spectra_times) + 1)
    else:
        spectra_time_intervals = []
        spectra_counts = []
    # Calculate time intervals for temperature data (in seconds)
    if fiber_temp_times:
        fiber_temp_intervals = [(t - fiber_temp_times[0]).total_seconds() for t in fiber_temp_times]
        temp_counts = np.arange(1, len(fiber_temp_times) + 1)
    else:
        fiber_temp_intervals = []
        temp_counts = []
    
    # Convert fiber_temp_intervals from minutes to seconds
    fiber_temp_intervals = [interval * 60 for interval in fiber_temp_intervals]

    # Get the current time as the start time
    start_time = datetime.now()
    time_formatter = mdates.DateFormatter('%H:%M:%S')

    # Plotting
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=600)
    ax2 = ax1.twinx()
    
    # Plot number of temperature readings over time (in seconds)
    ax1.plot([start_time + timedelta(seconds=interval) for interval in fiber_temp_intervals], temp_counts, 'b-', label="Temperature Data Count")
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Number of Temperature Data', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.xaxis.set_major_formatter(time_formatter)
    ax1.grid(color="gray", linestyle="--", linewidth=0.5)
    
    # Plot number of spectra recorded over time (in seconds)
    ax2.plot([start_time + timedelta(seconds=interval) for interval in spectra_time_intervals], spectra_counts, 'g-', label="Spectra Count")
    ax2.set_ylabel('Number of Spectra', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.grid(color="gray", linestyle="--", linewidth=0.5)
    
    plt.title('Concurrency Check: Number of Data Points Recorded Over Time')
    fig.tight_layout()
    plt.show()

def plot_temperature_vs_time(fiber_temperature_data, ambient_temperature_data):
    # Extract time and temperature data
    fiber_time_values, fiber_temperature_values = zip(*fiber_temperature_data)
    ambient_time_values, ambient_temperature_values = zip(*ambient_temperature_data)
    # Convert time values to datetime objects
    fiber_times = [datetime.fromtimestamp(ts) for ts in fiber_time_values]
    ambient_times = [datetime.fromtimestamp(ts) for ts in ambient_time_values]
    # Sort the data points by time
    fiber_sorted_indices = sorted(range(len(fiber_times)), key=lambda k: fiber_times[k])
    ambient_sorted_indices = sorted(range(len(ambient_times)), key=lambda k: ambient_times[k])
    sorted_fiber_times = [fiber_times[i] for i in fiber_sorted_indices]
    sorted_fiber_temperatures = [fiber_temperature_values[i] for i in fiber_sorted_indices]
    sorted_ambient_times = [ambient_times[i] for i in ambient_sorted_indices]
    sorted_ambient_temperatures = [ambient_temperature_values[i] for i in ambient_sorted_indices]
    # Get the current time as the start time
    start_time = datetime.now()
    time_formatter = mdates.DateFormatter('%H:%M:%S')
    # Calculate time intervals in seconds
    fiber_time_intervals = [(t - sorted_fiber_times[0]).total_seconds() for t in sorted_fiber_times]
    ambient_time_intervals = [(t - sorted_ambient_times[0]).total_seconds() for t in sorted_ambient_times]
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    ax.plot([start_time + timedelta(seconds=interval) for interval in fiber_time_intervals], sorted_fiber_temperatures, 'b-', label='Fiber Temperature')
    ax.plot([start_time + timedelta(seconds=interval) for interval in ambient_time_intervals], sorted_ambient_temperatures, 'g-', label='Ambient Temperature')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature vs. Time')
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    ax.legend()
    # Customize x-axis formatting
    ax.xaxis.set_major_formatter(time_formatter)
    plt.tight_layout()
    plt.show()

def plot_max_intensity_vs_time(spectra, spectra_timestamps):
    max_intensities = []
    for wavelengths, spectrum_data in spectra:
        max_intensity = np.max(spectrum_data[1])
        max_intensities.append(max_intensity)
    # Convert timestamps to datetime objects
    spectra_times = [datetime.fromtimestamp(ts) for ts in spectra_timestamps]
    # Sort timestamps
    spectra_times.sort()
    spectra_time_intervals = [(t - spectra_times[0]).total_seconds() for t in spectra_times]
    # Get the current time as the start time
    start_time = datetime.now()
    time_formatter = mdates.DateFormatter('%H:%M:%S')
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    # Plot number of spectra recorded over time (in seconds)
    ax.plot([start_time + timedelta(seconds=interval) for interval in spectra_time_intervals], max_intensities, 'b-', label="Max Intensity")
    ax.set_xlabel('Time')
    ax.set_ylabel('Max Intensity (a.u.)')
    ax.tick_params(axis='y')
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    ax.legend()
    ax.xaxis.set_major_formatter(time_formatter)
    plt.title('Max Intensity vs. Time')
    fig.tight_layout()
    plt.show()
    
def plot_temperature_and_max_intensity_vs_time(fiber_temperature_data)
    
        
def main():
    try:
        spectra_integration_time_ms = 10
        print("Initializing data directory and spectrometer...")
        data_directory, spectrometer = initialize_data_and_spectrometer(spectra_integration_time_ms)
        if not spectrometer:
            print("Exiting due to spectrometer initialization failure.")
            return
        print("Recording background spectrum...")
        bg_time_interval_seconds, bg_number_of_spectra, bg_integration_time_ms, bg_total_duration_seconds, bg_time_background = get_background_measurement_settings()
        background_spectra, background_timestamps, avg_background = record_background_spectrum(spectrometer, bg_time_interval_seconds, bg_integration_time_ms, bg_time_background)
        
        ser = setup_serial_port()
        if ser is None:
            return
        cycle_settings, num_cycles = define_cycle_settings()
        base_output_file = r'C:\Users\DELL\Documents\optofluidics-master\optofluidics-master\Python\tempcontroller\output_start'
        output_file_path = get_output_file_path(base_output_file)
        fiber_temperature_data = []  # List to store fiber temperature data
        ambient_temperature_data = []  # List to store ambient temperature data
        
        spectra_time_interval_seconds, spectra_number_of_spectra, spectra_integration_time_ms, spectra_total_duration_seconds = get_spectra_measurement_settings()
        
        print("Press Enter when ready to start recording spectra and ramping up temperature...")
        input()  # Wait for user to press Enter
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            print("Starting concurrent tasks for recording spectra and temperature ramp...")
            start_time = time.time()
            spectra_future = executor.submit(record_spectra, spectrometer, spectra_time_interval_seconds, spectra_number_of_spectra, spectra_integration_time_ms, spectra_total_duration_seconds)
            
            for cycle in range(num_cycles):
                numerator, target_temperature, stop_temperature, wait_time = cycle_settings[cycle]
                print(f"\nCycle {cycle + 1}")
                configure_cycle(ser, numerator, target_temperature, stop_temperature, start_time, output_file_path, fiber_temperature_data, ambient_temperature_data)
                print(f"Waiting for {wait_time} seconds before starting the next cycle...")
                time.sleep(wait_time)  # Wait for the specified time before the next cycle
                
            spectra, spectra_timestamps = spectra_future.result()
        
        print("Disconnecting spectrometer...")
        close_spectrometer(spectrometer)

        print("Disconnecting serial port...")
        disconnect_serial_port(ser)

        #fiber_temperature_timestamps = [timestamp for timestamp, temp in fiber_temperature_data]

        #print("Plotting temperature data...")
        #plot_temperature_data(fiber_temperature_data, ambient_temperature_data)

        # Uncomment to plot individual spectra if needed
        # print("Plotting spectra...")
        # for i, (wavelengths, spectrum_data) in enumerate(spectra):
        #     fig_title = f"Individual Spectrum {i+1}"
        #     plot_spectrum(wavelengths, spectrum_data, fig_title)
        
        #print("Plotting concurrency check...")
        #plot_concurrency_check(spectra_timestamps, fiber_temperature_timestamps)
        
        #print("Plotting temperature data...")
        #plot_temperature_vs_time(fiber_temperature_data, ambient_temperature_data)
        
        #print("Plotting max intensity data...")
        #plot_max_intensity_vs_time(spectra, spectra_timestamps)
        
        #print("Plotting temperature and max intensity data...")
        
    except Exception as e:
        print("An error occurred:", str(e))

if __name__ == "__main__":
    main()
