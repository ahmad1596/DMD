# Author: Ahmad Azizan (aaaba2@cam.ac.uk)
import paramiko
import time
import os
import h5py
from seabreeze.spectrometers import list_devices, Spectrometer
import concurrent.futures
import ctypes
from picosdk.ps3000a import ps3000a as ps
import numpy as np
import matplotlib.pyplot as plt
from picosdk.functions import adc2mV, assert_pico_ok

def open_picoscope():
    status = {}
    chandle = ctypes.c_int16()
    status["openunit"] = ps.ps3000aOpenUnit(ctypes.byref(chandle), None)
    try:
        assert_pico_ok(status["openunit"])
    except:
        powerstate = status["openunit"]
        if powerstate == 282:
            status["ChangePowerSource"] = ps.ps3000aChangePowerSource(chandle, 282)
        elif powerstate == 286:
            status["ChangePowerSource"] = ps.ps3000aChangePowerSource(chandle, 286)
        else:
            raise
        assert_pico_ok(status["ChangePowerSource"])
    return chandle, status

def configure_channel(chandle):
    status = {}
    chARange = 8
    status["setChA"] = ps.ps3000aSetChannel(chandle, 0, 1, 1, chARange, 0)
    assert_pico_ok(status["setChA"])
    return status

def configure_trigger(chandle):
    status = {}
    status["trigger"] = ps.ps3000aSetSimpleTrigger(chandle, 1, 0, 1024, 3, 0, 1000)
    assert_pico_ok(status["trigger"])
    return status

def run_capture(chandle, preTriggerSamples, postTriggerSamples, timebase, maxsamples):
    status = {}
    returnedMaxSamples = ctypes.c_int16()
    timeIntervalns = ctypes.c_float()
    status["GetTimebase"] = ps.ps3000aGetTimebase2(chandle, timebase, maxsamples, ctypes.byref(timeIntervalns), 1, ctypes.byref(returnedMaxSamples), 0)
    assert_pico_ok(status["GetTimebase"])
    cmaxSamples = ctypes.c_int32(maxsamples)
    status["runblock"] = ps.ps3000aRunBlock(chandle, preTriggerSamples, postTriggerSamples, timebase, 1, None, 0, None, None)
    assert_pico_ok(status["runblock"])
    return cmaxSamples, timeIntervalns, status

def collect_data(chandle, maxsamples):
    status = {}
    bufferAMax = (ctypes.c_int16 * maxsamples)()
    bufferAMin = (ctypes.c_int16 * maxsamples)()
    status["SetDataBuffers"] = ps.ps3000aSetDataBuffers(chandle, 0, ctypes.byref(bufferAMax), ctypes.byref(bufferAMin), maxsamples, 0, 0)
    assert_pico_ok(status["SetDataBuffers"])
    ready = ctypes.c_int16(0)
    check = ctypes.c_int16(0)
    while ready.value == check.value:
        status["isReady"] = ps.ps3000aIsReady(chandle, ctypes.byref(ready))
    overflow = (ctypes.c_int16 * 10)()
    cmaxSamples = ctypes.c_int32(maxsamples)
    status["GetValues"] = ps.ps3000aGetValues(chandle, 0, ctypes.byref(cmaxSamples), 0, 0, 0, ctypes.byref(overflow))
    assert_pico_ok(status["GetValues"])
    return bufferAMax, cmaxSamples, status


def configure_measurement(chandle, timebase, capture_duration_milliseconds):
    status = {}
    status.update(configure_channel(chandle)) 
    status["trigger"] = ps.ps3000aSetSimpleTrigger(chandle, 0, 0, 0, 0, 0, 0)  
    assert_pico_ok(status["trigger"])  
    sample_interval_ns = (timebase-2) / 0.125  
    maxsamples = int(capture_duration_milliseconds * 1e6 / sample_interval_ns)  
    preTriggerSamples = 0
    postTriggerSamples = maxsamples
    return status, sample_interval_ns, maxsamples, preTriggerSamples, postTriggerSamples

def postprocess_data(chandle, maxsamples):
    status = {}
    bufferAMax, cmaxSamples, status = collect_data(chandle, maxsamples) 
    maxADC = ctypes.c_int16()
    status["maximumValue"] = ps.ps3000aMaximumValue(chandle, ctypes.byref(maxADC))
    assert_pico_ok(status["maximumValue"])  
    return bufferAMax, maxADC, status

def convert_and_plot_data(bufferAMax_list, chARange, maxADC, capture_duration_milliseconds, block_number):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.set_dpi(200)
    ax.set_xlabel("Time (ms)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Voltage (V)", fontsize=14, fontweight="bold")
    ax.set_title(f"Voltage vs Time (Block {block_number})", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    ax.set_ylim(0, 6)  
    ax.yaxis.set_major_locator(plt.MultipleLocator(base=1))  
    time = np.linspace(0, capture_duration_milliseconds, len(bufferAMax_list[0]))  
    for bufferAMax in bufferAMax_list:
        adc2mVChAMax = adc2mV(bufferAMax, chARange, maxADC)
        voltage_values = [mv / 1000 for mv in adc2mVChAMax]  
        ax.plot(time, voltage_values)
    plt.show()

def stop_and_close(chandle):
    status = {}
    status["stop"] = ps.ps3000aStop(chandle)
    assert_pico_ok(status["stop"])
    status["close"] = ps.ps3000aCloseUnit(chandle)
    assert_pico_ok(status["close"])
    return status

def get_picoscope_settings():
    timebase = 7
    capture_duration_milliseconds = 200
    return timebase, capture_duration_milliseconds

def process_and_plot_results(tasks, chandle, maxsamples, sample_interval_ns, capture_duration_milliseconds, client):
    completed_tasks = run_tasks_concurrently(tasks)
    print_task_results(completed_tasks)
    bufferAMax, maxADC, status = postprocess_data(chandle, maxsamples)
    print(f"Sample Interval: {sample_interval_ns} ns")
    print(f"Capture Duration: {capture_duration_milliseconds:.2f} milliseconds")
    convert_and_plot_data([bufferAMax], 8, maxADC, capture_duration_milliseconds=200, block_number=1)

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

def establish_ssh_connection(hostname, port, username, password):
    client = paramiko.SSHClient()
    try:
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname, port, username, password)
        return client
    except paramiko.AuthenticationException as auth_ex:
        print("Authentication failed:", str(auth_ex))
    except paramiko.SSHException as ssh_ex:
        print("SSH connection failed:", str(ssh_ex))
    except Exception as ex:
        print("Error:", str(ex))
    return None

def setup_ssh_connection():
    hostname = "beaglebone"
    port = 22
    username = "debian"
    password = "temppwd"
    client = establish_ssh_connection(hostname, port, username, password)
    return client

def execute_command(client, command):
    stdin, stdout, stderr = client.exec_command(command)

def get_script_path(script_name):
    base_path = "/home/debian/boot-scripts-master/boot-scripts-master/device/bone/capes/DLPDLCR2000/"
    return os.path.join(base_path, script_name)

def initialise_switching(client):
    execute_command(client, f"/usr/bin/python2 {get_script_path('LEDSwitchOff.py')}")
    time.sleep(3)
    execute_command(client, get_script_path("deinitialise.sh"))
    time.sleep(3)
    execute_command(client, get_script_path("initialise.sh"))
    time.sleep(3)
    
def record_spectra_concurently(data_directory, client, spectrometer, time_interval_seconds, integration_time_ms, total_duration_seconds, avg_background):
    time.sleep(0.35)
    start_time = time.time()
    spectra, timestamps = record_spectra(spectrometer, time_interval_seconds, integration_time_ms, total_duration_seconds, avg_background)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken by record_spectra_concurently: {total_time:.2f} seconds")
    return spectra, timestamps
    
def perform_dmd_switching_concurently(client):
    start_time = time.time()
    execute_command(client,  f"/usr/bin/python2 {get_script_path('switching.py')}")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken by perform_dmd_switching_concurently: {total_time:.2f} seconds")
    
def capture_picoscope_data_concurently(chandle, timebase, maxsamples, preTriggerSamples, postTriggerSamples):
    start_time = time.time()
    status = {}
    time.sleep(0.22)
    cmaxSamples, timeIntervalns, status = run_capture(chandle, preTriggerSamples, postTriggerSamples, timebase, maxsamples)
    time.sleep(0.05)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken by capture_data_concurently: {total_time:.2f} seconds")

def run_tasks_concurrently(tasks):
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        future_to_task = {executor.submit(task['function'], *task['args']): task for task in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                task['result'] = result
            except Exception as e:
                task['result'] = e
    return tasks

def print_task_results(completed_tasks):
    for task in completed_tasks:
        if 'result' in task:
            result = task['result']
            if isinstance(result, Exception):
                print(f"An error occurred in {task['function'].__name__}: {result}")
            else:
                print(f"{task['function'].__name__} completed successfully!")
        
def close_ssh_connection(client):
    if client:
        client.exec_command("exit")
        client.close()
        print("SSH connection closed.")
        
def cleanup_resources(client, chandle, spectrometer):
    if client:
        close_ssh_connection(client)
        stop_and_close(chandle)
    if spectrometer:
        spectrometer.close()
        
def main():
    try:
        chandle, status = open_picoscope()
        timebase, capture_duration_milliseconds = get_picoscope_settings()
        status, sample_interval_ns, maxsamples, preTriggerSamples, postTriggerSamples = configure_measurement(chandle, timebase=7, capture_duration_milliseconds=200)
        data_directory, background_file_path, spectrometer = initialize_data_and_spectrometer()
        time_interval_seconds, number_of_spectra, integration_time_ms, total_duration_seconds, time_background = get_measurement_settings()
        background_wavelengths, avg_background = check_and_handle_background_spectrum(background_file_path, spectrometer, time_interval_seconds, integration_time_ms, time_background)
        client = setup_ssh_connection()
        print("SSH connection established successfully!")
        print("Start recording spectra...")
        if client:
            #initialise_switching(client)
            tasks = [
                {'function': record_spectra_concurently, 'args': (data_directory, client, spectrometer, time_interval_seconds, integration_time_ms, total_duration_seconds, avg_background)},
                {'function': perform_dmd_switching_concurently, 'args': (client,)},
                {'function': capture_picoscope_data_concurently, 'args': (chandle, timebase, maxsamples, preTriggerSamples, postTriggerSamples)}
            ]

        completed_tasks = run_tasks_concurrently(tasks)
        print_task_results(completed_tasks)
        spectra, timestamps = completed_tasks[0]['result']
        process_and_save_data(data_directory, spectra, timestamps)
        process_and_plot_results(tasks, chandle, maxsamples, sample_interval_ns, capture_duration_milliseconds, client)
         
    except Exception as e:
        print("An error occurred:", str(e))
    finally:
        cleanup_resources(client, chandle, spectrometer)

if __name__ == "__main__":
    main()
