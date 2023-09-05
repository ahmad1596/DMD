# Author: Ahmad Azizan (aaaba2@cam.ac.uk)
import paramiko
import time
import os
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

def capture_picoscope_data_concurently(chandle, timebase, maxsamples, preTriggerSamples, postTriggerSamples):
    start_time = time.time()
    status = {}
    time.sleep(0.22)
    cmaxSamples, timeIntervalns, status = run_capture(chandle, preTriggerSamples, postTriggerSamples, timebase, maxsamples)
    time.sleep(0.05)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken by capture_data_concurently: {total_time:.2f} seconds")
    return cmaxSamples, status

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
    execute_command(client, get_script_path("deinitialise.sh"))
    time.sleep(3)
    execute_command(client, get_script_path("initialise.sh"))
    time.sleep(3)
    
def perform_dmd_switching_concurently(client):
    start_time_2 = time.time()
    execute_command(client,  f"/usr/bin/python2 {get_script_path('switching.py')}")
    end_time_2 = time.time()
    total_time = end_time_2 - start_time_2
    print(f"Total time taken by perform_dmd_switching_concurently: {total_time:.2f} seconds")


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
        
def cleanup_resources(client, chandle):
    if client:
        close_ssh_connection(client)
        stop_and_close(chandle)
            
def process_and_plot_results(tasks, chandle, maxsamples, sample_interval_ns, capture_duration_milliseconds, client):
    completed_tasks = run_tasks_concurrently(tasks)
    print_task_results(completed_tasks)
    bufferAMax, maxADC, status = postprocess_data(chandle, maxsamples)
    print(f"Sample Interval: {sample_interval_ns} ns")
    print(f"Capture Duration: {capture_duration_milliseconds:.2f} milliseconds")
    convert_and_plot_data([bufferAMax], 8, maxADC, capture_duration_milliseconds=200, block_number=1)
    
def main():
    try:
        chandle, status = open_picoscope()
        timebase, capture_duration_milliseconds = get_picoscope_settings()
        status, sample_interval_ns, maxsamples, preTriggerSamples, postTriggerSamples = configure_measurement(chandle, timebase=7, capture_duration_milliseconds=200)
        client = setup_ssh_connection()
        print("SSH connection established successfully!")
        if client:
            tasks = [
                {'function': capture_picoscope_data_concurently, 'args': (chandle, timebase, maxsamples, preTriggerSamples, postTriggerSamples)},
                {'function': perform_dmd_switching_concurently, 'args': (client,)}
            ]
        process_and_plot_results(tasks, chandle, maxsamples, sample_interval_ns, capture_duration_milliseconds, client)
    except Exception as e:
        print("An error occurred:", str(e))
    finally:
        cleanup_resources(client, chandle)
if __name__ == "__main__":
    main()

