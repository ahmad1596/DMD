# Author: Ahmad Azizan (aaaba2@cam.ac.uk)
import ctypes
from picosdk.ps3000a import ps3000a as ps
import numpy as np
import matplotlib.pyplot as plt
from picosdk.functions import adc2mV, assert_pico_ok
import time

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

def capture_data(chandle, timebase, maxsamples, preTriggerSamples, postTriggerSamples):
    start_time = time.time()
    status = {}
    cmaxSamples, timeIntervalns, status = run_capture(chandle, preTriggerSamples, postTriggerSamples, timebase, maxsamples)
    time.sleep(0.05)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken by capture_data: {total_time:.2f} seconds")
    return cmaxSamples, status

def postprocess_data(chandle, maxsamples):
    status = {}
    bufferAMax, cmaxSamples, status = collect_data(chandle, maxsamples) 
    maxADC = ctypes.c_int16()
    status["maximumValue"] = ps.ps3000aMaximumValue(chandle, ctypes.byref(maxADC))
    assert_pico_ok(status["maximumValue"])  
    return bufferAMax, maxADC, status

def convert_and_plot_data(bufferAMax_list, chARange, maxADC, capture_duration_milliseconds, block_number):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.set_dpi(600)
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

def main():
    try:
        chandle, status = open_picoscope()
        capture_duration_milliseconds = 200
        status, sample_interval_ns, maxsamples, preTriggerSamples, postTriggerSamples = configure_measurement(chandle, timebase=7, capture_duration_milliseconds=200)
        cmaxSamples, status = capture_data(chandle, timebase=7, maxsamples=maxsamples, preTriggerSamples=preTriggerSamples, postTriggerSamples=postTriggerSamples)
        bufferAMax, maxADC, status = postprocess_data(chandle, maxsamples)
        print(f"Sample Interval: {sample_interval_ns} ns")
        print(f"Capture Duration: {capture_duration_milliseconds:.2f} milliseconds")
        convert_and_plot_data([bufferAMax], 8, maxADC, capture_duration_milliseconds=200, block_number=1)  
        status.update(stop_and_close(chandle))
        print(status)   
    except Exception as e:
        print("An error occurred:", str(e))  
        
if __name__ == "__main__":
    main()

