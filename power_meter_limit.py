# Author: Ahmad Azizan (aaaba2@cam.ac.uk)
import time
import pyvisa
import matplotlib.pyplot as plt

class USBTMC:
    def __init__(self, device):
        self.rm = pyvisa.ResourceManager()
        self.device = device
        self.instrument = self.rm.open_resource(device)
    def write(self, command):
        self.instrument.write(command)
    def read(self, length=4000):
        return self.instrument.read(length)
    def query(self, command, length=4000):
        return self.instrument.query(command)
    def getName(self):
        return self.instrument.query("*IDN?")
    def sendReset(self):
        self.instrument.write("*RST")
    def close(self):
        self.instrument.close()
class PM16(USBTMC):
    def __init__(self, device):
        super().__init__(device)
        print("Current wavelength: {:.0f} nm".format(self.get_wavelength()))
    def power(self):
        """Read the power from the meter in Watts."""
        return float(self.query("Read?"))
    def stream(self, samples=None, duration=None, delay=0):
        log = []
        poll_start = time.time()
        while (samples is None and duration is None) or (samples is not None and len(log) < samples) or (duration is not None and time.time() - poll_start < duration):
            try:
                val = self.power() * 1000
                print("{:.10f} mW".format(val))
                log.append(val)
                time.sleep(delay)
            except KeyboardInterrupt:
                break
        return log
    def set_wavelength(self, wavelength):
        if not 400 <= wavelength <= 1100:
            raise ValueError("{} nm is not in [400, 1100] nm.".format(wavelength))
        self.write("SENS:CORR:WAV {}".format(wavelength))
    def get_wavelength(self):
        """Get the current wavelength of the power meter, in nm."""
        return float(self.query("SENS:CORR:WAV?"))

def measure_and_get_power_meter_limit(power_meter):
    duration = 0.01
    total_power_number = 101
    power_number = 1
    timestamps = []
    time_per_power_recording = []
    power_data_list = []
    while power_number <= total_power_number:
        print("Taking power measurement number {}...".format(power_number))
        power_start_time = time.perf_counter()
        power_data = power_meter.stream(duration=duration)
        power_end_time = time.perf_counter()
        power_data_list.extend(power_data)
        time_taken_ms = (power_end_time - power_start_time) * 1000
        time_per_power_recording.append(time_taken_ms)
        timestamps.append(time.perf_counter())
        power_number += 1
    print(time_per_power_recording)
    return time_per_power_recording

def plot_power_data(time_per_power_recording):
    time_per_power_recording = time_per_power_recording[1:]
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.set_dpi(600)
    ax.set_xlabel("Number of Power Recordings", fontsize=14, fontweight="bold")
    ax.set_ylabel("Time Taken (milliseconds)", fontsize=14, fontweight="bold")
    ax.set_title("Time Taken to Record Each Power", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    mean_time = sum(time_per_power_recording) / len(time_per_power_recording)
    median_time = sorted(time_per_power_recording)[len(time_per_power_recording) // 2]  
    std_deviation = (sum((x - mean_time) ** 2 for x in time_per_power_recording) / len(time_per_power_recording)) ** 0.5
    ax.errorbar(
        range(1, len(time_per_power_recording) + 1),
        time_per_power_recording,
        yerr=std_deviation,
        fmt='o',
        markersize=3,
        capsize=3,
        label='Time Taken',
    )
    ax.axhline(y=mean_time, color='blue', linestyle='-', label=f'Mean Time: {mean_time:.2f} ms')
    ax.axhline(y=median_time, color='green', linestyle='-', label=f'Median Time: {median_time:.2f} ms')
    ax.text(0.66, 0.80, f"Total Power Recordings: {len(time_per_power_recording)}", transform=ax.transAxes, fontsize=10, color='blue')
    ax.text(0.66, 0.76, f"Total Time Taken: {sum(time_per_power_recording):.2f} ms", transform=ax.transAxes, fontsize=10, color='blue')
    ax.legend(loc='upper right', fontsize=10)
    plt.show()

    
def main():
    instrument_visa_resource = "USB0::0x1313::0x807B::190704312::INSTR"
    power_meter = PM16(instrument_visa_resource)
    try:
        print("Instrument identification:", power_meter.getName())
        time_per_power_recording = measure_and_get_power_meter_limit(power_meter)
        plot_power_data(time_per_power_recording)
    except Exception as e:
        print("An error occurred:", e)
    finally:
        power_meter.close()
        print("Connection closed.")

if __name__ == "__main__":
    main()
