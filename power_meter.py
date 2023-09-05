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
    def stream(self, samples=None, duration=None, delay=0.5):
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
    def set_wavelengths(self, wavelengths):
        for wavelength in wavelengths:
            if not 400 <= wavelength <= 1100:
                raise ValueError("{} nm is not in [400, 1100] nm.".format(wavelength))
            self.write("SENS:CORR:WAV {}".format(wavelength))
    def get_wavelength(self):
        """Get the current wavelength of the power meter, in nm."""
        return float(self.query("SENS:CORR:WAV?"))

def measure_and_get_avg_power(power_meter):
    duration = 0.01 
    print("Streaming power readings for {} seconds...".format(duration))
    power_data = power_meter.stream(duration=duration)
    avg_power_mW = sum(power_data) / len(power_data) 
    return avg_power_mW

def user_press_enter(message):
    input("Press Enter when ready to take {} reading...".format(message))

def collect_measurement_data(power_meter, wavelengths):
    incident_bg_powers = []
    incident_actual_powers = []
    reflected_or_transmitted_bg_powers = []
    reflected_or_transmitted_actual_powers = []
    transmitted_or_reflected_actual_powers = []
    power_meter.set_wavelengths([450])
    time.sleep(0.5)
    user_press_enter("Incident Background")
    for wavelength in wavelengths:
        print(f"incident_bg_powers at {wavelength} nm")
        incident_bg_power = measure_and_get_avg_power(power_meter)
        incident_bg_powers.append(incident_bg_power)
    power_meter.set_wavelengths([450])
    time.sleep(0.5) 
    user_press_enter("Incident Actual")
    for wavelength in wavelengths:
        print(f"incident_actual_powers at {wavelength} nm")
        incident_actual_power = measure_and_get_avg_power(power_meter)
        incident_actual_powers.append(incident_actual_power)
    power_meter.set_wavelengths([450])
    time.sleep(0.5) 
    user_press_enter("Reflected or Transmitted background")
    for wavelength in wavelengths:
        print(f"reflected_or_transmitted_bg_powers at {wavelength} nm")
        reflected_or_transmitted_bg_power = measure_and_get_avg_power(power_meter)
        reflected_or_transmitted_bg_powers.append(reflected_or_transmitted_bg_power)
    power_meter.set_wavelengths([450])
    time.sleep(0.5)
    user_press_enter("Reflected or Transmitted actual")
    for wavelength in wavelengths:
        print(f"reflected_or_transmitted_actual_powers at {wavelength} nm")
        reflected_or_transmitted_actual_power = measure_and_get_avg_power(power_meter)
        reflected_or_transmitted_actual_powers.append(reflected_or_transmitted_actual_power)
        corrected_incident_power = abs(incident_actual_power - incident_bg_power)
        corrected_reflected_or_transmitted_power = abs(reflected_or_transmitted_actual_power - reflected_or_transmitted_bg_power)
        transmitted_or_reflected_actual_power = abs(corrected_incident_power - corrected_reflected_or_transmitted_power)
        transmitted_or_reflected_actual_powers.append(transmitted_or_reflected_actual_power)
    return (
        incident_bg_powers,
        incident_actual_powers,
        reflected_or_transmitted_bg_powers,
        reflected_or_transmitted_actual_powers,
        transmitted_or_reflected_actual_powers
    )
def plot_selected_measurements(wavelengths, incident_bg_powers, incident_actual_powers, reflected_or_transmitted_bg_powers, reflected_or_transmitted_actual_powers, transmitted_or_reflected_actual_powers):
    fig, ax = plt.subplots()    
    fig.set_size_inches(10, 8)
    fig.set_dpi(600)
    ax.plot(wavelengths, incident_bg_powers, marker='o', label="Background Incident")
    ax.plot(wavelengths,reflected_or_transmitted_bg_powers, marker='o', label="Background Reflected or Transmitted")
    ax.plot(wavelengths, incident_actual_powers, marker='o', label="Corrected Incident")
    ax.plot(wavelengths, reflected_or_transmitted_actual_powers, marker='o', label="Corrected Reflected or Transmitted")
    ax.set_xlabel("Wavelength (nm)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Power (mW)", fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=8)
    plt.title("Selected Power Measurements", fontsize=16, fontweight="bold")
    plt.show()

def plot_normalized_measurements(wavelengths, incident_actual_powers, reflected_or_transmitted_actual_powers, transmitted_or_reflected_actual_powers):
    normalized_incident = [(power / incident) * 100 for incident, power in zip(incident_actual_powers, incident_actual_powers)]
    normalized_reflected_or_transmitted = [(reflected_or_transmitted / incident) * 100 for incident, reflected_or_transmitted in zip(incident_actual_powers, reflected_or_transmitted_actual_powers)]
    normalized_transmitted_or_reflected = [(transmitted_or_reflected / incident) * 100 for incident, transmitted_or_reflected in zip(incident_actual_powers, transmitted_or_reflected_actual_powers)]
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)
    fig.set_dpi(600)
    ax.plot(wavelengths, normalized_incident, marker='o', label="Normalized Corrected Incident")
    ax.plot(wavelengths, normalized_reflected_or_transmitted, marker='o', label="Normalized Corrected Reflected or Transmitted")    
    ax.plot(wavelengths, normalized_transmitted_or_reflected, marker='o', label="Normalized Transmitted or Reflected")
    ax.set_xlabel("Wavelength (nm)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Normalized Power (%)", fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=8)
    plt.title("Normalized Power Measurements", fontsize=16, fontweight="bold")
    plt.show()

def main():
    instrument_visa_resource = "USB0::0x1313::0x807B::190704312::INSTR"
    power_meter = PM16(instrument_visa_resource)
    try:
        print("Instrument identification:", power_meter.getName())
        wavelengths = list(range(450, 461, 5))
        (
            incident_bg_powers,
            incident_actual_powers,
            reflected_or_transmitted_bg_powers,
            reflected_or_transmitted_actual_powers,
            transmitted_or_reflected_actual_powers,
        ) = collect_measurement_data(power_meter, wavelengths)
        
        plot_selected_measurements(
            wavelengths,
            incident_bg_powers,
            incident_actual_powers,
            reflected_or_transmitted_bg_powers,
            reflected_or_transmitted_actual_powers,
            transmitted_or_reflected_actual_powers,
        )
        plot_normalized_measurements(
            wavelengths,
            incident_actual_powers,
            reflected_or_transmitted_actual_powers,
            transmitted_or_reflected_actual_powers,
        )
    except Exception as e:
        print("An error occurred:", e)
    finally:
        power_meter.close()
        print("Connection closed.")

if __name__ == "__main__":
    main()

