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
    def __init__(self, device, initial_wavelength=785):
        super().__init__(device)
        self.set_wavelengths([initial_wavelength])
        print("Current wavelength: {:.0f} nm".format(initial_wavelength))

    def power(self):
        """Read the power from the meter in Watts."""
        return float(self.query("Read?"))

    def stream(self, duration=None, delay=0.5, num_readings=10):
        log = []
        poll_start = time.time()
        current_wavelength = self.get_wavelength()
        print("Streaming {} power readings for {} second each at {} nm...".format(num_readings, duration, current_wavelength))

        while (duration is None or time.time() - poll_start < duration):
            try:
                for _ in range(num_readings):
                    val = self.power() * 1e6  # Convert to uW
                    print("{:.10f} uW".format(val))
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

def average_power_over_time(power_meter, duration=1, delay=0.5):
    log = []
    poll_start = time.time()
    current_wavelength = power_meter.get_wavelength()
    print("Streaming power readings for {} second each at {} nm...".format(duration, current_wavelength))

    while (duration is None or time.time() - poll_start < duration):
        try:
            val = power_meter.power() * 1e6  # Convert to uW
            print("{:.10f} uW".format(val))
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

def measure_and_get_avg_power(power_meter, num_readings=10):
    duration = 1
    power_data = power_meter.stream(duration=duration, num_readings=num_readings)
    avg_power_mW = sum(power_data) / len(power_data)
    return avg_power_mW

def user_press_enter(message):
    input("Press Enter when ready to take {} reading...".format(message))

def collect_measurement_data(power_meter, num_readings=10):
    start_wavelength = power_meter.get_wavelength()
    end_wavelength = start_wavelength + 20
    wavelength_step = 2

    wavelengths = list(range(int(start_wavelength), int(end_wavelength) + 1, wavelength_step))
    incident_bg_powers = []
    incident_raw_powers = []
    reflected_bg_powers = []
    reflected_raw_powers = []
    corrected_incident_powers = []
    corrected_reflected_powers = []
    corrected_transmitted_powers = []

    for wavelength in wavelengths:
        power_meter.set_wavelengths([wavelength])
        time.sleep(1.0)
        if wavelength == int(start_wavelength):
            user_press_enter("Incident Background")
        incident_bg_power = measure_and_get_avg_power(power_meter, num_readings)
        incident_bg_powers.append(incident_bg_power)

    power_meter.set_wavelengths(wavelengths)
    time.sleep(1.0)
    user_press_enter("Incident Raw")
    for wavelength in wavelengths:
        print(f"Incident Raw at {wavelength} nm")
        incident_raw_power = measure_and_get_avg_power(power_meter, num_readings)
        incident_raw_powers.append(incident_raw_power)

    power_meter.set_wavelengths(wavelengths)
    time.sleep(1.0)
    user_press_enter("Reflected background")
    for wavelength in wavelengths:
        print(f"Reflected Background at {wavelength} nm")
        reflected_bg_power = measure_and_get_avg_power(power_meter, num_readings)
        reflected_bg_powers.append(reflected_bg_power)

    power_meter.set_wavelengths(wavelengths)
    time.sleep(1.0)
    user_press_enter("Reflected raw")
    for wavelength in wavelengths:
        print(f"Reflected Raw at {wavelength} nm")
        reflected_raw_power = measure_and_get_avg_power(power_meter, num_readings)
        reflected_raw_powers.append(reflected_raw_power)

    power_meter.set_wavelengths(wavelengths)
    time.sleep(1.0)
    print("Processing Data...")
    for wavelength in wavelengths:
        corrected_incident_power = abs(incident_raw_power - incident_bg_power)
        corrected_incident_powers.append(corrected_incident_power)
        corrected_reflected_power = abs(reflected_raw_power - reflected_bg_power)
        corrected_reflected_powers.append(corrected_reflected_power)
        corrected_transmitted_power = abs(corrected_incident_power - corrected_reflected_power)
        corrected_transmitted_powers.append(corrected_transmitted_power)

    return (
        incident_bg_powers,
        incident_raw_powers,
        reflected_bg_powers,
        reflected_raw_powers,
        corrected_incident_powers,
        corrected_reflected_powers,
        corrected_transmitted_powers,
        wavelengths
    )

def plot_selected_measurements(wavelengths, incident_bg_powers, incident_raw_powers, reflected_bg_powers, reflected_raw_powers, corrected_incident_powers, corrected_reflected_powers, corrected_transmitted_powers):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)
    fig.set_dpi(600)
    ax.plot(wavelengths, incident_bg_powers, marker='o', label="Background Incident")
    ax.plot(wavelengths, reflected_bg_powers, marker='o', label="Background Transmitted")
    ax.plot(wavelengths, incident_raw_powers, marker='o', label="Uncorrected Incident")
    ax.plot(wavelengths, reflected_raw_powers, marker='o', label="Uncorrected Transmitted")
    ax.plot(wavelengths, corrected_incident_powers, marker='o', label="Corrected Incident")
    ax.plot(wavelengths, corrected_reflected_powers, marker='o', label="Corrected Transmitted")
    ax.plot(wavelengths, corrected_transmitted_powers, marker='o', label="Corrected Reflected")
    ax.set_xlabel("Wavelength (nm)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Power (uW)", fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    ax.legend(loc="center right", fontsize=8)
    plt.title("Selected Power Measurements", fontsize=16, fontweight="bold")
    plt.show()

def plot_normalized_measurements(wavelengths, corrected_incident_powers, corrected_reflected_powers, corrected_transmitted_powers):
    normalized_incident = [(power / incident) * 100 for incident, power in zip(corrected_incident_powers, corrected_incident_powers)]
    normalized_reflected = [(reflected / incident) * 100 for incident, reflected in zip(corrected_incident_powers, corrected_reflected_powers)]
    normalized_transmitted = [(transmitted / incident) * 100 for incident, transmitted in zip(corrected_incident_powers, corrected_transmitted_powers)]
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)
    fig.set_dpi(600)
    ax.plot(wavelengths, normalized_incident, marker='o', label="Normalized Corrected Incident")
    ax.plot(wavelengths, normalized_reflected, marker='o', label="Normalized Corrected Transmitted")
    ax.plot(wavelengths, normalized_transmitted, marker='o', label="Normalized Corrected Reflected")
    ax.set_xlabel("Wavelength (nm)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Normalized Power (%)", fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    ax.legend(loc="center right", fontsize=8)
    plt.title("Normalized Power Measurements", fontsize=16, fontweight="bold")
    plt.show()

def main():
    instrument_visa_resource = "USB0::0x1313::0x807B::190704312::INSTR"
    initial_wavelength = 520
    power_meter = PM16(instrument_visa_resource, initial_wavelength)
    try:
        print("Instrument identification:", power_meter.getName())
        (
            incident_bg_powers,
            incident_raw_powers,
            reflected_bg_powers,
            reflected_raw_powers,
            corrected_incident_powers,
            corrected_reflected_powers,
            corrected_transmitted_powers,
            wavelengths
        ) = collect_measurement_data(power_meter, num_readings=10)

        plot_selected_measurements(
            wavelengths,
            incident_bg_powers,
            incident_raw_powers,
            reflected_bg_powers,
            reflected_raw_powers,
            corrected_incident_powers,
            corrected_reflected_powers,
            corrected_transmitted_powers,
        )
        plot_normalized_measurements(
            wavelengths,
            corrected_incident_powers,
            corrected_reflected_powers,
            corrected_transmitted_powers,
        )

        print("\nAverage Power Values:")
        print("  Incident Background: {:.4f} uW".format(sum(incident_bg_powers) / len(incident_bg_powers)))
        print("  Incident Raw: {:.4f} uW".format(sum(incident_raw_powers) / len(incident_raw_powers)))
        print("  Transmitted Background: {:.4f} uW".format(sum(reflected_bg_powers) / len(reflected_bg_powers)))
        print("  Transmitted Raw: {:.4f} uW".format(sum(reflected_raw_powers) / len(reflected_raw_powers)))
        print("  Corrected Incident: {:.4f} uW".format(sum(corrected_incident_powers) / len(corrected_incident_powers)))
        print("  Corrected Transmitted: {:.4f} uW".format(sum(corrected_reflected_powers) / len(corrected_reflected_powers)))
        print("  Corrected Reflected: {:.4f} uW".format(sum(corrected_transmitted_powers) / len(corrected_transmitted_powers)))

    except Exception as e:
        print("An error occurred:", e)
    finally:
        power_meter.close()
        print("Connection closed.")

if __name__ == "__main__":
    main()
