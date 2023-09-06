import serial
from NKTP_DLL import registerWriteU8, RegisterResultTypes
import time

class SuperKCompactControl:
    def __init__(self, module_type='74', serial_port=None):
        self.module_type = module_type
        self.serial_port = serial_port
        self.timeout = 0.05

    def connect(self):
        if self.serial_port is None:
            serial_port = 'COM3'
            print(f"Attempting to open {serial_port}...")
            try:
                self.serial_port = serial.Serial(serial_port, baudrate=115200, timeout=self.timeout)
                print(f"Successfully opened {serial_port}")
                response = self.send_command(f'74{self.module_type}0000')
                if response and response.startswith('74'):
                    print(f'Successfully connected to COM port {serial_port}')
                    print('SuperK Compact connected')
                else:
                    self.serial_port.close()
                    self.serial_port = None
            except Exception as e:
                print(f'Error connecting to COM port {serial_port}: {str(e)}')

    def send_command(self, command):
        if self.serial_port:
            self.serial_port.write(bytes.fromhex(command))
            response = self.serial_port.read(128).hex()
            return response
        return None

    def set_supply_voltage(self, voltage_mV):
        result = registerWriteU8('COM3', 1, 0x1A, voltage_mV, -1)
        print('Setting supply voltage:', RegisterResultTypes(result))

    def set_heat_sink_temperature(self, temperature_tenths_C):
        result = registerWriteU8('COM3', 1, 0x1B, temperature_tenths_C, -1)
        print('Setting heat sink temperature:', RegisterResultTypes(result))

    def set_trig_level(self, level_mV):
        result = registerWriteU8('COM3', 1, 0x24, level_mV, -1)
        print('Setting trig level:', RegisterResultTypes(result))

    def set_display_backlight(self, level_percent):
        result = registerWriteU8('COM3', 1, 0x26, level_percent, -1)
        print('Setting display backlight level:', RegisterResultTypes(result))

    def set_emission_status(self, status):
        result = registerWriteU8('COM3', 1, 0x30, status, -1)
        if status == 1:
            print('Setting emission ON:', RegisterResultTypes(result))
        elif status == 0:
            print('Setting emission OFF:', RegisterResultTypes(result))
        else:
            print('Invalid emission status:', status)

    def set_trig_mode(self, mode):
        result = registerWriteU8('COM3', 1, 0x31, mode, -1)
        print('Setting trig mode:', RegisterResultTypes(result))

    def set_interlock(self, interlock_value):
        result = registerWriteU8('COM3', 1, 0x32, interlock_value, -1)
        print('Setting interlock:', RegisterResultTypes(result))

    def set_internal_pulse_frequency(self, frequency_Hz):
        result = registerWriteU8('COM3', 1, 0x33, frequency_Hz, -1)
        print('Setting internal pulse frequency:', RegisterResultTypes(result))

    def set_burst_pulses(self, pulses):
        result = registerWriteU8('COM3', 1, 0x34, pulses, -1)
        print('Setting burst pulses:', RegisterResultTypes(result))

    def set_watchdog_interval(self, interval_seconds):
        result = registerWriteU8('COM3', 1, 0x35, interval_seconds, -1)
        print('Setting watchdog interval:', RegisterResultTypes(result))

    def set_internal_pulse_frequency_limit(self, limit_Hz):
        result = registerWriteU8('COM3', 1, 0x36, limit_Hz, -1)
        print('Setting internal pulse frequency limit:', RegisterResultTypes(result))

    def set_power_level(self, power_percent):
        result = registerWriteU8('COM3', 1, 0x3E, power_percent, -1)
        print('Setting power level:', RegisterResultTypes(result))

    def get_status_bits(self):
        result = registerWriteU8('COM3', 1, 0x66, 0, -1)
        print('Getting status bits:', RegisterResultTypes(result))

    def get_optical_pulse_frequency(self):
        result = registerWriteU8('COM3', 1, 0x71, 0, -1)
        print('Getting optical pulse frequency:', RegisterResultTypes(result))

    def get_actual_internal_trig_frequency(self):
        result = registerWriteU8('COM3', 1, 0x75, 0, -1)
        print('Getting actual internal trig frequency:', RegisterResultTypes(result))

    def get_display_text(self):
        result = registerWriteU8('COM3', 1, 0x78, 0, -1)
        print('Getting display text:', RegisterResultTypes(result))

    def get_power_level(self):
        result = registerWriteU8('COM3', 1, 0x7A, 0, -1)
        print('Getting power level:', RegisterResultTypes(result))

    def get_user_area(self):
        result = registerWriteU8('COM3', 1, 0x8D, 0, -1)
        print('Getting user area:', RegisterResultTypes(result))

    def disconnect(self):
        if self.serial_port:
            self.serial_port.close()
            self.serial_port = None
            print('SuperK Compact disconnected')
def main():
    laser = SuperKCompactControl()
    laser.connect()
    laser.set_emission_status(1)
    time.sleep(2)
    laser.set_emission_status(0)
    laser.set_supply_voltage(3300) 
    laser.set_heat_sink_temperature(250) 
    laser.set_trig_level(2000)
    laser.set_display_backlight(50)
    laser.set_trig_mode(1)
    laser.set_interlock(2)
    laser.set_internal_pulse_frequency(1000)
    laser.set_burst_pulses(10)
    laser.set_watchdog_interval(60)
    laser.set_internal_pulse_frequency_limit(5000)
    laser.set_power_level(75)

    laser.get_status_bits()
    laser.get_optical_pulse_frequency() 
    laser.get_actual_internal_trig_frequency()
    laser.get_display_text() 
    laser.get_power_level()
    laser.get_user_area()

    laser.disconnect()

if __name__ == "__main__":
    main()
