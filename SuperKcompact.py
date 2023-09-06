import serial
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

    def disconnect(self):
        if self.serial_port:
            self.serial_port.close()
            self.serial_port = None
            print('SuperK Compact disconnected')

    def set_timeout(self, timeout):
        self.timeout = timeout

    def get_timeout(self):
        return self.timeout

    def send_command(self, command):
        if self.serial_port:
            self.serial_port.write(bytes.fromhex(command))
            response = self.serial_port.read(128).hex()
            return response
        return None

    def get_supply_voltage(self):
        response = self.send_command('74001A0000')
        if response and response.startswith('74001A'):
            voltage_hex = response[6:10]
            voltage = int(voltage_hex, 16) * 0.001
            return voltage
        return None

    def get_heat_sink_temperature(self):
        response = self.send_command('74001B0000')
        if response and response.startswith('74001B'):
            temperature_hex = response[6:10]
            temperature = int(temperature_hex, 16) * 0.1
            return temperature
        return None

    def get_optical_pulse_frequency(self):
        response = self.send_command('7400710000')
        if response and response.startswith('740071'):
            frequency_hex = response[6:14]
            frequency = int(frequency_hex, 16) * 0.001
            return frequency
        return None

    def get_display_text(self):
        response = self.send_command('7400780000')
        if response and response.startswith('740078'):
            text_hex = response[6:]
            text = bytes.fromhex(text_hex).decode('utf-8')
            return text
        return None

    def get_power_readout(self):
        response = self.send_command('74007A0000')
        if response and response.startswith('74007A'):
            power = int(response[6:8], 16)
            return power
        return None

    def set_emission(self, on=True):
        emission_value = '01' if on else '00'
        response = self.send_command(f'740030{emission_value}00')
        return response and response.startswith('74')

    def set_trig_mode(self, mode):
        response = self.send_command(f'740031{mode:02X}00')
        return response and response.startswith('74')

    def set_interlock(self, interlock_value):
        response = self.send_command(f'740032{interlock_value:04X}')
        return response and response.startswith('74')

    def get_status(self):
        response = self.send_command('7400660000')
        if response and response.startswith('740066'):
            status_bits = response[6:]
            return status_bits
        return None

def main():
    laser = SuperKCompactControl()
    laser.connect()
    if laser.set_emission(on=True):
        print("Laser is turned on")
    else:
        print("Failed to turn on the laser")
    time.sleep(2)
    if laser.set_emission(on=False):
        print("Laser is turned off")
    else:
        print("Failed to turn off the laser")
    laser.disconnect()
if __name__ == "__main__":
    main()
