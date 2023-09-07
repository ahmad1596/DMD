# Author: Ahmad Azizan (aaaba2@cam.ac.uk)
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

    def set_power_level(self, power_percent):
        result = registerWriteU8('COM3', 1, 0x3E, power_percent, -1)
        print('Setting power level:', RegisterResultTypes(result))
    
    def configure_laser(laser):
        laser.set_interlock(2)
        laser.set_watchdog_interval(0)
        laser.set_display_backlight(100)
        laser.set_power_level(100) 
    
    def set_trigger_parameters(self):
        self.set_trig_level(2000)
        self.set_trig_mode(0)
        self.set_internal_pulse_frequency(1000)
        self.set_burst_pulses(10)
        
    def set_emission_sequence(self):
        self.set_emission_status(1)
        time.sleep(5)
        self.set_emission_status(0)
        
    def disconnect(self):
        if self.serial_port:
            self.serial_port.close()
            self.serial_port = None
            print('SuperK Compact disconnected')
            
def main():
    laser = SuperKCompactControl()
    try:
        laser.connect()
        laser.configure_laser()
        laser.set_trigger_parameters()
        laser.set_emission_sequence()
    except Exception as e:
        print("An error occurred:", str(e))
    finally:
        laser.disconnect()

if __name__ == "__main__":
    main()
