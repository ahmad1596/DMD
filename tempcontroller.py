# Author: Ahmad Azizan (aaaba2@cam.ac.uk)
import serial
import threading
import time
import os

def setup_serial_port(com_port='COM5', baud_rate=115200):
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

def read_and_append_data(ser, target_temperature, numerator, start_time, output_file_path):
    with open(output_file_path, 'a') as txt_file:
        while True:
            data = ser.readline().decode('utf-8').strip()
            print(data)
            elapsed_time = time.time() - start_time
            if data.startswith('T1;') or data.startswith('T2;'):
                temperature = float(data.split(';')[2])
                txt_file.write(f"{data}, {elapsed_time:.2f}\n")
                if (numerator > 0 and temperature >= target_temperature) or (numerator < 0 and temperature <= target_temperature):
                    print(f"Temperature has reached the target temperature ({temperature}°C).")
                    return
                
def define_cycle_settings():
    cycle_settings = [
        (255, 30.0),
        (-255, 20.0),
        (255, 30.0),
        (-255, 20.0),
        (255, 30.0),
        (-255, 20.0)
    ]
    num_cycles = 6
    return cycle_settings, num_cycles

def configure_cycle(ser, numerator, target_temperature, start_time, output_file_path):
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
    data_thread = threading.Thread(target=read_and_append_data, args=(ser, target_temperature, numerator, start_time, output_file_path))
    data_thread.start()
    data_thread.join()
    
def disconnect_serial_port(ser):
    ser.write(b'regoff\n')
    time.sleep(1)
    ser.write(b'stop\n')
    time.sleep(1)
    ser.write(b'off\n')
    ser.close()
    print("Serial connection is disconnected.")
        
def main():
    try:
        ser = setup_serial_port()
        if ser is None:
            return
        cycle_settings, num_cycles = define_cycle_settings()
        base_output_file = r'C:\Users\DELL\Documents\optofluidics-master\optofluidics-master\Python\tempcontroller\output_start'
        output_file_path = get_output_file_path(base_output_file)
        start_time = time.time()
        for cycle in range(num_cycles):
            numerator, target_temperature = cycle_settings[cycle]
            print(f"\nCycle {cycle + 1}")
            configure_cycle(ser, numerator, target_temperature, start_time, output_file_path)
    except Exception as e:
        print("An error occurred:", str(e))
    finally:
        disconnect_serial_port(ser)

if __name__ == "__main__":
    main()

