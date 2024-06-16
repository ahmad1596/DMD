import serial
import threading
import time
import os
import matplotlib.pyplot as plt

def setup_serial_port(com_port='COM4', baud_rate=115200):
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

def read_and_append_data(ser, target_temperature, stop_temperature, numerator, start_time, output_file_path, fiber_temperature_data, ambient_temperature_data):
    while True:
        data = ser.readline().decode('utf-8').strip()
        print(data)
        elapsed_time = (time.time() - start_time) / 60  # Convert elapsed time to minutes
        if data.startswith('T1;'):
            temperature = float(data.split(';')[2])
            fiber_temperature_data.append((elapsed_time, temperature))  # 
            with open(output_file_path, 'a') as txt_file:
                txt_file.write(f"{data}, {elapsed_time:.2f}\n")

            if numerator > 0:
                if temperature >= stop_temperature:
                    print(f"Fibre Temperature has reached the stop temperature ({stop_temperature}°C).")
                    return
                if temperature >= target_temperature:
                    print(f"Fibre Temperature has reached the target temperature ({target_temperature}°C).")
                    return
            else:
                if temperature <= stop_temperature:
                    print(f"Fibre Temperature has reached the stop temperature ({stop_temperature}°C).")
                    return
                if temperature <= target_temperature:
                    print(f"Fibre Temperature has reached the target temperature ({target_temperature}°C).")
                    return

        elif data.startswith('T2;'):
            temperature = float(data.split(';')[2])
            ambient_temperature_data.append((elapsed_time, temperature))  # Append data to ambient list
            with open(output_file_path, 'a') as txt_file:
                txt_file.write(f"{data}, {elapsed_time:.2f}\n")

def define_cycle_settings():
    cycle_settings = [
        (255, 25, 25, 10),  # (numerator, target_temperature, stop_temperature, wait_time)
        (-255, 20, 20, 10),
        (255, 25, 25, 10),
        (-255, 20, 20, 10)
    ]
    num_cycles = len(cycle_settings)
    return cycle_settings, num_cycles

def configure_cycle(ser, numerator, target_temperature, stop_temperature, start_time, output_file_path, fiber_temperature_data, ambient_temperature_data):
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
    read_thread = threading.Thread(target=read_and_append_data, args=(ser, target_temperature, stop_temperature, numerator, start_time, output_file_path, fiber_temperature_data, ambient_temperature_data))
    read_thread.start()
    read_thread.join()

def disconnect_serial_port(ser):
    ser.write(b'regoff\n')
    time.sleep(1)
    ser.write(b'stop\n')
    time.sleep(1)
    ser.write(b'off\n')
    ser.close()
    print("Serial connection is disconnected.")

def plot_temperature_data(fiber_temperature_data, ambient_temperature_data):
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_dpi(600)

    fiber_time_values, fiber_temperature_values = zip(*fiber_temperature_data)
    ambient_time_values, ambient_temperature_values = zip(*ambient_temperature_data)
    
    ax.set_xlabel("Time Elapsed (minutes)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Temperature (°C)", fontsize=14, fontweight="bold")
    ax.set_title("Temperature vs. Time", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    
    line1, = ax.plot(fiber_time_values, fiber_temperature_values, label='Fibre Temperature', 
                     marker='o', linestyle='-', color=cmap(0), markersize=2)
    line2, = ax.plot(ambient_time_values, ambient_temperature_values, label='Ambient Temperature', 
                     marker='o', linestyle='-', color=cmap(1), markersize=2)
    
    ax.legend(handles=[line1, line2], loc="center right", fontsize=10)
    plt.show()

def main():
    try:
        ser = setup_serial_port()
        if ser is None:
            return
        cycle_settings, num_cycles = define_cycle_settings()
        base_output_file = r'C:\Users\DELL\Documents\optofluidics-master\optofluidics-master\Python\tempcontroller\output_start'
        output_file_path = get_output_file_path(base_output_file)
        fiber_temperature_data = []  # List to store fiber temperature data
        ambient_temperature_data = []  # List to store ambient temperature data
        start_time = time.time()
        for cycle in range(num_cycles):
            numerator, target_temperature, stop_temperature, wait_time = cycle_settings[cycle]
            print(f"\nCycle {cycle + 1}")
            configure_cycle(ser, numerator, target_temperature, stop_temperature, start_time, output_file_path, fiber_temperature_data, ambient_temperature_data)
            print(f"Waiting for {wait_time} seconds before starting the next cycle...")
            time.sleep(wait_time)  # Wait for the specified time before the next cycle
        plot_temperature_data(fiber_temperature_data, ambient_temperature_data)
    except Exception as e:
        print("An error occurred:", str(e))
    finally:
        disconnect_serial_port(ser)

if __name__ == "__main__":
    main()
