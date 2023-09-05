# Author: Ahmad Azizan (aaaba2@cam.ac.uk)
import matplotlib.pyplot as plt
import numpy as np
import os

def process_data(input_file_path, output_file_path):
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        t1_data = []
        t2_data = []
        line_count = 0 
        for line in input_file:
            parts = line.strip().split(';')
            if len(parts) >= 3:
                if parts[0] == 'T1':
                    t1_data = f"{parts[0]}, {parts[2]}"
                elif parts[0] == 'T2':
                    t2_data = f"{parts[0]}, {parts[2]}"
                    if line_count >= 0: 
                        output_file.write(f"{t1_data}, {t2_data}\n")
                    line_count += 1  

def plot_temperature_vs_time(file_path):
    data = np.genfromtxt(file_path, delimiter=',', dtype=float)
    time_elapsed = data[:, 2]
    temp1 = data[:, 1]
    temp2 = data[:, 4]
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.set_dpi(600)
    ax.set_xlabel("Time Elapsed (seconds)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Temperature (°C)", fontsize=14, fontweight="bold")
    ax.set_title("Temperature vs. Time", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    line1, = ax.plot(time_elapsed, temp1, label=f'Fibre Temperature ({len(temp1)} data points)', marker='o', linestyle='-', color=cmap(0), markersize=2)
    line2, = ax.plot(time_elapsed, temp2, label=f'Ambient Temperature ({len(temp2)} data points)', marker='o', linestyle='-', color=cmap(1), markersize=2)
    ax.legend(handles=[line1, line2], loc="upper right", fontsize=10)
    plt.show()

def calculate_time_per_temperature_recording(file_path):
    data = np.genfromtxt(file_path, delimiter=',', dtype=float)
    time_elapsed = data[:, 2]
    time_per_temperature_recording = [time_elapsed[i] - time_elapsed[i - 1] for i in range(1, len(time_elapsed))]
    return time_per_temperature_recording

def plot_temperature_data(file_path):
    time_per_temperature_recording = calculate_time_per_temperature_recording(file_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.set_dpi(600)
    ax.set_xlabel("Number of Temperature Recordings", fontsize=14, fontweight="bold")
    ax.set_ylabel("Time Taken (milliseconds)", fontsize=14, fontweight="bold")
    ax.set_title("Time Taken to Record Each Temperature", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    mean_time = sum(time_per_temperature_recording) / len(time_per_temperature_recording)
    median_time = sorted(time_per_temperature_recording)[len(time_per_temperature_recording) // 2]
    std_deviation = (sum((x - mean_time) ** 2 for x in time_per_temperature_recording) / len(time_per_temperature_recording)) ** 0.5
    ax.errorbar(
        range(1, len(time_per_temperature_recording) + 1),
        time_per_temperature_recording,
        yerr=std_deviation,
        fmt='o',
        markersize=3,
        capsize=3,
        label='Time Taken',
    )
    ax.axhline(y=mean_time, color='blue', linestyle='-', label=f'Mean Time: {mean_time:.2f} ms')
    ax.axhline(y=median_time, color='green', linestyle='-', label=f'Median Time: {median_time:.2f} ms')
    ax.text(0.58, 0.80, f"Total Temperature Recordings: {len(time_per_temperature_recording)}", transform=ax.transAxes, fontsize=10, color='blue')
    ax.text(0.58, 0.76, f"Total Time Taken: {sum(time_per_temperature_recording):.2f} seconds", transform=ax.transAxes, fontsize=10, color='blue')
    ax.legend(loc='upper right', fontsize=10)
    plt.show()

def main():
    try:
        base_path = r'C:\Users\DELL\Documents\optofluidics-master\optofluidics-master\Python\tempcontroller\\'        
        input_file_path = os.path.join(base_path, 'output_start.txt')
        output_file_path = os.path.join(base_path, 'output_cleaned.txt')
        process_data(input_file_path, output_file_path)
        plot_temperature_vs_time(output_file_path)
        plot_temperature_data(output_file_path)
    except Exception as e:
        print("An error occurred:", str(e))

if __name__ == "__main__":
    main()
