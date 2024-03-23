##############################################################################################################################################################

import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import sys

##############################################################################################################################################################


def plot_spectrum(x_values, intensities, x_label, y_label, title, legend_label=None):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    ax.plot(x_values, intensities, label=legend_label)
    ax.set_xlabel(x_label, fontsize=14, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=16, fontweight="bold")
    plt.show()
    
##############################################################################################################################################################

date = "2024-03-20T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/hera/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 5
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    print("\nKeys: %s" % f.keys())
    keys = list(f.keys())
key = keys[0]
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    g = f[key]
    indices = sorted([d.replace("spectrum_", "") for d in g.keys()])
    dsets = ["{0}".format(n) for n in indices]
    print("attributes for test:", g[dsets[0]].attrs.keys())
    print(dsets)
    sp = 0
    h = g[dsets[sp]]
    print("\nh:", h)
    indices2 = sorted(h.keys())
    dsets1 = ["{0}".format(n) for n in indices2]
    dsets2 = sorted(dsets1, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    sp2 = 0 
    print("\nspectra:", dsets2)
    laser_wavelength = 650  
    wavelengths = np.array(h[dsets2[sp2]].attrs["wavelengths"])
    start_times = np.array(h[dsets2[sp2]].attrs["creation_timestamp"]) 
    integration_time = np.array(h[dsets2[sp2]].attrs["integration_time"]) 
    creation_time = np.array(h[dsets2[sp2]].attrs["creation_timestamp"])  
    time_interval = np.array(h[dsets2[sp2]].attrs["time_interval"])
    information = np.array(h[dsets2[sp2]].attrs["information"])
    data = np.zeros((len(h[dsets2[sp2]]), len(dsets2)), dtype=float)
    creation_times = np.array((len(h[dsets2[sp2]].attrs["creation_timestamp"]), len(dsets2)), dtype=str)
    m = np.array([], dtype=np.float32)
    for j in range(len(dsets2)):
        data[:, j] = np.array(h[dsets2[j]]) - np.array(h[dsets2[j]].attrs["background"])
        starttime = h[dsets2[sp2]].attrs["creation_timestamp"]
        time = h[dsets2[j]].attrs["creation_timestamp"]
        strstarttime = starttime
        strtime = time
        newstarttime = strstarttime.split("T")[-1]
        newtime = strtime.split("T")[-1]
        newstarttime = newstarttime.replace(date, "")
        newtime = newtime.replace(date, "")
        start_time_s = pd.Timestamp(newstarttime).timestamp()
        time_s = pd.Timestamp(newtime).timestamp()
        delta_time = time_s - start_time_s
        m = np.append(m, delta_time)
t = np.reshape(m, (1, len(dsets2)))
data_average = np.sum(data, 1) / len(dsets2)
data_wl_wavelength_1 = np.zeros((len(data_average), 2), dtype=float)
data_wl_wavelength_1[:, 0] = wavelengths
data_wl_wavelength_1[:, 1] = data_average
plot_spectrum(data_wl_wavelength_1[:, 0], data_wl_wavelength_1[:, 1],
              x_label='Wavelength (nm)', y_label='Intensity (a.u.)',
              title='Average Spectrum',
              legend_label='Intensity at Wavelength')
specific_wavelength = laser_wavelength
wavelength_index = np.argmin(np.abs(wavelengths - specific_wavelength))
intensity_at_specific_wavelength = data[wavelength_index, :]
plt.figure(figsize=(8, 6), dpi=600)
for j in range(len(dsets2) - 1):
    plt.plot([m[j], m[j + 1]], [intensity_at_specific_wavelength[j], intensity_at_specific_wavelength[j + 1]],
             marker='o', linestyle='-', linewidth=1, label=f'Spectrum {j + 1}-{j + 2}')
plt.plot(m[-1], intensity_at_specific_wavelength[-1], marker='o', label=f'Spectrum {len(dsets2)}')
plt.xlabel('Time (seconds)', fontsize=14, fontweight="bold")
plt.ylabel(f'Intensity at {specific_wavelength} nm', fontsize=14, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12, direction="in")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend(loc="lower right", fontsize=8)
plt.title(f'Intensity Variation at {specific_wavelength} nm Over Time', fontsize=16, fontweight="bold")
plt.show()
folder_name = 'Ref_Spectrum_Data'
folder_path = os.path.join(DIRPATH, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
#csv_file_path1 = os.path.join(folder_path, "Ref_QE65000_Spectrometer.csv")
#pd.DataFrame(data_wl_wavelength).to_csv(csv_file_path1, header=None, index=None)

##############################################################################################################################################################

date = "2024-03-23T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/hera/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 8
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    print("\nKeys: %s" % f.keys())
    keys = list(f.keys())
key = keys[16]
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    g = f[key]
    indices = sorted([d.replace("spectrum_", "") for d in g.keys()])
    dsets = ["{0}".format(n) for n in indices]
    print("attributes for test:", g[dsets[0]].attrs.keys())
    print(dsets)
    sp = 0
    h = g[dsets[sp]]
    print("\nh:", h)
    indices2 = sorted(h.keys())
    dsets1 = ["{0}".format(n) for n in indices2]
    dsets2 = sorted(dsets1, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    sp2 = 0 
    print("\nspectra:", dsets2)
    laser_wavelength = 650  
    wavelengths = np.array(h[dsets2[sp2]].attrs["wavelengths"])
    start_times = np.array(h[dsets2[sp2]].attrs["creation_timestamp"]) 
    integration_time = np.array(h[dsets2[sp2]].attrs["integration_time"]) 
    creation_time = np.array(h[dsets2[sp2]].attrs["creation_timestamp"])  
    time_interval = np.array(h[dsets2[sp2]].attrs["time_interval"])
    information = np.array(h[dsets2[sp2]].attrs["information"])
    data = np.zeros((len(h[dsets2[sp2]]), len(dsets2)), dtype=float)
    creation_times = np.array((len(h[dsets2[sp2]].attrs["creation_timestamp"]), len(dsets2)), dtype=str)
    m = np.array([], dtype=np.float32)
    for j in range(len(dsets2)):
        data[:, j] = np.array(h[dsets2[j]]) - np.array(h[dsets2[j]].attrs["background"])
        starttime = h[dsets2[sp2]].attrs["creation_timestamp"]
        time = h[dsets2[j]].attrs["creation_timestamp"]
        strstarttime = starttime
        strtime = time
        newstarttime = strstarttime.split("T")[-1]
        newtime = strtime.split("T")[-1]
        newstarttime = newstarttime.replace(date, "")
        newtime = newtime.replace(date, "")
        start_time_s = pd.Timestamp(newstarttime).timestamp()
        time_s = pd.Timestamp(newtime).timestamp()
        delta_time = time_s - start_time_s
        m = np.append(m, delta_time)
t = np.reshape(m, (1, len(dsets2)))
data_average = np.sum(data, 1) / len(dsets2)
data_wl_wavelength_2 = np.zeros((len(data_average), 2), dtype=float)
data_wl_wavelength_2[:, 0] = wavelengths
data_wl_wavelength_2[:, 1] = data_average
plot_spectrum(data_wl_wavelength_2[:, 0], data_wl_wavelength_2[:, 1],
              x_label='Wavelength (nm)', y_label='Intensity (a.u.)',
              title='Average Spectrum',
              legend_label='Intensity at Wavelength')
specific_wavelength = laser_wavelength
wavelength_index = np.argmin(np.abs(wavelengths - specific_wavelength))
intensity_at_specific_wavelength = data[wavelength_index, :]
plt.figure(figsize=(8, 6), dpi=600)
for j in range(len(dsets2) - 1):
    plt.plot([m[j], m[j + 1]], [intensity_at_specific_wavelength[j], intensity_at_specific_wavelength[j + 1]],
             marker='o', linestyle='-', linewidth=1, label=f'Spectrum {j + 1}-{j + 2}')
plt.plot(m[-1], intensity_at_specific_wavelength[-1], marker='o', label=f'Spectrum {len(dsets2)}')
plt.xlabel('Time (seconds)', fontsize=14, fontweight="bold")
plt.ylabel(f'Intensity at {specific_wavelength} nm', fontsize=14, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12, direction="in")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend(loc="lower right", fontsize=8)
plt.title(f'Intensity Variation at {specific_wavelength} nm Over Time', fontsize=16, fontweight="bold")
plt.show()
folder_name = 'Ref_Spectrum_Data'
folder_path = os.path.join(DIRPATH, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
#csv_file_path2 = os.path.join(folder_path, "Ref_USB2000_Spectrometer.csv")
#pd.DataFrame(data_wl_wavelength).to_csv(csv_file_path2, header=None, index=None)

##############################################################################################################################################################

df_normalized = pd.DataFrame({
    'Wavelength': data_wl_wavelength_1[:, 0],
    'Intensity_1': data_wl_wavelength_1[:, 1],
    'Intensity_2': data_wl_wavelength_2[:, 1],
    'Normalized_Intensity': abs(data_wl_wavelength_2[:, 1] / data_wl_wavelength_1[:, 1]) 
})
plot_spectrum(df_normalized['Wavelength'], df_normalized['Normalized_Intensity'],
              x_label='Wavelength (nm)', y_label='Normalized Intensity',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')
#csv_file_path_normalized = os.path.join(folder_path, "Normalized_Intensity.csv")
#df_normalized.to_csv(csv_file_path_normalized, index=None)

##############################################################################################################################################################

input_wavelength = float(input("Enter the wavelength: "))
power_percentage = float(input(f"Enter the power percentage at {input_wavelength} nm: "))
closest_wavelength_index = np.argmin(np.abs(df_normalized['Wavelength'] - input_wavelength))
closest_wavelength = df_normalized.loc[closest_wavelength_index, 'Wavelength']
scaled_factor = power_percentage / df_normalized.loc[closest_wavelength_index, 'Normalized_Intensity']
df_normalized['Percentage_Transmission_1'] = df_normalized['Normalized_Intensity'] * scaled_factor
df_filtered = df_normalized[(df_normalized['Wavelength'] > 400)] #& (df_normalized['Wavelength'] < 1000)]
plot_spectrum(df_filtered['Wavelength'], df_filtered['Percentage_Transmission_1'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')
#csv_file_path_percentage_transmission = os.path.join(folder_path, "Percentage_Transmission.csv")
#df_normalized[['Wavelength', 'Percentage_Transmission']].to_csv(csv_file_path_percentage_transmission, index=None)


##############################################################################################################################################################

user_input = input("Press Enter to continue or type 'exit' to quit: ")
if user_input.lower() == 'exit':
    sys.exit()

##############################################################################################################################################################

date = "2024-03-20T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/hera/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 5
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    print("\nKeys: %s" % f.keys())
    keys = list(f.keys())
key = keys[0]
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    g = f[key]
    indices = sorted([d.replace("spectrum_", "") for d in g.keys()])
    dsets = ["{0}".format(n) for n in indices]
    print("attributes for test:", g[dsets[0]].attrs.keys())
    print(dsets)
    sp = 0
    h = g[dsets[sp]]
    print("\nh:", h)
    indices2 = sorted(h.keys())
    dsets1 = ["{0}".format(n) for n in indices2]
    dsets2 = sorted(dsets1, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    sp2 = 0 
    print("\nspectra:", dsets2)
    laser_wavelength = 650  
    wavelengths = np.array(h[dsets2[sp2]].attrs["wavelengths"])
    start_times = np.array(h[dsets2[sp2]].attrs["creation_timestamp"]) 
    integration_time = np.array(h[dsets2[sp2]].attrs["integration_time"]) 
    creation_time = np.array(h[dsets2[sp2]].attrs["creation_timestamp"])  
    time_interval = np.array(h[dsets2[sp2]].attrs["time_interval"])
    information = np.array(h[dsets2[sp2]].attrs["information"])
    data = np.zeros((len(h[dsets2[sp2]]), len(dsets2)), dtype=float)
    creation_times = np.array((len(h[dsets2[sp2]].attrs["creation_timestamp"]), len(dsets2)), dtype=str)
    m = np.array([], dtype=np.float32)
    for j in range(len(dsets2)):
        data[:, j] = np.array(h[dsets2[j]]) - np.array(h[dsets2[j]].attrs["background"])
        starttime = h[dsets2[sp2]].attrs["creation_timestamp"]
        time = h[dsets2[j]].attrs["creation_timestamp"]
        strstarttime = starttime
        strtime = time
        newstarttime = strstarttime.split("T")[-1]
        newtime = strtime.split("T")[-1]
        newstarttime = newstarttime.replace(date, "")
        newtime = newtime.replace(date, "")
        start_time_s = pd.Timestamp(newstarttime).timestamp()
        time_s = pd.Timestamp(newtime).timestamp()
        delta_time = time_s - start_time_s
        m = np.append(m, delta_time)
t = np.reshape(m, (1, len(dsets2)))
data_average = np.sum(data, 1) / len(dsets2)
data_wl_wavelength_1 = np.zeros((len(data_average), 2), dtype=float)
data_wl_wavelength_1[:, 0] = wavelengths
data_wl_wavelength_1[:, 1] = data_average
plot_spectrum(data_wl_wavelength_1[:, 0], data_wl_wavelength_1[:, 1],
              x_label='Wavelength (nm)', y_label='Intensity (a.u.)',
              title='Average Spectrum',
              legend_label='Intensity at Wavelength')
specific_wavelength = laser_wavelength
wavelength_index = np.argmin(np.abs(wavelengths - specific_wavelength))
intensity_at_specific_wavelength = data[wavelength_index, :]
plt.figure(figsize=(8, 6), dpi=600)
for j in range(len(dsets2) - 1):
    plt.plot([m[j], m[j + 1]], [intensity_at_specific_wavelength[j], intensity_at_specific_wavelength[j + 1]],
             marker='o', linestyle='-', linewidth=1, label=f'Spectrum {j + 1}-{j + 2}')
plt.plot(m[-1], intensity_at_specific_wavelength[-1], marker='o', label=f'Spectrum {len(dsets2)}')
plt.xlabel('Time (seconds)', fontsize=14, fontweight="bold")
plt.ylabel(f'Intensity at {specific_wavelength} nm', fontsize=14, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12, direction="in")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend(loc="lower right", fontsize=8)
plt.title(f'Intensity Variation at {specific_wavelength} nm Over Time', fontsize=16, fontweight="bold")
plt.show()
folder_name = 'Ref_Spectrum_Data'
folder_path = os.path.join(DIRPATH, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
#csv_file_path1 = os.path.join(folder_path, "Ref_QE65000_Spectrometer.csv")
#pd.DataFrame(data_wl_wavelength).to_csv(csv_file_path1, header=None, index=None)

##############################################################################################################################################################

date = "2024-03-23T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/hera/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 8
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    print("\nKeys: %s" % f.keys())
    keys = list(f.keys())
key = keys[16]
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    g = f[key]
    indices = sorted([d.replace("spectrum_", "") for d in g.keys()])
    dsets = ["{0}".format(n) for n in indices]
    print("attributes for test:", g[dsets[0]].attrs.keys())
    print(dsets)
    sp = 0
    h = g[dsets[sp]]
    print("\nh:", h)
    indices2 = sorted(h.keys())
    dsets1 = ["{0}".format(n) for n in indices2]
    dsets2 = sorted(dsets1, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    sp2 = 0 
    print("\nspectra:", dsets2)
    laser_wavelength = 650  
    wavelengths = np.array(h[dsets2[sp2]].attrs["wavelengths"])
    start_times = np.array(h[dsets2[sp2]].attrs["creation_timestamp"]) 
    integration_time = np.array(h[dsets2[sp2]].attrs["integration_time"]) 
    creation_time = np.array(h[dsets2[sp2]].attrs["creation_timestamp"])  
    time_interval = np.array(h[dsets2[sp2]].attrs["time_interval"])
    information = np.array(h[dsets2[sp2]].attrs["information"])
    data = np.zeros((len(h[dsets2[sp2]]), len(dsets2)), dtype=float)
    creation_times = np.array((len(h[dsets2[sp2]].attrs["creation_timestamp"]), len(dsets2)), dtype=str)
    m = np.array([], dtype=np.float32)
    for j in range(len(dsets2)):
        data[:, j] = np.array(h[dsets2[j]]) - np.array(h[dsets2[j]].attrs["background"])
        starttime = h[dsets2[sp2]].attrs["creation_timestamp"]
        time = h[dsets2[j]].attrs["creation_timestamp"]
        strstarttime = starttime
        strtime = time
        newstarttime = strstarttime.split("T")[-1]
        newtime = strtime.split("T")[-1]
        newstarttime = newstarttime.replace(date, "")
        newtime = newtime.replace(date, "")
        start_time_s = pd.Timestamp(newstarttime).timestamp()
        time_s = pd.Timestamp(newtime).timestamp()
        delta_time = time_s - start_time_s
        m = np.append(m, delta_time)
t = np.reshape(m, (1, len(dsets2)))
data_average = np.sum(data, 1) / len(dsets2)
data_wl_wavelength_2 = np.zeros((len(data_average), 2), dtype=float)
data_wl_wavelength_2[:, 0] = wavelengths
data_wl_wavelength_2[:, 1] = data_average
plot_spectrum(data_wl_wavelength_2[:, 0], data_wl_wavelength_2[:, 1],
              x_label='Wavelength (nm)', y_label='Intensity (a.u.)',
              title='Average Spectrum',
              legend_label='Intensity at Wavelength')
specific_wavelength = laser_wavelength
wavelength_index = np.argmin(np.abs(wavelengths - specific_wavelength))
intensity_at_specific_wavelength = data[wavelength_index, :]
plt.figure(figsize=(8, 6), dpi=600)
for j in range(len(dsets2) - 1):
    plt.plot([m[j], m[j + 1]], [intensity_at_specific_wavelength[j], intensity_at_specific_wavelength[j + 1]],
             marker='o', linestyle='-', linewidth=1, label=f'Spectrum {j + 1}-{j + 2}')
plt.plot(m[-1], intensity_at_specific_wavelength[-1], marker='o', label=f'Spectrum {len(dsets2)}')
plt.xlabel('Time (seconds)', fontsize=14, fontweight="bold")
plt.ylabel(f'Intensity at {specific_wavelength} nm', fontsize=14, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12, direction="in")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend(loc="lower right", fontsize=8)
plt.title(f'Intensity Variation at {specific_wavelength} nm Over Time', fontsize=16, fontweight="bold")
plt.show()
folder_name = 'Ref_Spectrum_Data'
folder_path = os.path.join(DIRPATH, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
#csv_file_path2 = os.path.join(folder_path, "Ref_USB2000_Spectrometer.csv")
#pd.DataFrame(data_wl_wavelength).to_csv(csv_file_path2, header=None, index=None)

##############################################################################################################################################################

df_normalized = pd.DataFrame({
    'Wavelength': data_wl_wavelength_1[:, 0],
    'Intensity_1': data_wl_wavelength_1[:, 1],
    'Intensity_2': data_wl_wavelength_2[:, 1],
    'Normalized_Intensity': abs(data_wl_wavelength_2[:, 1] / data_wl_wavelength_1[:, 1]) 
})
plot_spectrum(df_normalized['Wavelength'], df_normalized['Normalized_Intensity'],
              x_label='Wavelength (nm)', y_label='Normalized Intensity',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')
#csv_file_path_normalized = os.path.join(folder_path, "Normalized_Intensity.csv")
#df_normalized.to_csv(csv_file_path_normalized, index=None)

##############################################################################################################################################################

input_wavelength = float(input("Enter the wavelength: "))
power_percentage = float(input(f"Enter the power percentage at {input_wavelength} nm: "))
closest_wavelength_index = np.argmin(np.abs(df_normalized['Wavelength'] - input_wavelength))
closest_wavelength = df_normalized.loc[closest_wavelength_index, 'Wavelength']
scaled_factor = power_percentage / df_normalized.loc[closest_wavelength_index, 'Normalized_Intensity']
df_normalized['Percentage_Transmission_2'] = df_normalized['Normalized_Intensity'] * scaled_factor
df_filtered = df_normalized[(df_normalized['Wavelength'] > 400)] #& (df_normalized['Wavelength'] < 1000)]
plot_spectrum(df_filtered['Wavelength'], df_filtered['Percentage_Transmission_2'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')
#csv_file_path_percentage_transmission = os.path.join(folder_path, "Percentage_Transmission_2.csv")
#df_normalized[['Wavelength', 'Percentage_Transmission']].to_csv(csv_file_path_percentage_transmission, index=None)


##############################################################################################################################################################

user_input = input("Press Enter to continue or type 'exit' to quit: ")
if user_input.lower() == 'exit':
    sys.exit()

##############################################################################################################################################################

date = "2024-03-20T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/hera/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 5
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    print("\nKeys: %s" % f.keys())
    keys = list(f.keys())
key = keys[0]
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    g = f[key]
    indices = sorted([d.replace("spectrum_", "") for d in g.keys()])
    dsets = ["{0}".format(n) for n in indices]
    print("attributes for test:", g[dsets[0]].attrs.keys())
    print(dsets)
    sp = 0
    h = g[dsets[sp]]
    print("\nh:", h)
    indices2 = sorted(h.keys())
    dsets1 = ["{0}".format(n) for n in indices2]
    dsets2 = sorted(dsets1, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    sp2 = 0 
    print("\nspectra:", dsets2)
    laser_wavelength = 650  
    wavelengths = np.array(h[dsets2[sp2]].attrs["wavelengths"])
    start_times = np.array(h[dsets2[sp2]].attrs["creation_timestamp"]) 
    integration_time = np.array(h[dsets2[sp2]].attrs["integration_time"]) 
    creation_time = np.array(h[dsets2[sp2]].attrs["creation_timestamp"])  
    time_interval = np.array(h[dsets2[sp2]].attrs["time_interval"])
    information = np.array(h[dsets2[sp2]].attrs["information"])
    data = np.zeros((len(h[dsets2[sp2]]), len(dsets2)), dtype=float)
    creation_times = np.array((len(h[dsets2[sp2]].attrs["creation_timestamp"]), len(dsets2)), dtype=str)
    m = np.array([], dtype=np.float32)
    for j in range(len(dsets2)):
        data[:, j] = np.array(h[dsets2[j]]) - np.array(h[dsets2[j]].attrs["background"])
        starttime = h[dsets2[sp2]].attrs["creation_timestamp"]
        time = h[dsets2[j]].attrs["creation_timestamp"]
        strstarttime = starttime
        strtime = time
        newstarttime = strstarttime.split("T")[-1]
        newtime = strtime.split("T")[-1]
        newstarttime = newstarttime.replace(date, "")
        newtime = newtime.replace(date, "")
        start_time_s = pd.Timestamp(newstarttime).timestamp()
        time_s = pd.Timestamp(newtime).timestamp()
        delta_time = time_s - start_time_s
        m = np.append(m, delta_time)
t = np.reshape(m, (1, len(dsets2)))
data_average = np.sum(data, 1) / len(dsets2)
data_wl_wavelength_1 = np.zeros((len(data_average), 2), dtype=float)
data_wl_wavelength_1[:, 0] = wavelengths
data_wl_wavelength_1[:, 1] = data_average
plot_spectrum(data_wl_wavelength_1[:, 0], data_wl_wavelength_1[:, 1],
              x_label='Wavelength (nm)', y_label='Intensity (a.u.)',
              title='Average Spectrum',
              legend_label='Intensity at Wavelength')
specific_wavelength = laser_wavelength
wavelength_index = np.argmin(np.abs(wavelengths - specific_wavelength))
intensity_at_specific_wavelength = data[wavelength_index, :]
plt.figure(figsize=(8, 6), dpi=600)
for j in range(len(dsets2) - 1):
    plt.plot([m[j], m[j + 1]], [intensity_at_specific_wavelength[j], intensity_at_specific_wavelength[j + 1]],
             marker='o', linestyle='-', linewidth=1, label=f'Spectrum {j + 1}-{j + 2}')
plt.plot(m[-1], intensity_at_specific_wavelength[-1], marker='o', label=f'Spectrum {len(dsets2)}')
plt.xlabel('Time (seconds)', fontsize=14, fontweight="bold")
plt.ylabel(f'Intensity at {specific_wavelength} nm', fontsize=14, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12, direction="in")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend(loc="lower right", fontsize=8)
plt.title(f'Intensity Variation at {specific_wavelength} nm Over Time', fontsize=16, fontweight="bold")
plt.show()
folder_name = 'Ref_Spectrum_Data'
folder_path = os.path.join(DIRPATH, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
#csv_file_path1 = os.path.join(folder_path, "Ref_QE65000_Spectrometer.csv")
#pd.DataFrame(data_wl_wavelength).to_csv(csv_file_path1, header=None, index=None)

##############################################################################################################################################################

date = "2024-03-23T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/hera/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 8
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    print("\nKeys: %s" % f.keys())
    keys = list(f.keys())
key = keys[16]
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    g = f[key]
    indices = sorted([d.replace("spectrum_", "") for d in g.keys()])
    dsets = ["{0}".format(n) for n in indices]
    print("attributes for test:", g[dsets[0]].attrs.keys())
    print(dsets)
    sp = 0
    h = g[dsets[sp]]
    print("\nh:", h)
    indices2 = sorted(h.keys())
    dsets1 = ["{0}".format(n) for n in indices2]
    dsets2 = sorted(dsets1, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    sp2 = 0 
    print("\nspectra:", dsets2)
    laser_wavelength = 650  
    wavelengths = np.array(h[dsets2[sp2]].attrs["wavelengths"])
    start_times = np.array(h[dsets2[sp2]].attrs["creation_timestamp"]) 
    integration_time = np.array(h[dsets2[sp2]].attrs["integration_time"]) 
    creation_time = np.array(h[dsets2[sp2]].attrs["creation_timestamp"])  
    time_interval = np.array(h[dsets2[sp2]].attrs["time_interval"])
    information = np.array(h[dsets2[sp2]].attrs["information"])
    data = np.zeros((len(h[dsets2[sp2]]), len(dsets2)), dtype=float)
    creation_times = np.array((len(h[dsets2[sp2]].attrs["creation_timestamp"]), len(dsets2)), dtype=str)
    m = np.array([], dtype=np.float32)
    for j in range(len(dsets2)):
        data[:, j] = np.array(h[dsets2[j]]) - np.array(h[dsets2[j]].attrs["background"])
        starttime = h[dsets2[sp2]].attrs["creation_timestamp"]
        time = h[dsets2[j]].attrs["creation_timestamp"]
        strstarttime = starttime
        strtime = time
        newstarttime = strstarttime.split("T")[-1]
        newtime = strtime.split("T")[-1]
        newstarttime = newstarttime.replace(date, "")
        newtime = newtime.replace(date, "")
        start_time_s = pd.Timestamp(newstarttime).timestamp()
        time_s = pd.Timestamp(newtime).timestamp()
        delta_time = time_s - start_time_s
        m = np.append(m, delta_time)
t = np.reshape(m, (1, len(dsets2)))
data_average = np.sum(data, 1) / len(dsets2)
data_wl_wavelength_2 = np.zeros((len(data_average), 2), dtype=float)
data_wl_wavelength_2[:, 0] = wavelengths
data_wl_wavelength_2[:, 1] = data_average
plot_spectrum(data_wl_wavelength_2[:, 0], data_wl_wavelength_2[:, 1],
              x_label='Wavelength (nm)', y_label='Intensity (a.u.)',
              title='Average Spectrum',
              legend_label='Intensity at Wavelength')
specific_wavelength = laser_wavelength
wavelength_index = np.argmin(np.abs(wavelengths - specific_wavelength))
intensity_at_specific_wavelength = data[wavelength_index, :]
plt.figure(figsize=(8, 6), dpi=600)
for j in range(len(dsets2) - 1):
    plt.plot([m[j], m[j + 1]], [intensity_at_specific_wavelength[j], intensity_at_specific_wavelength[j + 1]],
             marker='o', linestyle='-', linewidth=1, label=f'Spectrum {j + 1}-{j + 2}')
plt.plot(m[-1], intensity_at_specific_wavelength[-1], marker='o', label=f'Spectrum {len(dsets2)}')
plt.xlabel('Time (seconds)', fontsize=14, fontweight="bold")
plt.ylabel(f'Intensity at {specific_wavelength} nm', fontsize=14, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12, direction="in")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend(loc="lower right", fontsize=8)
plt.title(f'Intensity Variation at {specific_wavelength} nm Over Time', fontsize=16, fontweight="bold")
plt.show()
folder_name = 'Ref_Spectrum_Data'
folder_path = os.path.join(DIRPATH, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
#csv_file_path2 = os.path.join(folder_path, "Ref_USB2000_Spectrometer.csv")
#pd.DataFrame(data_wl_wavelength).to_csv(csv_file_path2, header=None, index=None)

##############################################################################################################################################################

df_normalized = pd.DataFrame({
    'Wavelength': data_wl_wavelength_1[:, 0],
    'Intensity_1': data_wl_wavelength_1[:, 1],
    'Intensity_2': data_wl_wavelength_2[:, 1],
    'Normalized_Intensity': abs(data_wl_wavelength_2[:, 1] / data_wl_wavelength_1[:, 1]) 
})
plot_spectrum(df_normalized['Wavelength'], df_normalized['Normalized_Intensity'],
              x_label='Wavelength (nm)', y_label='Normalized Intensity',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')
#csv_file_path_normalized = os.path.join(folder_path, "Normalized_Intensity.csv")
#df_normalized.to_csv(csv_file_path_normalized, index=None)

##############################################################################################################################################################

input_wavelength = float(input("Enter the wavelength: "))
power_percentage = float(input(f"Enter the power percentage at {input_wavelength} nm: "))
closest_wavelength_index = np.argmin(np.abs(df_normalized['Wavelength'] - input_wavelength))
closest_wavelength = df_normalized.loc[closest_wavelength_index, 'Wavelength']
scaled_factor = power_percentage / df_normalized.loc[closest_wavelength_index, 'Normalized_Intensity']
df_normalized['Percentage_Transmission_3'] = df_normalized['Normalized_Intensity'] * scaled_factor
df_filtered = df_normalized[(df_normalized['Wavelength'] > 400)] #& (df_normalized['Wavelength'] < 1000)]
plot_spectrum(df_filtered['Wavelength'], df_filtered['Percentage_Transmission_3'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')
#csv_file_path_percentage_transmission = os.path.join(folder_path, "Percentage_Transmission.csv")
#df_normalized[['Wavelength', 'Percentage_Transmission']].to_csv(csv_file_path_percentage_transmission, index=None)


##############################################################################################################################################################

user_input = input("Press Enter to continue or type 'exit' to quit: ")
if user_input.lower() == 'exit':
    sys.exit()

##############################################################################################################################################################

date = "2024-03-20T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/hera/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 5
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    print("\nKeys: %s" % f.keys())
    keys = list(f.keys())
key = keys[0]
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    g = f[key]
    indices = sorted([d.replace("spectrum_", "") for d in g.keys()])
    dsets = ["{0}".format(n) for n in indices]
    print("attributes for test:", g[dsets[0]].attrs.keys())
    print(dsets)
    sp = 0
    h = g[dsets[sp]]
    print("\nh:", h)
    indices2 = sorted(h.keys())
    dsets1 = ["{0}".format(n) for n in indices2]
    dsets2 = sorted(dsets1, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    sp2 = 0 
    print("\nspectra:", dsets2)
    laser_wavelength = 650  
    wavelengths = np.array(h[dsets2[sp2]].attrs["wavelengths"])
    start_times = np.array(h[dsets2[sp2]].attrs["creation_timestamp"]) 
    integration_time = np.array(h[dsets2[sp2]].attrs["integration_time"]) 
    creation_time = np.array(h[dsets2[sp2]].attrs["creation_timestamp"])  
    time_interval = np.array(h[dsets2[sp2]].attrs["time_interval"])
    information = np.array(h[dsets2[sp2]].attrs["information"])
    data = np.zeros((len(h[dsets2[sp2]]), len(dsets2)), dtype=float)
    creation_times = np.array((len(h[dsets2[sp2]].attrs["creation_timestamp"]), len(dsets2)), dtype=str)
    m = np.array([], dtype=np.float32)
    for j in range(len(dsets2)):
        data[:, j] = np.array(h[dsets2[j]]) - np.array(h[dsets2[j]].attrs["background"])
        starttime = h[dsets2[sp2]].attrs["creation_timestamp"]
        time = h[dsets2[j]].attrs["creation_timestamp"]
        strstarttime = starttime
        strtime = time
        newstarttime = strstarttime.split("T")[-1]
        newtime = strtime.split("T")[-1]
        newstarttime = newstarttime.replace(date, "")
        newtime = newtime.replace(date, "")
        start_time_s = pd.Timestamp(newstarttime).timestamp()
        time_s = pd.Timestamp(newtime).timestamp()
        delta_time = time_s - start_time_s
        m = np.append(m, delta_time)
t = np.reshape(m, (1, len(dsets2)))
data_average = np.sum(data, 1) / len(dsets2)
data_wl_wavelength_1 = np.zeros((len(data_average), 2), dtype=float)
data_wl_wavelength_1[:, 0] = wavelengths
data_wl_wavelength_1[:, 1] = data_average
plot_spectrum(data_wl_wavelength_1[:, 0], data_wl_wavelength_1[:, 1],
              x_label='Wavelength (nm)', y_label='Intensity (a.u.)',
              title='Average Spectrum',
              legend_label='Intensity at Wavelength')
specific_wavelength = laser_wavelength
wavelength_index = np.argmin(np.abs(wavelengths - specific_wavelength))
intensity_at_specific_wavelength = data[wavelength_index, :]
plt.figure(figsize=(8, 6), dpi=600)
for j in range(len(dsets2) - 1):
    plt.plot([m[j], m[j + 1]], [intensity_at_specific_wavelength[j], intensity_at_specific_wavelength[j + 1]],
             marker='o', linestyle='-', linewidth=1, label=f'Spectrum {j + 1}-{j + 2}')
plt.plot(m[-1], intensity_at_specific_wavelength[-1], marker='o', label=f'Spectrum {len(dsets2)}')
plt.xlabel('Time (seconds)', fontsize=14, fontweight="bold")
plt.ylabel(f'Intensity at {specific_wavelength} nm', fontsize=14, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12, direction="in")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend(loc="lower right", fontsize=8)
plt.title(f'Intensity Variation at {specific_wavelength} nm Over Time', fontsize=16, fontweight="bold")
plt.show()
folder_name = 'Ref_Spectrum_Data'
folder_path = os.path.join(DIRPATH, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
#csv_file_path1 = os.path.join(folder_path, "Ref_QE65000_Spectrometer.csv")
#pd.DataFrame(data_wl_wavelength).to_csv(csv_file_path1, header=None, index=None)

##############################################################################################################################################################

date = "2024-03-23T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/hera/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 8
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    print("\nKeys: %s" % f.keys())
    keys = list(f.keys())
key = keys[16]
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    g = f[key]
    indices = sorted([d.replace("spectrum_", "") for d in g.keys()])
    dsets = ["{0}".format(n) for n in indices]
    print("attributes for test:", g[dsets[0]].attrs.keys())
    print(dsets)
    sp = 0
    h = g[dsets[sp]]
    print("\nh:", h)
    indices2 = sorted(h.keys())
    dsets1 = ["{0}".format(n) for n in indices2]
    dsets2 = sorted(dsets1, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    sp2 = 0 
    print("\nspectra:", dsets2)
    laser_wavelength = 650  
    wavelengths = np.array(h[dsets2[sp2]].attrs["wavelengths"])
    start_times = np.array(h[dsets2[sp2]].attrs["creation_timestamp"]) 
    integration_time = np.array(h[dsets2[sp2]].attrs["integration_time"]) 
    creation_time = np.array(h[dsets2[sp2]].attrs["creation_timestamp"])  
    time_interval = np.array(h[dsets2[sp2]].attrs["time_interval"])
    information = np.array(h[dsets2[sp2]].attrs["information"])
    data = np.zeros((len(h[dsets2[sp2]]), len(dsets2)), dtype=float)
    creation_times = np.array((len(h[dsets2[sp2]].attrs["creation_timestamp"]), len(dsets2)), dtype=str)
    m = np.array([], dtype=np.float32)
    for j in range(len(dsets2)):
        data[:, j] = np.array(h[dsets2[j]]) - np.array(h[dsets2[j]].attrs["background"])
        starttime = h[dsets2[sp2]].attrs["creation_timestamp"]
        time = h[dsets2[j]].attrs["creation_timestamp"]
        strstarttime = starttime
        strtime = time
        newstarttime = strstarttime.split("T")[-1]
        newtime = strtime.split("T")[-1]
        newstarttime = newstarttime.replace(date, "")
        newtime = newtime.replace(date, "")
        start_time_s = pd.Timestamp(newstarttime).timestamp()
        time_s = pd.Timestamp(newtime).timestamp()
        delta_time = time_s - start_time_s
        m = np.append(m, delta_time)
t = np.reshape(m, (1, len(dsets2)))
data_average = np.sum(data, 1) / len(dsets2)
data_wl_wavelength_2 = np.zeros((len(data_average), 2), dtype=float)
data_wl_wavelength_2[:, 0] = wavelengths
data_wl_wavelength_2[:, 1] = data_average
plot_spectrum(data_wl_wavelength_2[:, 0], data_wl_wavelength_2[:, 1],
              x_label='Wavelength (nm)', y_label='Intensity (a.u.)',
              title='Average Spectrum',
              legend_label='Intensity at Wavelength')
specific_wavelength = laser_wavelength
wavelength_index = np.argmin(np.abs(wavelengths - specific_wavelength))
intensity_at_specific_wavelength = data[wavelength_index, :]
plt.figure(figsize=(8, 6), dpi=600)
for j in range(len(dsets2) - 1):
    plt.plot([m[j], m[j + 1]], [intensity_at_specific_wavelength[j], intensity_at_specific_wavelength[j + 1]],
             marker='o', linestyle='-', linewidth=1, label=f'Spectrum {j + 1}-{j + 2}')
plt.plot(m[-1], intensity_at_specific_wavelength[-1], marker='o', label=f'Spectrum {len(dsets2)}')
plt.xlabel('Time (seconds)', fontsize=14, fontweight="bold")
plt.ylabel(f'Intensity at {specific_wavelength} nm', fontsize=14, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12, direction="in")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend(loc="lower right", fontsize=8)
plt.title(f'Intensity Variation at {specific_wavelength} nm Over Time', fontsize=16, fontweight="bold")
plt.show()
folder_name = 'Ref_Spectrum_Data'
folder_path = os.path.join(DIRPATH, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
#csv_file_path2 = os.path.join(folder_path, "Ref_USB2000_Spectrometer.csv")
#pd.DataFrame(data_wl_wavelength).to_csv(csv_file_path2, header=None, index=None)

##############################################################################################################################################################

df_normalized = pd.DataFrame({
    'Wavelength': data_wl_wavelength_1[:, 0],
    'Intensity_1': data_wl_wavelength_1[:, 1],
    'Intensity_2': data_wl_wavelength_2[:, 1],
    'Normalized_Intensity': abs(data_wl_wavelength_2[:, 1] / data_wl_wavelength_1[:, 1]) 
})
plot_spectrum(df_normalized['Wavelength'], df_normalized['Normalized_Intensity'],
              x_label='Wavelength (nm)', y_label='Normalized Intensity',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')
csv_file_path_normalized = os.path.join(folder_path, "Normalized_Intensity.csv")
df_normalized.to_csv(csv_file_path_normalized, index=None)

##############################################################################################################################################################

input_wavelength = float(input("Enter the wavelength: "))
power_percentage = float(input(f"Enter the power percentage at {input_wavelength} nm: "))
closest_wavelength_index = np.argmin(np.abs(df_normalized['Wavelength'] - input_wavelength))
closest_wavelength = df_normalized.loc[closest_wavelength_index, 'Wavelength']
scaled_factor = power_percentage / df_normalized.loc[closest_wavelength_index, 'Normalized_Intensity']
df_normalized['Percentage_Transmission_4'] = df_normalized['Normalized_Intensity'] * scaled_factor
df_filtered = df_normalized[(df_normalized['Wavelength'] > 400)] #& (df_normalized['Wavelength'] < 1000)]
plot_spectrum(df_filtered['Wavelength'], df_filtered['Percentage_Transmission_4'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')
#csv_file_path_percentage_transmission = os.path.join(folder_path, "Percentage_Transmission_2.csv")
#df_normalized[['Wavelength', 'Percentage_Transmission']].to_csv(csv_file_path_percentage_transmission, index=None)


##############################################################################################################################################################

user_input = input("Press Enter to continue or type 'exit' to quit: ")
if user_input.lower() == 'exit':
    sys.exit()
    
##############################################################################################################################################################

date = "2024-03-20T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/hera/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 5
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    print("\nKeys: %s" % f.keys())
    keys = list(f.keys())
key = keys[0]
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    g = f[key]
    indices = sorted([d.replace("spectrum_", "") for d in g.keys()])
    dsets = ["{0}".format(n) for n in indices]
    print("attributes for test:", g[dsets[0]].attrs.keys())
    print(dsets)
    sp = 0
    h = g[dsets[sp]]
    print("\nh:", h)
    indices2 = sorted(h.keys())
    dsets1 = ["{0}".format(n) for n in indices2]
    dsets2 = sorted(dsets1, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    sp2 = 0 
    print("\nspectra:", dsets2)
    laser_wavelength = 650  
    wavelengths = np.array(h[dsets2[sp2]].attrs["wavelengths"])
    start_times = np.array(h[dsets2[sp2]].attrs["creation_timestamp"]) 
    integration_time = np.array(h[dsets2[sp2]].attrs["integration_time"]) 
    creation_time = np.array(h[dsets2[sp2]].attrs["creation_timestamp"])  
    time_interval = np.array(h[dsets2[sp2]].attrs["time_interval"])
    information = np.array(h[dsets2[sp2]].attrs["information"])
    data = np.zeros((len(h[dsets2[sp2]]), len(dsets2)), dtype=float)
    creation_times = np.array((len(h[dsets2[sp2]].attrs["creation_timestamp"]), len(dsets2)), dtype=str)
    m = np.array([], dtype=np.float32)
    for j in range(len(dsets2)):
        data[:, j] = np.array(h[dsets2[j]]) - np.array(h[dsets2[j]].attrs["background"])
        starttime = h[dsets2[sp2]].attrs["creation_timestamp"]
        time = h[dsets2[j]].attrs["creation_timestamp"]
        strstarttime = starttime
        strtime = time
        newstarttime = strstarttime.split("T")[-1]
        newtime = strtime.split("T")[-1]
        newstarttime = newstarttime.replace(date, "")
        newtime = newtime.replace(date, "")
        start_time_s = pd.Timestamp(newstarttime).timestamp()
        time_s = pd.Timestamp(newtime).timestamp()
        delta_time = time_s - start_time_s
        m = np.append(m, delta_time)
t = np.reshape(m, (1, len(dsets2)))
data_average = np.sum(data, 1) / len(dsets2)
data_wl_wavelength_1 = np.zeros((len(data_average), 2), dtype=float)
data_wl_wavelength_1[:, 0] = wavelengths
data_wl_wavelength_1[:, 1] = data_average
plot_spectrum(data_wl_wavelength_1[:, 0], data_wl_wavelength_1[:, 1],
              x_label='Wavelength (nm)', y_label='Intensity (a.u.)',
              title='Average Spectrum',
              legend_label='Intensity at Wavelength')
specific_wavelength = laser_wavelength
wavelength_index = np.argmin(np.abs(wavelengths - specific_wavelength))
intensity_at_specific_wavelength = data[wavelength_index, :]
plt.figure(figsize=(8, 6), dpi=600)
for j in range(len(dsets2) - 1):
    plt.plot([m[j], m[j + 1]], [intensity_at_specific_wavelength[j], intensity_at_specific_wavelength[j + 1]],
             marker='o', linestyle='-', linewidth=1, label=f'Spectrum {j + 1}-{j + 2}')
plt.plot(m[-1], intensity_at_specific_wavelength[-1], marker='o', label=f'Spectrum {len(dsets2)}')
plt.xlabel('Time (seconds)', fontsize=14, fontweight="bold")
plt.ylabel(f'Intensity at {specific_wavelength} nm', fontsize=14, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12, direction="in")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend(loc="lower right", fontsize=8)
plt.title(f'Intensity Variation at {specific_wavelength} nm Over Time', fontsize=16, fontweight="bold")
plt.show()
folder_name = 'Ref_Spectrum_Data'
folder_path = os.path.join(DIRPATH, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
#csv_file_path1 = os.path.join(folder_path, "Ref_QE65000_Spectrometer.csv")
#pd.DataFrame(data_wl_wavelength).to_csv(csv_file_path1, header=None, index=None)

##############################################################################################################################################################

date = "2024-03-23T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/hera/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 8
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    print("\nKeys: %s" % f.keys())
    keys = list(f.keys())
key = keys[16]
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    g = f[key]
    indices = sorted([d.replace("spectrum_", "") for d in g.keys()])
    dsets = ["{0}".format(n) for n in indices]
    print("attributes for test:", g[dsets[0]].attrs.keys())
    print(dsets)
    sp = 0
    h = g[dsets[sp]]
    print("\nh:", h)
    indices2 = sorted(h.keys())
    dsets1 = ["{0}".format(n) for n in indices2]
    dsets2 = sorted(dsets1, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    sp2 = 0 
    print("\nspectra:", dsets2)
    laser_wavelength = 650  
    wavelengths = np.array(h[dsets2[sp2]].attrs["wavelengths"])
    start_times = np.array(h[dsets2[sp2]].attrs["creation_timestamp"]) 
    integration_time = np.array(h[dsets2[sp2]].attrs["integration_time"]) 
    creation_time = np.array(h[dsets2[sp2]].attrs["creation_timestamp"])  
    time_interval = np.array(h[dsets2[sp2]].attrs["time_interval"])
    information = np.array(h[dsets2[sp2]].attrs["information"])
    data = np.zeros((len(h[dsets2[sp2]]), len(dsets2)), dtype=float)
    creation_times = np.array((len(h[dsets2[sp2]].attrs["creation_timestamp"]), len(dsets2)), dtype=str)
    m = np.array([], dtype=np.float32)
    for j in range(len(dsets2)):
        data[:, j] = np.array(h[dsets2[j]]) - np.array(h[dsets2[j]].attrs["background"])
        starttime = h[dsets2[sp2]].attrs["creation_timestamp"]
        time = h[dsets2[j]].attrs["creation_timestamp"]
        strstarttime = starttime
        strtime = time
        newstarttime = strstarttime.split("T")[-1]
        newtime = strtime.split("T")[-1]
        newstarttime = newstarttime.replace(date, "")
        newtime = newtime.replace(date, "")
        start_time_s = pd.Timestamp(newstarttime).timestamp()
        time_s = pd.Timestamp(newtime).timestamp()
        delta_time = time_s - start_time_s
        m = np.append(m, delta_time)
t = np.reshape(m, (1, len(dsets2)))
data_average = np.sum(data, 1) / len(dsets2)
data_wl_wavelength_2 = np.zeros((len(data_average), 2), dtype=float)
data_wl_wavelength_2[:, 0] = wavelengths
data_wl_wavelength_2[:, 1] = data_average
plot_spectrum(data_wl_wavelength_2[:, 0], data_wl_wavelength_2[:, 1],
              x_label='Wavelength (nm)', y_label='Intensity (a.u.)',
              title='Average Spectrum',
              legend_label='Intensity at Wavelength')
specific_wavelength = laser_wavelength
wavelength_index = np.argmin(np.abs(wavelengths - specific_wavelength))
intensity_at_specific_wavelength = data[wavelength_index, :]
plt.figure(figsize=(8, 6), dpi=600)
for j in range(len(dsets2) - 1):
    plt.plot([m[j], m[j + 1]], [intensity_at_specific_wavelength[j], intensity_at_specific_wavelength[j + 1]],
             marker='o', linestyle='-', linewidth=1, label=f'Spectrum {j + 1}-{j + 2}')
plt.plot(m[-1], intensity_at_specific_wavelength[-1], marker='o', label=f'Spectrum {len(dsets2)}')
plt.xlabel('Time (seconds)', fontsize=14, fontweight="bold")
plt.ylabel(f'Intensity at {specific_wavelength} nm', fontsize=14, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12, direction="in")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend(loc="lower right", fontsize=8)
plt.title(f'Intensity Variation at {specific_wavelength} nm Over Time', fontsize=16, fontweight="bold")
plt.show()
folder_name = 'Ref_Spectrum_Data'
folder_path = os.path.join(DIRPATH, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
#csv_file_path2 = os.path.join(folder_path, "Ref_USB2000_Spectrometer.csv")
#pd.DataFrame(data_wl_wavelength).to_csv(csv_file_path2, header=None, index=None)

##############################################################################################################################################################

df_normalized = pd.DataFrame({
    'Wavelength': data_wl_wavelength_1[:, 0],
    'Intensity_1': data_wl_wavelength_1[:, 1],
    'Intensity_2': data_wl_wavelength_2[:, 1],
    'Normalized_Intensity': abs(data_wl_wavelength_2[:, 1] / data_wl_wavelength_1[:, 1]) 
})
plot_spectrum(df_normalized['Wavelength'], df_normalized['Normalized_Intensity'],
              x_label='Wavelength (nm)', y_label='Normalized Intensity',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')
#csv_file_path_normalized = os.path.join(folder_path, "Normalized_Intensity.csv")
#df_normalized.to_csv(csv_file_path_normalized, index=None)

##############################################################################################################################################################

input_wavelength = float(input("Enter the wavelength: "))
power_percentage = float(input(f"Enter the power percentage at {input_wavelength} nm: "))
closest_wavelength_index = np.argmin(np.abs(df_normalized['Wavelength'] - input_wavelength))
closest_wavelength = df_normalized.loc[closest_wavelength_index, 'Wavelength']
scaled_factor = power_percentage / df_normalized.loc[closest_wavelength_index, 'Normalized_Intensity']
df_normalized['Percentage_Transmission_5'] = df_normalized['Normalized_Intensity'] * scaled_factor
df_filtered = df_normalized[(df_normalized['Wavelength'] > 400)] #& (df_normalized['Wavelength'] < 1000)]
plot_spectrum(df_filtered['Wavelength'], df_filtered['Percentage_Transmission_5'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')
#csv_file_path_percentage_transmission = os.path.join(folder_path, "Percentage_Transmission.csv")
#df_normalized[['Wavelength', 'Percentage_Transmission']].to_csv(csv_file_path_percentage_transmission, index=None)


##############################################################################################################################################################

user_input = input("Press Enter to continue or type 'exit' to quit: ")
if user_input.lower() == 'exit':
    sys.exit()

##############################################################################################################################################################

date = "2024-03-20T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/hera/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 5
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    print("\nKeys: %s" % f.keys())
    keys = list(f.keys())
key = keys[0]
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    g = f[key]
    indices = sorted([d.replace("spectrum_", "") for d in g.keys()])
    dsets = ["{0}".format(n) for n in indices]
    print("attributes for test:", g[dsets[0]].attrs.keys())
    print(dsets)
    sp = 0
    h = g[dsets[sp]]
    print("\nh:", h)
    indices2 = sorted(h.keys())
    dsets1 = ["{0}".format(n) for n in indices2]
    dsets2 = sorted(dsets1, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    sp2 = 0 
    print("\nspectra:", dsets2)
    laser_wavelength = 650  
    wavelengths = np.array(h[dsets2[sp2]].attrs["wavelengths"])
    start_times = np.array(h[dsets2[sp2]].attrs["creation_timestamp"]) 
    integration_time = np.array(h[dsets2[sp2]].attrs["integration_time"]) 
    creation_time = np.array(h[dsets2[sp2]].attrs["creation_timestamp"])  
    time_interval = np.array(h[dsets2[sp2]].attrs["time_interval"])
    information = np.array(h[dsets2[sp2]].attrs["information"])
    data = np.zeros((len(h[dsets2[sp2]]), len(dsets2)), dtype=float)
    creation_times = np.array((len(h[dsets2[sp2]].attrs["creation_timestamp"]), len(dsets2)), dtype=str)
    m = np.array([], dtype=np.float32)
    for j in range(len(dsets2)):
        data[:, j] = np.array(h[dsets2[j]]) - np.array(h[dsets2[j]].attrs["background"])
        starttime = h[dsets2[sp2]].attrs["creation_timestamp"]
        time = h[dsets2[j]].attrs["creation_timestamp"]
        strstarttime = starttime
        strtime = time
        newstarttime = strstarttime.split("T")[-1]
        newtime = strtime.split("T")[-1]
        newstarttime = newstarttime.replace(date, "")
        newtime = newtime.replace(date, "")
        start_time_s = pd.Timestamp(newstarttime).timestamp()
        time_s = pd.Timestamp(newtime).timestamp()
        delta_time = time_s - start_time_s
        m = np.append(m, delta_time)
t = np.reshape(m, (1, len(dsets2)))
data_average = np.sum(data, 1) / len(dsets2)
data_wl_wavelength_1 = np.zeros((len(data_average), 2), dtype=float)
data_wl_wavelength_1[:, 0] = wavelengths
data_wl_wavelength_1[:, 1] = data_average
plot_spectrum(data_wl_wavelength_1[:, 0], data_wl_wavelength_1[:, 1],
              x_label='Wavelength (nm)', y_label='Intensity (a.u.)',
              title='Average Spectrum',
              legend_label='Intensity at Wavelength')
specific_wavelength = laser_wavelength
wavelength_index = np.argmin(np.abs(wavelengths - specific_wavelength))
intensity_at_specific_wavelength = data[wavelength_index, :]
plt.figure(figsize=(8, 6), dpi=600)
for j in range(len(dsets2) - 1):
    plt.plot([m[j], m[j + 1]], [intensity_at_specific_wavelength[j], intensity_at_specific_wavelength[j + 1]],
             marker='o', linestyle='-', linewidth=1, label=f'Spectrum {j + 1}-{j + 2}')
plt.plot(m[-1], intensity_at_specific_wavelength[-1], marker='o', label=f'Spectrum {len(dsets2)}')
plt.xlabel('Time (seconds)', fontsize=14, fontweight="bold")
plt.ylabel(f'Intensity at {specific_wavelength} nm', fontsize=14, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12, direction="in")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend(loc="lower right", fontsize=8)
plt.title(f'Intensity Variation at {specific_wavelength} nm Over Time', fontsize=16, fontweight="bold")
plt.show()
folder_name = 'Ref_Spectrum_Data'
folder_path = os.path.join(DIRPATH, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
#csv_file_path1 = os.path.join(folder_path, "Ref_QE65000_Spectrometer.csv")
#pd.DataFrame(data_wl_wavelength).to_csv(csv_file_path1, header=None, index=None)

##############################################################################################################################################################

date = "2024-03-23T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/hera/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 8
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    print("\nKeys: %s" % f.keys())
    keys = list(f.keys())
key = keys[16]
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    g = f[key]
    indices = sorted([d.replace("spectrum_", "") for d in g.keys()])
    dsets = ["{0}".format(n) for n in indices]
    print("attributes for test:", g[dsets[0]].attrs.keys())
    print(dsets)
    sp = 0
    h = g[dsets[sp]]
    print("\nh:", h)
    indices2 = sorted(h.keys())
    dsets1 = ["{0}".format(n) for n in indices2]
    dsets2 = sorted(dsets1, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    sp2 = 0 
    print("\nspectra:", dsets2)
    laser_wavelength = 650  
    wavelengths = np.array(h[dsets2[sp2]].attrs["wavelengths"])
    start_times = np.array(h[dsets2[sp2]].attrs["creation_timestamp"]) 
    integration_time = np.array(h[dsets2[sp2]].attrs["integration_time"]) 
    creation_time = np.array(h[dsets2[sp2]].attrs["creation_timestamp"])  
    time_interval = np.array(h[dsets2[sp2]].attrs["time_interval"])
    information = np.array(h[dsets2[sp2]].attrs["information"])
    data = np.zeros((len(h[dsets2[sp2]]), len(dsets2)), dtype=float)
    creation_times = np.array((len(h[dsets2[sp2]].attrs["creation_timestamp"]), len(dsets2)), dtype=str)
    m = np.array([], dtype=np.float32)
    for j in range(len(dsets2)):
        data[:, j] = np.array(h[dsets2[j]]) - np.array(h[dsets2[j]].attrs["background"])
        starttime = h[dsets2[sp2]].attrs["creation_timestamp"]
        time = h[dsets2[j]].attrs["creation_timestamp"]
        strstarttime = starttime
        strtime = time
        newstarttime = strstarttime.split("T")[-1]
        newtime = strtime.split("T")[-1]
        newstarttime = newstarttime.replace(date, "")
        newtime = newtime.replace(date, "")
        start_time_s = pd.Timestamp(newstarttime).timestamp()
        time_s = pd.Timestamp(newtime).timestamp()
        delta_time = time_s - start_time_s
        m = np.append(m, delta_time)
t = np.reshape(m, (1, len(dsets2)))
data_average = np.sum(data, 1) / len(dsets2)
data_wl_wavelength_2 = np.zeros((len(data_average), 2), dtype=float)
data_wl_wavelength_2[:, 0] = wavelengths
data_wl_wavelength_2[:, 1] = data_average
plot_spectrum(data_wl_wavelength_2[:, 0], data_wl_wavelength_2[:, 1],
              x_label='Wavelength (nm)', y_label='Intensity (a.u.)',
              title='Average Spectrum',
              legend_label='Intensity at Wavelength')
specific_wavelength = laser_wavelength
wavelength_index = np.argmin(np.abs(wavelengths - specific_wavelength))
intensity_at_specific_wavelength = data[wavelength_index, :]
plt.figure(figsize=(8, 6), dpi=600)
for j in range(len(dsets2) - 1):
    plt.plot([m[j], m[j + 1]], [intensity_at_specific_wavelength[j], intensity_at_specific_wavelength[j + 1]],
             marker='o', linestyle='-', linewidth=1, label=f'Spectrum {j + 1}-{j + 2}')
plt.plot(m[-1], intensity_at_specific_wavelength[-1], marker='o', label=f'Spectrum {len(dsets2)}')
plt.xlabel('Time (seconds)', fontsize=14, fontweight="bold")
plt.ylabel(f'Intensity at {specific_wavelength} nm', fontsize=14, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12, direction="in")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend(loc="lower right", fontsize=8)
plt.title(f'Intensity Variation at {specific_wavelength} nm Over Time', fontsize=16, fontweight="bold")
plt.show()
folder_name = 'Ref_Spectrum_Data'
folder_path = os.path.join(DIRPATH, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
#csv_file_path2 = os.path.join(folder_path, "Ref_USB2000_Spectrometer.csv")
#pd.DataFrame(data_wl_wavelength).to_csv(csv_file_path2, header=None, index=None)

##############################################################################################################################################################

df_normalized = pd.DataFrame({
    'Wavelength': data_wl_wavelength_1[:, 0],
    'Intensity_1': data_wl_wavelength_1[:, 1],
    'Intensity_2': data_wl_wavelength_2[:, 1],
    'Normalized_Intensity': abs(data_wl_wavelength_2[:, 1] / data_wl_wavelength_1[:, 1]) 
})
plot_spectrum(df_normalized['Wavelength'], df_normalized['Normalized_Intensity'],
              x_label='Wavelength (nm)', y_label='Normalized Intensity',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')
#csv_file_path_normalized = os.path.join(folder_path, "Normalized_Intensity.csv")
#df_normalized.to_csv(csv_file_path_normalized, index=None)

##############################################################################################################################################################

input_wavelength = float(input("Enter the wavelength: "))
power_percentage = float(input(f"Enter the power percentage at {input_wavelength} nm: "))
closest_wavelength_index = np.argmin(np.abs(df_normalized['Wavelength'] - input_wavelength))
closest_wavelength = df_normalized.loc[closest_wavelength_index, 'Wavelength']
scaled_factor = power_percentage / df_normalized.loc[closest_wavelength_index, 'Normalized_Intensity']
df_normalized['Percentage_Transmission_6'] = df_normalized['Normalized_Intensity'] * scaled_factor
df_filtered = df_normalized[(df_normalized['Wavelength'] > 400)] #& (df_normalized['Wavelength'] < 1000)]
plot_spectrum(df_filtered['Wavelength'], df_filtered['Percentage_Transmission_6'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')
#csv_file_path_percentage_transmission = os.path.join(folder_path, "Percentage_Transmission_2.csv")
#df_normalized[['Wavelength', 'Percentage_Transmission']].to_csv(csv_file_path_percentage_transmission, index=None)


##############################################################################################################################################################

user_input = input("Press Enter to continue or type 'exit' to quit: ")
if user_input.lower() == 'exit':
    sys.exit()

##############################################################################################################################################################

date = "2024-03-20T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/hera/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 5
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    print("\nKeys: %s" % f.keys())
    keys = list(f.keys())
key = keys[0]
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    g = f[key]
    indices = sorted([d.replace("spectrum_", "") for d in g.keys()])
    dsets = ["{0}".format(n) for n in indices]
    print("attributes for test:", g[dsets[0]].attrs.keys())
    print(dsets)
    sp = 0
    h = g[dsets[sp]]
    print("\nh:", h)
    indices2 = sorted(h.keys())
    dsets1 = ["{0}".format(n) for n in indices2]
    dsets2 = sorted(dsets1, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    sp2 = 0 
    print("\nspectra:", dsets2)
    laser_wavelength = 650  
    wavelengths = np.array(h[dsets2[sp2]].attrs["wavelengths"])
    start_times = np.array(h[dsets2[sp2]].attrs["creation_timestamp"]) 
    integration_time = np.array(h[dsets2[sp2]].attrs["integration_time"]) 
    creation_time = np.array(h[dsets2[sp2]].attrs["creation_timestamp"])  
    time_interval = np.array(h[dsets2[sp2]].attrs["time_interval"])
    information = np.array(h[dsets2[sp2]].attrs["information"])
    data = np.zeros((len(h[dsets2[sp2]]), len(dsets2)), dtype=float)
    creation_times = np.array((len(h[dsets2[sp2]].attrs["creation_timestamp"]), len(dsets2)), dtype=str)
    m = np.array([], dtype=np.float32)
    for j in range(len(dsets2)):
        data[:, j] = np.array(h[dsets2[j]]) - np.array(h[dsets2[j]].attrs["background"])
        starttime = h[dsets2[sp2]].attrs["creation_timestamp"]
        time = h[dsets2[j]].attrs["creation_timestamp"]
        strstarttime = starttime
        strtime = time
        newstarttime = strstarttime.split("T")[-1]
        newtime = strtime.split("T")[-1]
        newstarttime = newstarttime.replace(date, "")
        newtime = newtime.replace(date, "")
        start_time_s = pd.Timestamp(newstarttime).timestamp()
        time_s = pd.Timestamp(newtime).timestamp()
        delta_time = time_s - start_time_s
        m = np.append(m, delta_time)
t = np.reshape(m, (1, len(dsets2)))
data_average = np.sum(data, 1) / len(dsets2)
data_wl_wavelength_1 = np.zeros((len(data_average), 2), dtype=float)
data_wl_wavelength_1[:, 0] = wavelengths
data_wl_wavelength_1[:, 1] = data_average
plot_spectrum(data_wl_wavelength_1[:, 0], data_wl_wavelength_1[:, 1],
              x_label='Wavelength (nm)', y_label='Intensity (a.u.)',
              title='Average Spectrum',
              legend_label='Intensity at Wavelength')
specific_wavelength = laser_wavelength
wavelength_index = np.argmin(np.abs(wavelengths - specific_wavelength))
intensity_at_specific_wavelength = data[wavelength_index, :]
plt.figure(figsize=(8, 6), dpi=600)
for j in range(len(dsets2) - 1):
    plt.plot([m[j], m[j + 1]], [intensity_at_specific_wavelength[j], intensity_at_specific_wavelength[j + 1]],
             marker='o', linestyle='-', linewidth=1, label=f'Spectrum {j + 1}-{j + 2}')
plt.plot(m[-1], intensity_at_specific_wavelength[-1], marker='o', label=f'Spectrum {len(dsets2)}')
plt.xlabel('Time (seconds)', fontsize=14, fontweight="bold")
plt.ylabel(f'Intensity at {specific_wavelength} nm', fontsize=14, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12, direction="in")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend(loc="lower right", fontsize=8)
plt.title(f'Intensity Variation at {specific_wavelength} nm Over Time', fontsize=16, fontweight="bold")
plt.show()
folder_name = 'Ref_Spectrum_Data'
folder_path = os.path.join(DIRPATH, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
#csv_file_path1 = os.path.join(folder_path, "Ref_QE65000_Spectrometer.csv")
#pd.DataFrame(data_wl_wavelength).to_csv(csv_file_path1, header=None, index=None)

##############################################################################################################################################################

date = "2024-03-23T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/hera/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 8
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    print("\nKeys: %s" % f.keys())
    keys = list(f.keys())
key = keys[16]
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    g = f[key]
    indices = sorted([d.replace("spectrum_", "") for d in g.keys()])
    dsets = ["{0}".format(n) for n in indices]
    print("attributes for test:", g[dsets[0]].attrs.keys())
    print(dsets)
    sp = 0
    h = g[dsets[sp]]
    print("\nh:", h)
    indices2 = sorted(h.keys())
    dsets1 = ["{0}".format(n) for n in indices2]
    dsets2 = sorted(dsets1, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    sp2 = 0 
    print("\nspectra:", dsets2)
    laser_wavelength = 650  
    wavelengths = np.array(h[dsets2[sp2]].attrs["wavelengths"])
    start_times = np.array(h[dsets2[sp2]].attrs["creation_timestamp"]) 
    integration_time = np.array(h[dsets2[sp2]].attrs["integration_time"]) 
    creation_time = np.array(h[dsets2[sp2]].attrs["creation_timestamp"])  
    time_interval = np.array(h[dsets2[sp2]].attrs["time_interval"])
    information = np.array(h[dsets2[sp2]].attrs["information"])
    data = np.zeros((len(h[dsets2[sp2]]), len(dsets2)), dtype=float)
    creation_times = np.array((len(h[dsets2[sp2]].attrs["creation_timestamp"]), len(dsets2)), dtype=str)
    m = np.array([], dtype=np.float32)
    for j in range(len(dsets2)):
        data[:, j] = np.array(h[dsets2[j]]) - np.array(h[dsets2[j]].attrs["background"])
        starttime = h[dsets2[sp2]].attrs["creation_timestamp"]
        time = h[dsets2[j]].attrs["creation_timestamp"]
        strstarttime = starttime
        strtime = time
        newstarttime = strstarttime.split("T")[-1]
        newtime = strtime.split("T")[-1]
        newstarttime = newstarttime.replace(date, "")
        newtime = newtime.replace(date, "")
        start_time_s = pd.Timestamp(newstarttime).timestamp()
        time_s = pd.Timestamp(newtime).timestamp()
        delta_time = time_s - start_time_s
        m = np.append(m, delta_time)
t = np.reshape(m, (1, len(dsets2)))
data_average = np.sum(data, 1) / len(dsets2)
data_wl_wavelength_2 = np.zeros((len(data_average), 2), dtype=float)
data_wl_wavelength_2[:, 0] = wavelengths
data_wl_wavelength_2[:, 1] = data_average
plot_spectrum(data_wl_wavelength_2[:, 0], data_wl_wavelength_2[:, 1],
              x_label='Wavelength (nm)', y_label='Intensity (a.u.)',
              title='Average Spectrum',
              legend_label='Intensity at Wavelength')
specific_wavelength = laser_wavelength
wavelength_index = np.argmin(np.abs(wavelengths - specific_wavelength))
intensity_at_specific_wavelength = data[wavelength_index, :]
plt.figure(figsize=(8, 6), dpi=600)
for j in range(len(dsets2) - 1):
    plt.plot([m[j], m[j + 1]], [intensity_at_specific_wavelength[j], intensity_at_specific_wavelength[j + 1]],
             marker='o', linestyle='-', linewidth=1, label=f'Spectrum {j + 1}-{j + 2}')
plt.plot(m[-1], intensity_at_specific_wavelength[-1], marker='o', label=f'Spectrum {len(dsets2)}')
plt.xlabel('Time (seconds)', fontsize=14, fontweight="bold")
plt.ylabel(f'Intensity at {specific_wavelength} nm', fontsize=14, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12, direction="in")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend(loc="lower right", fontsize=8)
plt.title(f'Intensity Variation at {specific_wavelength} nm Over Time', fontsize=16, fontweight="bold")
plt.show()
folder_name = 'Ref_Spectrum_Data'
folder_path = os.path.join(DIRPATH, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
#csv_file_path2 = os.path.join(folder_path, "Ref_USB2000_Spectrometer.csv")
#pd.DataFrame(data_wl_wavelength).to_csv(csv_file_path2, header=None, index=None)

##############################################################################################################################################################

df_normalized = pd.DataFrame({
    'Wavelength': data_wl_wavelength_1[:, 0],
    'Intensity_1': data_wl_wavelength_1[:, 1],
    'Intensity_2': data_wl_wavelength_2[:, 1],
    'Normalized_Intensity': abs(data_wl_wavelength_2[:, 1] / data_wl_wavelength_1[:, 1]) 
})
plot_spectrum(df_normalized['Wavelength'], df_normalized['Normalized_Intensity'],
              x_label='Wavelength (nm)', y_label='Normalized Intensity',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')
#csv_file_path_normalized = os.path.join(folder_path, "Normalized_Intensity.csv")
#df_normalized.to_csv(csv_file_path_normalized, index=None)

##############################################################################################################################################################

input_wavelength = float(input("Enter the wavelength: "))
power_percentage = float(input(f"Enter the power percentage at {input_wavelength} nm: "))
closest_wavelength_index = np.argmin(np.abs(df_normalized['Wavelength'] - input_wavelength))
closest_wavelength = df_normalized.loc[closest_wavelength_index, 'Wavelength']
scaled_factor = power_percentage / df_normalized.loc[closest_wavelength_index, 'Normalized_Intensity']
df_normalized['Percentage_Transmission_7'] = df_normalized['Normalized_Intensity'] * scaled_factor
df_filtered = df_normalized[(df_normalized['Wavelength'] > 400)] #& (df_normalized['Wavelength'] < 1000)]
plot_spectrum(df_filtered['Wavelength'], df_filtered['Percentage_Transmission_7'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')
#csv_file_path_percentage_transmission = os.path.join(folder_path, "Percentage_Transmission.csv")
#df_normalized[['Wavelength', 'Percentage_Transmission']].to_csv(csv_file_path_percentage_transmission, index=None)


##############################################################################################################################################################

user_input = input("Press Enter to continue or type 'exit' to quit: ")
if user_input.lower() == 'exit':
    sys.exit()

##############################################################################################################################################################

date = "2024-03-20T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/hera/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 5
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    print("\nKeys: %s" % f.keys())
    keys = list(f.keys())
key = keys[0]
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    g = f[key]
    indices = sorted([d.replace("spectrum_", "") for d in g.keys()])
    dsets = ["{0}".format(n) for n in indices]
    print("attributes for test:", g[dsets[0]].attrs.keys())
    print(dsets)
    sp = 0
    h = g[dsets[sp]]
    print("\nh:", h)
    indices2 = sorted(h.keys())
    dsets1 = ["{0}".format(n) for n in indices2]
    dsets2 = sorted(dsets1, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    sp2 = 0 
    print("\nspectra:", dsets2)
    laser_wavelength = 650  
    wavelengths = np.array(h[dsets2[sp2]].attrs["wavelengths"])
    start_times = np.array(h[dsets2[sp2]].attrs["creation_timestamp"]) 
    integration_time = np.array(h[dsets2[sp2]].attrs["integration_time"]) 
    creation_time = np.array(h[dsets2[sp2]].attrs["creation_timestamp"])  
    time_interval = np.array(h[dsets2[sp2]].attrs["time_interval"])
    information = np.array(h[dsets2[sp2]].attrs["information"])
    data = np.zeros((len(h[dsets2[sp2]]), len(dsets2)), dtype=float)
    creation_times = np.array((len(h[dsets2[sp2]].attrs["creation_timestamp"]), len(dsets2)), dtype=str)
    m = np.array([], dtype=np.float32)
    for j in range(len(dsets2)):
        data[:, j] = np.array(h[dsets2[j]]) - np.array(h[dsets2[j]].attrs["background"])
        starttime = h[dsets2[sp2]].attrs["creation_timestamp"]
        time = h[dsets2[j]].attrs["creation_timestamp"]
        strstarttime = starttime
        strtime = time
        newstarttime = strstarttime.split("T")[-1]
        newtime = strtime.split("T")[-1]
        newstarttime = newstarttime.replace(date, "")
        newtime = newtime.replace(date, "")
        start_time_s = pd.Timestamp(newstarttime).timestamp()
        time_s = pd.Timestamp(newtime).timestamp()
        delta_time = time_s - start_time_s
        m = np.append(m, delta_time)
t = np.reshape(m, (1, len(dsets2)))
data_average = np.sum(data, 1) / len(dsets2)
data_wl_wavelength_1 = np.zeros((len(data_average), 2), dtype=float)
data_wl_wavelength_1[:, 0] = wavelengths
data_wl_wavelength_1[:, 1] = data_average
plot_spectrum(data_wl_wavelength_1[:, 0], data_wl_wavelength_1[:, 1],
              x_label='Wavelength (nm)', y_label='Intensity (a.u.)',
              title='Average Spectrum',
              legend_label='Intensity at Wavelength')
specific_wavelength = laser_wavelength
wavelength_index = np.argmin(np.abs(wavelengths - specific_wavelength))
intensity_at_specific_wavelength = data[wavelength_index, :]
plt.figure(figsize=(8, 6), dpi=600)
for j in range(len(dsets2) - 1):
    plt.plot([m[j], m[j + 1]], [intensity_at_specific_wavelength[j], intensity_at_specific_wavelength[j + 1]],
             marker='o', linestyle='-', linewidth=1, label=f'Spectrum {j + 1}-{j + 2}')
plt.plot(m[-1], intensity_at_specific_wavelength[-1], marker='o', label=f'Spectrum {len(dsets2)}')
plt.xlabel('Time (seconds)', fontsize=14, fontweight="bold")
plt.ylabel(f'Intensity at {specific_wavelength} nm', fontsize=14, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12, direction="in")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend(loc="lower right", fontsize=8)
plt.title(f'Intensity Variation at {specific_wavelength} nm Over Time', fontsize=16, fontweight="bold")
plt.show()
folder_name = 'Ref_Spectrum_Data'
folder_path = os.path.join(DIRPATH, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
#csv_file_path1 = os.path.join(folder_path, "Ref_QE65000_Spectrometer.csv")
#pd.DataFrame(data_wl_wavelength).to_csv(csv_file_path1, header=None, index=None)

##############################################################################################################################################################

date = "2024-03-23T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/hera/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 8
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    print("\nKeys: %s" % f.keys())
    keys = list(f.keys())
key = keys[16]
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    g = f[key]
    indices = sorted([d.replace("spectrum_", "") for d in g.keys()])
    dsets = ["{0}".format(n) for n in indices]
    print("attributes for test:", g[dsets[0]].attrs.keys())
    print(dsets)
    sp = 0
    h = g[dsets[sp]]
    print("\nh:", h)
    indices2 = sorted(h.keys())
    dsets1 = ["{0}".format(n) for n in indices2]
    dsets2 = sorted(dsets1, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    sp2 = 0 
    print("\nspectra:", dsets2)
    laser_wavelength = 650  
    wavelengths = np.array(h[dsets2[sp2]].attrs["wavelengths"])
    start_times = np.array(h[dsets2[sp2]].attrs["creation_timestamp"]) 
    integration_time = np.array(h[dsets2[sp2]].attrs["integration_time"]) 
    creation_time = np.array(h[dsets2[sp2]].attrs["creation_timestamp"])  
    time_interval = np.array(h[dsets2[sp2]].attrs["time_interval"])
    information = np.array(h[dsets2[sp2]].attrs["information"])
    data = np.zeros((len(h[dsets2[sp2]]), len(dsets2)), dtype=float)
    creation_times = np.array((len(h[dsets2[sp2]].attrs["creation_timestamp"]), len(dsets2)), dtype=str)
    m = np.array([], dtype=np.float32)
    for j in range(len(dsets2)):
        data[:, j] = np.array(h[dsets2[j]]) - np.array(h[dsets2[j]].attrs["background"])
        starttime = h[dsets2[sp2]].attrs["creation_timestamp"]
        time = h[dsets2[j]].attrs["creation_timestamp"]
        strstarttime = starttime
        strtime = time
        newstarttime = strstarttime.split("T")[-1]
        newtime = strtime.split("T")[-1]
        newstarttime = newstarttime.replace(date, "")
        newtime = newtime.replace(date, "")
        start_time_s = pd.Timestamp(newstarttime).timestamp()
        time_s = pd.Timestamp(newtime).timestamp()
        delta_time = time_s - start_time_s
        m = np.append(m, delta_time)
t = np.reshape(m, (1, len(dsets2)))
data_average = np.sum(data, 1) / len(dsets2)
data_wl_wavelength_2 = np.zeros((len(data_average), 2), dtype=float)
data_wl_wavelength_2[:, 0] = wavelengths
data_wl_wavelength_2[:, 1] = data_average
plot_spectrum(data_wl_wavelength_2[:, 0], data_wl_wavelength_2[:, 1],
              x_label='Wavelength (nm)', y_label='Intensity (a.u.)',
              title='Average Spectrum',
              legend_label='Intensity at Wavelength')
specific_wavelength = laser_wavelength
wavelength_index = np.argmin(np.abs(wavelengths - specific_wavelength))
intensity_at_specific_wavelength = data[wavelength_index, :]
plt.figure(figsize=(8, 6), dpi=600)
for j in range(len(dsets2) - 1):
    plt.plot([m[j], m[j + 1]], [intensity_at_specific_wavelength[j], intensity_at_specific_wavelength[j + 1]],
             marker='o', linestyle='-', linewidth=1, label=f'Spectrum {j + 1}-{j + 2}')
plt.plot(m[-1], intensity_at_specific_wavelength[-1], marker='o', label=f'Spectrum {len(dsets2)}')
plt.xlabel('Time (seconds)', fontsize=14, fontweight="bold")
plt.ylabel(f'Intensity at {specific_wavelength} nm', fontsize=14, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12, direction="in")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend(loc="lower right", fontsize=8)
plt.title(f'Intensity Variation at {specific_wavelength} nm Over Time', fontsize=16, fontweight="bold")
plt.show()
folder_name = 'Ref_Spectrum_Data'
folder_path = os.path.join(DIRPATH, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
#csv_file_path2 = os.path.join(folder_path, "Ref_USB2000_Spectrometer.csv")
#pd.DataFrame(data_wl_wavelength).to_csv(csv_file_path2, header=None, index=None)

##############################################################################################################################################################

df_normalized = pd.DataFrame({
    'Wavelength': data_wl_wavelength_1[:, 0],
    'Intensity_1': data_wl_wavelength_1[:, 1],
    'Intensity_2': data_wl_wavelength_2[:, 1],
    'Normalized_Intensity': abs(data_wl_wavelength_2[:, 1] / data_wl_wavelength_1[:, 1]) 
})
plot_spectrum(df_normalized['Wavelength'], df_normalized['Normalized_Intensity'],
              x_label='Wavelength (nm)', y_label='Normalized Intensity',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')
csv_file_path_normalized = os.path.join(folder_path, "Normalized_Intensity.csv")
df_normalized.to_csv(csv_file_path_normalized, index=None)

##############################################################################################################################################################

input_wavelength = float(input("Enter the wavelength: "))
power_percentage = float(input(f"Enter the power percentage at {input_wavelength} nm: "))
closest_wavelength_index = np.argmin(np.abs(df_normalized['Wavelength'] - input_wavelength))
closest_wavelength = df_normalized.loc[closest_wavelength_index, 'Wavelength']
scaled_factor = power_percentage / df_normalized.loc[closest_wavelength_index, 'Normalized_Intensity']
df_normalized['Percentage_Transmission_8'] = df_normalized['Normalized_Intensity'] * scaled_factor
df_filtered = df_normalized[(df_normalized['Wavelength'] > 400)] #& (df_normalized['Wavelength'] < 1000)]
plot_spectrum(df_filtered['Wavelength'], df_filtered['Percentage_Transmission_8'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')
#csv_file_path_percentage_transmission = os.path.join(folder_path, "Percentage_Transmission_2.csv")
#df_normalized[['Wavelength', 'Percentage_Transmission']].to_csv(csv_file_path_percentage_transmission, index=None)


##############################################################################################################################################################

user_input = input("Press Enter to continue or type 'exit' to quit: ")
if user_input.lower() == 'exit':
    sys.exit()

##############################################################################################################################################################

date = "2024-03-20T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/hera/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 5
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    print("\nKeys: %s" % f.keys())
    keys = list(f.keys())
key = keys[0]
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    g = f[key]
    indices = sorted([d.replace("spectrum_", "") for d in g.keys()])
    dsets = ["{0}".format(n) for n in indices]
    print("attributes for test:", g[dsets[0]].attrs.keys())
    print(dsets)
    sp = 0
    h = g[dsets[sp]]
    print("\nh:", h)
    indices2 = sorted(h.keys())
    dsets1 = ["{0}".format(n) for n in indices2]
    dsets2 = sorted(dsets1, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    sp2 = 0 
    print("\nspectra:", dsets2)
    laser_wavelength = 650  
    wavelengths = np.array(h[dsets2[sp2]].attrs["wavelengths"])
    start_times = np.array(h[dsets2[sp2]].attrs["creation_timestamp"]) 
    integration_time = np.array(h[dsets2[sp2]].attrs["integration_time"]) 
    creation_time = np.array(h[dsets2[sp2]].attrs["creation_timestamp"])  
    time_interval = np.array(h[dsets2[sp2]].attrs["time_interval"])
    information = np.array(h[dsets2[sp2]].attrs["information"])
    data = np.zeros((len(h[dsets2[sp2]]), len(dsets2)), dtype=float)
    creation_times = np.array((len(h[dsets2[sp2]].attrs["creation_timestamp"]), len(dsets2)), dtype=str)
    m = np.array([], dtype=np.float32)
    for j in range(len(dsets2)):
        data[:, j] = np.array(h[dsets2[j]]) - np.array(h[dsets2[j]].attrs["background"])
        starttime = h[dsets2[sp2]].attrs["creation_timestamp"]
        time = h[dsets2[j]].attrs["creation_timestamp"]
        strstarttime = starttime
        strtime = time
        newstarttime = strstarttime.split("T")[-1]
        newtime = strtime.split("T")[-1]
        newstarttime = newstarttime.replace(date, "")
        newtime = newtime.replace(date, "")
        start_time_s = pd.Timestamp(newstarttime).timestamp()
        time_s = pd.Timestamp(newtime).timestamp()
        delta_time = time_s - start_time_s
        m = np.append(m, delta_time)
t = np.reshape(m, (1, len(dsets2)))
data_average = np.sum(data, 1) / len(dsets2)
data_wl_wavelength_1 = np.zeros((len(data_average), 2), dtype=float)
data_wl_wavelength_1[:, 0] = wavelengths
data_wl_wavelength_1[:, 1] = data_average
plot_spectrum(data_wl_wavelength_1[:, 0], data_wl_wavelength_1[:, 1],
              x_label='Wavelength (nm)', y_label='Intensity (a.u.)',
              title='Average Spectrum',
              legend_label='Intensity at Wavelength')
specific_wavelength = laser_wavelength
wavelength_index = np.argmin(np.abs(wavelengths - specific_wavelength))
intensity_at_specific_wavelength = data[wavelength_index, :]
plt.figure(figsize=(8, 6), dpi=600)
for j in range(len(dsets2) - 1):
    plt.plot([m[j], m[j + 1]], [intensity_at_specific_wavelength[j], intensity_at_specific_wavelength[j + 1]],
             marker='o', linestyle='-', linewidth=1, label=f'Spectrum {j + 1}-{j + 2}')
plt.plot(m[-1], intensity_at_specific_wavelength[-1], marker='o', label=f'Spectrum {len(dsets2)}')
plt.xlabel('Time (seconds)', fontsize=14, fontweight="bold")
plt.ylabel(f'Intensity at {specific_wavelength} nm', fontsize=14, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12, direction="in")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend(loc="lower right", fontsize=8)
plt.title(f'Intensity Variation at {specific_wavelength} nm Over Time', fontsize=16, fontweight="bold")
plt.show()
folder_name = 'Ref_Spectrum_Data'
folder_path = os.path.join(DIRPATH, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
#csv_file_path1 = os.path.join(folder_path, "Ref_QE65000_Spectrometer.csv")
#pd.DataFrame(data_wl_wavelength).to_csv(csv_file_path1, header=None, index=None)

##############################################################################################################################################################

date = "2024-03-23T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/hera/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 8
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    print("\nKeys: %s" % f.keys())
    keys = list(f.keys())
key = keys[16]
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    g = f[key]
    indices = sorted([d.replace("spectrum_", "") for d in g.keys()])
    dsets = ["{0}".format(n) for n in indices]
    print("attributes for test:", g[dsets[0]].attrs.keys())
    print(dsets)
    sp = 0
    h = g[dsets[sp]]
    print("\nh:", h)
    indices2 = sorted(h.keys())
    dsets1 = ["{0}".format(n) for n in indices2]
    dsets2 = sorted(dsets1, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    sp2 = 0 
    print("\nspectra:", dsets2)
    laser_wavelength = 650  
    wavelengths = np.array(h[dsets2[sp2]].attrs["wavelengths"])
    start_times = np.array(h[dsets2[sp2]].attrs["creation_timestamp"]) 
    integration_time = np.array(h[dsets2[sp2]].attrs["integration_time"]) 
    creation_time = np.array(h[dsets2[sp2]].attrs["creation_timestamp"])  
    time_interval = np.array(h[dsets2[sp2]].attrs["time_interval"])
    information = np.array(h[dsets2[sp2]].attrs["information"])
    data = np.zeros((len(h[dsets2[sp2]]), len(dsets2)), dtype=float)
    creation_times = np.array((len(h[dsets2[sp2]].attrs["creation_timestamp"]), len(dsets2)), dtype=str)
    m = np.array([], dtype=np.float32)
    for j in range(len(dsets2)):
        data[:, j] = np.array(h[dsets2[j]]) - np.array(h[dsets2[j]].attrs["background"])
        starttime = h[dsets2[sp2]].attrs["creation_timestamp"]
        time = h[dsets2[j]].attrs["creation_timestamp"]
        strstarttime = starttime
        strtime = time
        newstarttime = strstarttime.split("T")[-1]
        newtime = strtime.split("T")[-1]
        newstarttime = newstarttime.replace(date, "")
        newtime = newtime.replace(date, "")
        start_time_s = pd.Timestamp(newstarttime).timestamp()
        time_s = pd.Timestamp(newtime).timestamp()
        delta_time = time_s - start_time_s
        m = np.append(m, delta_time)
t = np.reshape(m, (1, len(dsets2)))
data_average = np.sum(data, 1) / len(dsets2)
data_wl_wavelength_2 = np.zeros((len(data_average), 2), dtype=float)
data_wl_wavelength_2[:, 0] = wavelengths
data_wl_wavelength_2[:, 1] = data_average
plot_spectrum(data_wl_wavelength_2[:, 0], data_wl_wavelength_2[:, 1],
              x_label='Wavelength (nm)', y_label='Intensity (a.u.)',
              title='Average Spectrum',
              legend_label='Intensity at Wavelength')
specific_wavelength = laser_wavelength
wavelength_index = np.argmin(np.abs(wavelengths - specific_wavelength))
intensity_at_specific_wavelength = data[wavelength_index, :]
plt.figure(figsize=(8, 6), dpi=600)
for j in range(len(dsets2) - 1):
    plt.plot([m[j], m[j + 1]], [intensity_at_specific_wavelength[j], intensity_at_specific_wavelength[j + 1]],
             marker='o', linestyle='-', linewidth=1, label=f'Spectrum {j + 1}-{j + 2}')
plt.plot(m[-1], intensity_at_specific_wavelength[-1], marker='o', label=f'Spectrum {len(dsets2)}')
plt.xlabel('Time (seconds)', fontsize=14, fontweight="bold")
plt.ylabel(f'Intensity at {specific_wavelength} nm', fontsize=14, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12, direction="in")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend(loc="lower right", fontsize=8)
plt.title(f'Intensity Variation at {specific_wavelength} nm Over Time', fontsize=16, fontweight="bold")
plt.show()
folder_name = 'Ref_Spectrum_Data'
folder_path = os.path.join(DIRPATH, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
#csv_file_path2 = os.path.join(folder_path, "Ref_USB2000_Spectrometer.csv")
#pd.DataFrame(data_wl_wavelength).to_csv(csv_file_path2, header=None, index=None)

##############################################################################################################################################################

df_normalized = pd.DataFrame({
    'Wavelength': data_wl_wavelength_1[:, 0],
    'Intensity_1': data_wl_wavelength_1[:, 1],
    'Intensity_2': data_wl_wavelength_2[:, 1],
    'Normalized_Intensity': abs(data_wl_wavelength_2[:, 1] / data_wl_wavelength_1[:, 1]) 
})
plot_spectrum(df_normalized['Wavelength'], df_normalized['Normalized_Intensity'],
              x_label='Wavelength (nm)', y_label='Normalized Intensity',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')
#csv_file_path_normalized = os.path.join(folder_path, "Normalized_Intensity.csv")
#df_normalized.to_csv(csv_file_path_normalized, index=None)

##############################################################################################################################################################

input_wavelength = float(input("Enter the wavelength: "))
power_percentage = float(input(f"Enter the power percentage at {input_wavelength} nm: "))
closest_wavelength_index = np.argmin(np.abs(df_normalized['Wavelength'] - input_wavelength))
closest_wavelength = df_normalized.loc[closest_wavelength_index, 'Wavelength']
scaled_factor = power_percentage / df_normalized.loc[closest_wavelength_index, 'Normalized_Intensity']
df_normalized['Percentage_Transmission_7'] = df_normalized['Normalized_Intensity'] * scaled_factor
df_filtered = df_normalized[(df_normalized['Wavelength'] > 400)] #& (df_normalized['Wavelength'] < 1000)]
plot_spectrum(df_filtered['Wavelength'], df_filtered['Percentage_Transmission_7'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')
#csv_file_path_percentage_transmission = os.path.join(folder_path, "Percentage_Transmission.csv")
#df_normalized[['Wavelength', 'Percentage_Transmission']].to_csv(csv_file_path_percentage_transmission, index=None)


##############################################################################################################################################################

user_input = input("Press Enter to continue or type 'exit' to quit: ")
if user_input.lower() == 'exit':
    sys.exit()

##############################################################################################################################################################

date = "2024-03-20T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/hera/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 5
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    print("\nKeys: %s" % f.keys())
    keys = list(f.keys())
key = keys[0]
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    g = f[key]
    indices = sorted([d.replace("spectrum_", "") for d in g.keys()])
    dsets = ["{0}".format(n) for n in indices]
    print("attributes for test:", g[dsets[0]].attrs.keys())
    print(dsets)
    sp = 0
    h = g[dsets[sp]]
    print("\nh:", h)
    indices2 = sorted(h.keys())
    dsets1 = ["{0}".format(n) for n in indices2]
    dsets2 = sorted(dsets1, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    sp2 = 0 
    print("\nspectra:", dsets2)
    laser_wavelength = 650  
    wavelengths = np.array(h[dsets2[sp2]].attrs["wavelengths"])
    start_times = np.array(h[dsets2[sp2]].attrs["creation_timestamp"]) 
    integration_time = np.array(h[dsets2[sp2]].attrs["integration_time"]) 
    creation_time = np.array(h[dsets2[sp2]].attrs["creation_timestamp"])  
    time_interval = np.array(h[dsets2[sp2]].attrs["time_interval"])
    information = np.array(h[dsets2[sp2]].attrs["information"])
    data = np.zeros((len(h[dsets2[sp2]]), len(dsets2)), dtype=float)
    creation_times = np.array((len(h[dsets2[sp2]].attrs["creation_timestamp"]), len(dsets2)), dtype=str)
    m = np.array([], dtype=np.float32)
    for j in range(len(dsets2)):
        data[:, j] = np.array(h[dsets2[j]]) - np.array(h[dsets2[j]].attrs["background"])
        starttime = h[dsets2[sp2]].attrs["creation_timestamp"]
        time = h[dsets2[j]].attrs["creation_timestamp"]
        strstarttime = starttime
        strtime = time
        newstarttime = strstarttime.split("T")[-1]
        newtime = strtime.split("T")[-1]
        newstarttime = newstarttime.replace(date, "")
        newtime = newtime.replace(date, "")
        start_time_s = pd.Timestamp(newstarttime).timestamp()
        time_s = pd.Timestamp(newtime).timestamp()
        delta_time = time_s - start_time_s
        m = np.append(m, delta_time)
t = np.reshape(m, (1, len(dsets2)))
data_average = np.sum(data, 1) / len(dsets2)
data_wl_wavelength_1 = np.zeros((len(data_average), 2), dtype=float)
data_wl_wavelength_1[:, 0] = wavelengths
data_wl_wavelength_1[:, 1] = data_average
plot_spectrum(data_wl_wavelength_1[:, 0], data_wl_wavelength_1[:, 1],
              x_label='Wavelength (nm)', y_label='Intensity (a.u.)',
              title='Average Spectrum',
              legend_label='Intensity at Wavelength')
specific_wavelength = laser_wavelength
wavelength_index = np.argmin(np.abs(wavelengths - specific_wavelength))
intensity_at_specific_wavelength = data[wavelength_index, :]
plt.figure(figsize=(8, 6), dpi=600)
for j in range(len(dsets2) - 1):
    plt.plot([m[j], m[j + 1]], [intensity_at_specific_wavelength[j], intensity_at_specific_wavelength[j + 1]],
             marker='o', linestyle='-', linewidth=1, label=f'Spectrum {j + 1}-{j + 2}')
plt.plot(m[-1], intensity_at_specific_wavelength[-1], marker='o', label=f'Spectrum {len(dsets2)}')
plt.xlabel('Time (seconds)', fontsize=14, fontweight="bold")
plt.ylabel(f'Intensity at {specific_wavelength} nm', fontsize=14, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12, direction="in")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend(loc="lower right", fontsize=8)
plt.title(f'Intensity Variation at {specific_wavelength} nm Over Time', fontsize=16, fontweight="bold")
plt.show()
folder_name = 'Ref_Spectrum_Data'
folder_path = os.path.join(DIRPATH, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
#csv_file_path1 = os.path.join(folder_path, "Ref_QE65000_Spectrometer.csv")
#pd.DataFrame(data_wl_wavelength).to_csv(csv_file_path1, header=None, index=None)

##############################################################################################################################################################

date = "2024-03-23T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/hera/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 8
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    print("\nKeys: %s" % f.keys())
    keys = list(f.keys())
key = keys[16]
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    g = f[key]
    indices = sorted([d.replace("spectrum_", "") for d in g.keys()])
    dsets = ["{0}".format(n) for n in indices]
    print("attributes for test:", g[dsets[0]].attrs.keys())
    print(dsets)
    sp = 0
    h = g[dsets[sp]]
    print("\nh:", h)
    indices2 = sorted(h.keys())
    dsets1 = ["{0}".format(n) for n in indices2]
    dsets2 = sorted(dsets1, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    sp2 = 0 
    print("\nspectra:", dsets2)
    laser_wavelength = 650  
    wavelengths = np.array(h[dsets2[sp2]].attrs["wavelengths"])
    start_times = np.array(h[dsets2[sp2]].attrs["creation_timestamp"]) 
    integration_time = np.array(h[dsets2[sp2]].attrs["integration_time"]) 
    creation_time = np.array(h[dsets2[sp2]].attrs["creation_timestamp"])  
    time_interval = np.array(h[dsets2[sp2]].attrs["time_interval"])
    information = np.array(h[dsets2[sp2]].attrs["information"])
    data = np.zeros((len(h[dsets2[sp2]]), len(dsets2)), dtype=float)
    creation_times = np.array((len(h[dsets2[sp2]].attrs["creation_timestamp"]), len(dsets2)), dtype=str)
    m = np.array([], dtype=np.float32)
    for j in range(len(dsets2)):
        data[:, j] = np.array(h[dsets2[j]]) - np.array(h[dsets2[j]].attrs["background"])
        starttime = h[dsets2[sp2]].attrs["creation_timestamp"]
        time = h[dsets2[j]].attrs["creation_timestamp"]
        strstarttime = starttime
        strtime = time
        newstarttime = strstarttime.split("T")[-1]
        newtime = strtime.split("T")[-1]
        newstarttime = newstarttime.replace(date, "")
        newtime = newtime.replace(date, "")
        start_time_s = pd.Timestamp(newstarttime).timestamp()
        time_s = pd.Timestamp(newtime).timestamp()
        delta_time = time_s - start_time_s
        m = np.append(m, delta_time)
t = np.reshape(m, (1, len(dsets2)))
data_average = np.sum(data, 1) / len(dsets2)
data_wl_wavelength_2 = np.zeros((len(data_average), 2), dtype=float)
data_wl_wavelength_2[:, 0] = wavelengths
data_wl_wavelength_2[:, 1] = data_average
plot_spectrum(data_wl_wavelength_2[:, 0], data_wl_wavelength_2[:, 1],
              x_label='Wavelength (nm)', y_label='Intensity (a.u.)',
              title='Average Spectrum',
              legend_label='Intensity at Wavelength')
specific_wavelength = laser_wavelength
wavelength_index = np.argmin(np.abs(wavelengths - specific_wavelength))
intensity_at_specific_wavelength = data[wavelength_index, :]
plt.figure(figsize=(8, 6), dpi=600)
for j in range(len(dsets2) - 1):
    plt.plot([m[j], m[j + 1]], [intensity_at_specific_wavelength[j], intensity_at_specific_wavelength[j + 1]],
             marker='o', linestyle='-', linewidth=1, label=f'Spectrum {j + 1}-{j + 2}')
plt.plot(m[-1], intensity_at_specific_wavelength[-1], marker='o', label=f'Spectrum {len(dsets2)}')
plt.xlabel('Time (seconds)', fontsize=14, fontweight="bold")
plt.ylabel(f'Intensity at {specific_wavelength} nm', fontsize=14, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12, direction="in")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend(loc="lower right", fontsize=8)
plt.title(f'Intensity Variation at {specific_wavelength} nm Over Time', fontsize=16, fontweight="bold")
plt.show()
folder_name = 'Ref_Spectrum_Data'
folder_path = os.path.join(DIRPATH, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
#csv_file_path2 = os.path.join(folder_path, "Ref_USB2000_Spectrometer.csv")
#pd.DataFrame(data_wl_wavelength).to_csv(csv_file_path2, header=None, index=None)

##############################################################################################################################################################

df_normalized = pd.DataFrame({
    'Wavelength': data_wl_wavelength_1[:, 0],
    'Intensity_1': data_wl_wavelength_1[:, 1],
    'Intensity_2': data_wl_wavelength_2[:, 1],
    'Normalized_Intensity': abs(data_wl_wavelength_2[:, 1] / data_wl_wavelength_1[:, 1]) 
})
plot_spectrum(df_normalized['Wavelength'], df_normalized['Normalized_Intensity'],
              x_label='Wavelength (nm)', y_label='Normalized Intensity',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')
csv_file_path_normalized = os.path.join(folder_path, "Normalized_Intensity.csv")
df_normalized.to_csv(csv_file_path_normalized, index=None)

##############################################################################################################################################################

input_wavelength = float(input("Enter the wavelength: "))
power_percentage = float(input(f"Enter the power percentage at {input_wavelength} nm: "))
closest_wavelength_index = np.argmin(np.abs(df_normalized['Wavelength'] - input_wavelength))
closest_wavelength = df_normalized.loc[closest_wavelength_index, 'Wavelength']
scaled_factor = power_percentage / df_normalized.loc[closest_wavelength_index, 'Normalized_Intensity']
df_normalized['Percentage_Transmission_8'] = df_normalized['Normalized_Intensity'] * scaled_factor
df_filtered = df_normalized[(df_normalized['Wavelength'] > 400)] #& (df_normalized['Wavelength'] < 1000)]
plot_spectrum(df_filtered['Wavelength'], df_filtered['Percentage_Transmission_8'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')
#csv_file_path_percentage_transmission = os.path.join(folder_path, "Percentage_Transmission_2.csv")
#df_normalized[['Wavelength', 'Percentage_Transmission']].to_csv(csv_file_path_percentage_transmission, index=None)


##############################################################################################################################################################

user_input = input("Press Enter to continue or type 'exit' to quit: ")
if user_input.lower() == 'exit':
    sys.exit()

##############################################################################################################################################################

date = "2024-03-20T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/hera/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 5
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    print("\nKeys: %s" % f.keys())
    keys = list(f.keys())
key = keys[0]
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    g = f[key]
    indices = sorted([d.replace("spectrum_", "") for d in g.keys()])
    dsets = ["{0}".format(n) for n in indices]
    print("attributes for test:", g[dsets[0]].attrs.keys())
    print(dsets)
    sp = 0
    h = g[dsets[sp]]
    print("\nh:", h)
    indices2 = sorted(h.keys())
    dsets1 = ["{0}".format(n) for n in indices2]
    dsets2 = sorted(dsets1, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    sp2 = 0 
    print("\nspectra:", dsets2)
    laser_wavelength = 650  
    wavelengths = np.array(h[dsets2[sp2]].attrs["wavelengths"])
    start_times = np.array(h[dsets2[sp2]].attrs["creation_timestamp"]) 
    integration_time = np.array(h[dsets2[sp2]].attrs["integration_time"]) 
    creation_time = np.array(h[dsets2[sp2]].attrs["creation_timestamp"])  
    time_interval = np.array(h[dsets2[sp2]].attrs["time_interval"])
    information = np.array(h[dsets2[sp2]].attrs["information"])
    data = np.zeros((len(h[dsets2[sp2]]), len(dsets2)), dtype=float)
    creation_times = np.array((len(h[dsets2[sp2]].attrs["creation_timestamp"]), len(dsets2)), dtype=str)
    m = np.array([], dtype=np.float32)
    for j in range(len(dsets2)):
        data[:, j] = np.array(h[dsets2[j]]) - np.array(h[dsets2[j]].attrs["background"])
        starttime = h[dsets2[sp2]].attrs["creation_timestamp"]
        time = h[dsets2[j]].attrs["creation_timestamp"]
        strstarttime = starttime
        strtime = time
        newstarttime = strstarttime.split("T")[-1]
        newtime = strtime.split("T")[-1]
        newstarttime = newstarttime.replace(date, "")
        newtime = newtime.replace(date, "")
        start_time_s = pd.Timestamp(newstarttime).timestamp()
        time_s = pd.Timestamp(newtime).timestamp()
        delta_time = time_s - start_time_s
        m = np.append(m, delta_time)
t = np.reshape(m, (1, len(dsets2)))
data_average = np.sum(data, 1) / len(dsets2)
data_wl_wavelength_1 = np.zeros((len(data_average), 2), dtype=float)
data_wl_wavelength_1[:, 0] = wavelengths
data_wl_wavelength_1[:, 1] = data_average
plot_spectrum(data_wl_wavelength_1[:, 0], data_wl_wavelength_1[:, 1],
              x_label='Wavelength (nm)', y_label='Intensity (a.u.)',
              title='Average Spectrum',
              legend_label='Intensity at Wavelength')
specific_wavelength = laser_wavelength
wavelength_index = np.argmin(np.abs(wavelengths - specific_wavelength))
intensity_at_specific_wavelength = data[wavelength_index, :]
plt.figure(figsize=(8, 6), dpi=600)
for j in range(len(dsets2) - 1):
    plt.plot([m[j], m[j + 1]], [intensity_at_specific_wavelength[j], intensity_at_specific_wavelength[j + 1]],
             marker='o', linestyle='-', linewidth=1, label=f'Spectrum {j + 1}-{j + 2}')
plt.plot(m[-1], intensity_at_specific_wavelength[-1], marker='o', label=f'Spectrum {len(dsets2)}')
plt.xlabel('Time (seconds)', fontsize=14, fontweight="bold")
plt.ylabel(f'Intensity at {specific_wavelength} nm', fontsize=14, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12, direction="in")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend(loc="lower right", fontsize=8)
plt.title(f'Intensity Variation at {specific_wavelength} nm Over Time', fontsize=16, fontweight="bold")
plt.show()
folder_name = 'Ref_Spectrum_Data'
folder_path = os.path.join(DIRPATH, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
#csv_file_path1 = os.path.join(folder_path, "Ref_QE65000_Spectrometer.csv")
#pd.DataFrame(data_wl_wavelength).to_csv(csv_file_path1, header=None, index=None)

##############################################################################################################################################################

date = "2024-03-23T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/hera/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 8
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    print("\nKeys: %s" % f.keys())
    keys = list(f.keys())
key = keys[16]
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    g = f[key]
    indices = sorted([d.replace("spectrum_", "") for d in g.keys()])
    dsets = ["{0}".format(n) for n in indices]
    print("attributes for test:", g[dsets[0]].attrs.keys())
    print(dsets)
    sp = 0
    h = g[dsets[sp]]
    print("\nh:", h)
    indices2 = sorted(h.keys())
    dsets1 = ["{0}".format(n) for n in indices2]
    dsets2 = sorted(dsets1, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    sp2 = 0 
    print("\nspectra:", dsets2)
    laser_wavelength = 650  
    wavelengths = np.array(h[dsets2[sp2]].attrs["wavelengths"])
    start_times = np.array(h[dsets2[sp2]].attrs["creation_timestamp"]) 
    integration_time = np.array(h[dsets2[sp2]].attrs["integration_time"]) 
    creation_time = np.array(h[dsets2[sp2]].attrs["creation_timestamp"])  
    time_interval = np.array(h[dsets2[sp2]].attrs["time_interval"])
    information = np.array(h[dsets2[sp2]].attrs["information"])
    data = np.zeros((len(h[dsets2[sp2]]), len(dsets2)), dtype=float)
    creation_times = np.array((len(h[dsets2[sp2]].attrs["creation_timestamp"]), len(dsets2)), dtype=str)
    m = np.array([], dtype=np.float32)
    for j in range(len(dsets2)):
        data[:, j] = np.array(h[dsets2[j]]) - np.array(h[dsets2[j]].attrs["background"])
        starttime = h[dsets2[sp2]].attrs["creation_timestamp"]
        time = h[dsets2[j]].attrs["creation_timestamp"]
        strstarttime = starttime
        strtime = time
        newstarttime = strstarttime.split("T")[-1]
        newtime = strtime.split("T")[-1]
        newstarttime = newstarttime.replace(date, "")
        newtime = newtime.replace(date, "")
        start_time_s = pd.Timestamp(newstarttime).timestamp()
        time_s = pd.Timestamp(newtime).timestamp()
        delta_time = time_s - start_time_s
        m = np.append(m, delta_time)
t = np.reshape(m, (1, len(dsets2)))
data_average = np.sum(data, 1) / len(dsets2)
data_wl_wavelength_2 = np.zeros((len(data_average), 2), dtype=float)
data_wl_wavelength_2[:, 0] = wavelengths
data_wl_wavelength_2[:, 1] = data_average
plot_spectrum(data_wl_wavelength_2[:, 0], data_wl_wavelength_2[:, 1],
              x_label='Wavelength (nm)', y_label='Intensity (a.u.)',
              title='Average Spectrum',
              legend_label='Intensity at Wavelength')
specific_wavelength = laser_wavelength
wavelength_index = np.argmin(np.abs(wavelengths - specific_wavelength))
intensity_at_specific_wavelength = data[wavelength_index, :]
plt.figure(figsize=(8, 6), dpi=600)
for j in range(len(dsets2) - 1):
    plt.plot([m[j], m[j + 1]], [intensity_at_specific_wavelength[j], intensity_at_specific_wavelength[j + 1]],
             marker='o', linestyle='-', linewidth=1, label=f'Spectrum {j + 1}-{j + 2}')
plt.plot(m[-1], intensity_at_specific_wavelength[-1], marker='o', label=f'Spectrum {len(dsets2)}')
plt.xlabel('Time (seconds)', fontsize=14, fontweight="bold")
plt.ylabel(f'Intensity at {specific_wavelength} nm', fontsize=14, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12, direction="in")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend(loc="lower right", fontsize=8)
plt.title(f'Intensity Variation at {specific_wavelength} nm Over Time', fontsize=16, fontweight="bold")
plt.show()
folder_name = 'Ref_Spectrum_Data'
folder_path = os.path.join(DIRPATH, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
#csv_file_path2 = os.path.join(folder_path, "Ref_USB2000_Spectrometer.csv")
#pd.DataFrame(data_wl_wavelength).to_csv(csv_file_path2, header=None, index=None)

##############################################################################################################################################################

df_normalized = pd.DataFrame({
    'Wavelength': data_wl_wavelength_1[:, 0],
    'Intensity_1': data_wl_wavelength_1[:, 1],
    'Intensity_2': data_wl_wavelength_2[:, 1],
    'Normalized_Intensity': abs(data_wl_wavelength_2[:, 1] / data_wl_wavelength_1[:, 1]) 
})
plot_spectrum(df_normalized['Wavelength'], df_normalized['Normalized_Intensity'],
              x_label='Wavelength (nm)', y_label='Normalized Intensity',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')
#csv_file_path_normalized = os.path.join(folder_path, "Normalized_Intensity.csv")
#df_normalized.to_csv(csv_file_path_normalized, index=None)

##############################################################################################################################################################

input_wavelength = float(input("Enter the wavelength: "))
power_percentage = float(input(f"Enter the power percentage at {input_wavelength} nm: "))
closest_wavelength_index = np.argmin(np.abs(df_normalized['Wavelength'] - input_wavelength))
closest_wavelength = df_normalized.loc[closest_wavelength_index, 'Wavelength']
scaled_factor = power_percentage / df_normalized.loc[closest_wavelength_index, 'Normalized_Intensity']
df_normalized['Percentage_Transmission_9'] = df_normalized['Normalized_Intensity'] * scaled_factor
df_filtered = df_normalized[(df_normalized['Wavelength'] > 400)] #& (df_normalized['Wavelength'] < 1000)]
plot_spectrum(df_filtered['Wavelength'], df_filtered['Percentage_Transmission_9'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')
#csv_file_path_percentage_transmission = os.path.join(folder_path, "Percentage_Transmission.csv")
#df_normalized[['Wavelength', 'Percentage_Transmission']].to_csv(csv_file_path_percentage_transmission, index=None)


##############################################################################################################################################################

user_input = input("Press Enter to continue or type 'exit' to quit: ")
if user_input.lower() == 'exit':
    sys.exit()

##############################################################################################################################################################

date = "2024-03-20T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/hera/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 5
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    print("\nKeys: %s" % f.keys())
    keys = list(f.keys())
key = keys[0]
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    g = f[key]
    indices = sorted([d.replace("spectrum_", "") for d in g.keys()])
    dsets = ["{0}".format(n) for n in indices]
    print("attributes for test:", g[dsets[0]].attrs.keys())
    print(dsets)
    sp = 0
    h = g[dsets[sp]]
    print("\nh:", h)
    indices2 = sorted(h.keys())
    dsets1 = ["{0}".format(n) for n in indices2]
    dsets2 = sorted(dsets1, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    sp2 = 0 
    print("\nspectra:", dsets2)
    laser_wavelength = 650  
    wavelengths = np.array(h[dsets2[sp2]].attrs["wavelengths"])
    start_times = np.array(h[dsets2[sp2]].attrs["creation_timestamp"]) 
    integration_time = np.array(h[dsets2[sp2]].attrs["integration_time"]) 
    creation_time = np.array(h[dsets2[sp2]].attrs["creation_timestamp"])  
    time_interval = np.array(h[dsets2[sp2]].attrs["time_interval"])
    information = np.array(h[dsets2[sp2]].attrs["information"])
    data = np.zeros((len(h[dsets2[sp2]]), len(dsets2)), dtype=float)
    creation_times = np.array((len(h[dsets2[sp2]].attrs["creation_timestamp"]), len(dsets2)), dtype=str)
    m = np.array([], dtype=np.float32)
    for j in range(len(dsets2)):
        data[:, j] = np.array(h[dsets2[j]]) - np.array(h[dsets2[j]].attrs["background"])
        starttime = h[dsets2[sp2]].attrs["creation_timestamp"]
        time = h[dsets2[j]].attrs["creation_timestamp"]
        strstarttime = starttime
        strtime = time
        newstarttime = strstarttime.split("T")[-1]
        newtime = strtime.split("T")[-1]
        newstarttime = newstarttime.replace(date, "")
        newtime = newtime.replace(date, "")
        start_time_s = pd.Timestamp(newstarttime).timestamp()
        time_s = pd.Timestamp(newtime).timestamp()
        delta_time = time_s - start_time_s
        m = np.append(m, delta_time)
t = np.reshape(m, (1, len(dsets2)))
data_average = np.sum(data, 1) / len(dsets2)
data_wl_wavelength_1 = np.zeros((len(data_average), 2), dtype=float)
data_wl_wavelength_1[:, 0] = wavelengths
data_wl_wavelength_1[:, 1] = data_average
plot_spectrum(data_wl_wavelength_1[:, 0], data_wl_wavelength_1[:, 1],
              x_label='Wavelength (nm)', y_label='Intensity (a.u.)',
              title='Average Spectrum',
              legend_label='Intensity at Wavelength')
specific_wavelength = laser_wavelength
wavelength_index = np.argmin(np.abs(wavelengths - specific_wavelength))
intensity_at_specific_wavelength = data[wavelength_index, :]
plt.figure(figsize=(8, 6), dpi=600)
for j in range(len(dsets2) - 1):
    plt.plot([m[j], m[j + 1]], [intensity_at_specific_wavelength[j], intensity_at_specific_wavelength[j + 1]],
             marker='o', linestyle='-', linewidth=1, label=f'Spectrum {j + 1}-{j + 2}')
plt.plot(m[-1], intensity_at_specific_wavelength[-1], marker='o', label=f'Spectrum {len(dsets2)}')
plt.xlabel('Time (seconds)', fontsize=14, fontweight="bold")
plt.ylabel(f'Intensity at {specific_wavelength} nm', fontsize=14, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12, direction="in")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend(loc="lower right", fontsize=8)
plt.title(f'Intensity Variation at {specific_wavelength} nm Over Time', fontsize=16, fontweight="bold")
plt.show()
folder_name = 'Ref_Spectrum_Data'
folder_path = os.path.join(DIRPATH, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
#csv_file_path1 = os.path.join(folder_path, "Ref_QE65000_Spectrometer.csv")
#pd.DataFrame(data_wl_wavelength).to_csv(csv_file_path1, header=None, index=None)

##############################################################################################################################################################

date = "2024-03-23T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/hera/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 8
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    print("\nKeys: %s" % f.keys())
    keys = list(f.keys())
key = keys[16]
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    g = f[key]
    indices = sorted([d.replace("spectrum_", "") for d in g.keys()])
    dsets = ["{0}".format(n) for n in indices]
    print("attributes for test:", g[dsets[0]].attrs.keys())
    print(dsets)
    sp = 0
    h = g[dsets[sp]]
    print("\nh:", h)
    indices2 = sorted(h.keys())
    dsets1 = ["{0}".format(n) for n in indices2]
    dsets2 = sorted(dsets1, key=lambda x: int("".join([i for i in x if i.isdigit()])))
    sp2 = 0 
    print("\nspectra:", dsets2)
    laser_wavelength = 650  
    wavelengths = np.array(h[dsets2[sp2]].attrs["wavelengths"])
    start_times = np.array(h[dsets2[sp2]].attrs["creation_timestamp"]) 
    integration_time = np.array(h[dsets2[sp2]].attrs["integration_time"]) 
    creation_time = np.array(h[dsets2[sp2]].attrs["creation_timestamp"])  
    time_interval = np.array(h[dsets2[sp2]].attrs["time_interval"])
    information = np.array(h[dsets2[sp2]].attrs["information"])
    data = np.zeros((len(h[dsets2[sp2]]), len(dsets2)), dtype=float)
    creation_times = np.array((len(h[dsets2[sp2]].attrs["creation_timestamp"]), len(dsets2)), dtype=str)
    m = np.array([], dtype=np.float32)
    for j in range(len(dsets2)):
        data[:, j] = np.array(h[dsets2[j]]) - np.array(h[dsets2[j]].attrs["background"])
        starttime = h[dsets2[sp2]].attrs["creation_timestamp"]
        time = h[dsets2[j]].attrs["creation_timestamp"]
        strstarttime = starttime
        strtime = time
        newstarttime = strstarttime.split("T")[-1]
        newtime = strtime.split("T")[-1]
        newstarttime = newstarttime.replace(date, "")
        newtime = newtime.replace(date, "")
        start_time_s = pd.Timestamp(newstarttime).timestamp()
        time_s = pd.Timestamp(newtime).timestamp()
        delta_time = time_s - start_time_s
        m = np.append(m, delta_time)
t = np.reshape(m, (1, len(dsets2)))
data_average = np.sum(data, 1) / len(dsets2)
data_wl_wavelength_2 = np.zeros((len(data_average), 2), dtype=float)
data_wl_wavelength_2[:, 0] = wavelengths
data_wl_wavelength_2[:, 1] = data_average
plot_spectrum(data_wl_wavelength_2[:, 0], data_wl_wavelength_2[:, 1],
              x_label='Wavelength (nm)', y_label='Intensity (a.u.)',
              title='Average Spectrum',
              legend_label='Intensity at Wavelength')
specific_wavelength = laser_wavelength
wavelength_index = np.argmin(np.abs(wavelengths - specific_wavelength))
intensity_at_specific_wavelength = data[wavelength_index, :]
plt.figure(figsize=(8, 6), dpi=600)
for j in range(len(dsets2) - 1):
    plt.plot([m[j], m[j + 1]], [intensity_at_specific_wavelength[j], intensity_at_specific_wavelength[j + 1]],
             marker='o', linestyle='-', linewidth=1, label=f'Spectrum {j + 1}-{j + 2}')
plt.plot(m[-1], intensity_at_specific_wavelength[-1], marker='o', label=f'Spectrum {len(dsets2)}')
plt.xlabel('Time (seconds)', fontsize=14, fontweight="bold")
plt.ylabel(f'Intensity at {specific_wavelength} nm', fontsize=14, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12, direction="in")
plt.grid(color="gray", linestyle="--", linewidth=0.5)
plt.legend(loc="lower right", fontsize=8)
plt.title(f'Intensity Variation at {specific_wavelength} nm Over Time', fontsize=16, fontweight="bold")
plt.show()
folder_name = 'Ref_Spectrum_Data'
folder_path = os.path.join(DIRPATH, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
#csv_file_path2 = os.path.join(folder_path, "Ref_USB2000_Spectrometer.csv")
#pd.DataFrame(data_wl_wavelength).to_csv(csv_file_path2, header=None, index=None)

##############################################################################################################################################################

df_normalized = pd.DataFrame({
    'Wavelength': data_wl_wavelength_1[:, 0],
    'Intensity_1': data_wl_wavelength_1[:, 1],
    'Intensity_2': data_wl_wavelength_2[:, 1],
    'Normalized_Intensity': abs(data_wl_wavelength_2[:, 1] / data_wl_wavelength_1[:, 1]) 
})
plot_spectrum(df_normalized['Wavelength'], df_normalized['Normalized_Intensity'],
              x_label='Wavelength (nm)', y_label='Normalized Intensity',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')
csv_file_path_normalized = os.path.join(folder_path, "Normalized_Intensity.csv")
df_normalized.to_csv(csv_file_path_normalized, index=None)

##############################################################################################################################################################

input_wavelength = float(input("Enter the wavelength: "))
power_percentage = float(input(f"Enter the power percentage at {input_wavelength} nm: "))
closest_wavelength_index = np.argmin(np.abs(df_normalized['Wavelength'] - input_wavelength))
closest_wavelength = df_normalized.loc[closest_wavelength_index, 'Wavelength']
scaled_factor = power_percentage / df_normalized.loc[closest_wavelength_index, 'Normalized_Intensity']
df_normalized['Percentage_Transmission_10'] = df_normalized['Normalized_Intensity'] * scaled_factor
df_filtered = df_normalized[(df_normalized['Wavelength'] > 400)] #& (df_normalized['Wavelength'] < 1000)]
plot_spectrum(df_filtered['Wavelength'], df_filtered['Percentage_Transmission_10'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')
#csv_file_path_percentage_transmission = os.path.join(folder_path, "Percentage_Transmission_2.csv")
#df_normalized[['Wavelength', 'Percentage_Transmission']].to_csv(csv_file_path_percentage_transmission, index=None)

##############################################################################################################################################################

df_combined = pd.DataFrame({
    'Wavelength': df_normalized['Wavelength'],
    'Percentage_Transmission_1': df_normalized['Percentage_Transmission_1'],
    'Percentage_Transmission_2': df_normalized['Percentage_Transmission_2'],
    'Percentage_Transmission_3': df_normalized['Percentage_Transmission_3'],
    'Percentage_Transmission_4': df_normalized['Percentage_Transmission_4'],
    'Percentage_Transmission_5': df_normalized['Percentage_Transmission_5'],
    'Percentage_Transmission_6': df_normalized['Percentage_Transmission_6'],
    'Percentage_Transmission_7': df_normalized['Percentage_Transmission_7'],
    'Percentage_Transmission_8': df_normalized['Percentage_Transmission_8'],
    'Percentage_Transmission_9': df_normalized['Percentage_Transmission_9'],
    'Percentage_Transmission_10': df_normalized['Percentage_Transmission_10']
})

plot_spectrum(df_combined['Wavelength'], df_combined['Percentage_Transmission_1'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Combined Percentage Transmission vs Wavelength',
              legend_label='Percentage Transmission - Spectrum 1')

plot_spectrum(df_combined['Wavelength'], df_combined['Percentage_Transmission_2'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Combined Percentage Transmission vs Wavelength',
              legend_label='Percentage Transmission - Spectrum 2')

plot_spectrum(df_combined['Wavelength'], df_combined['Percentage_Transmission_3'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Combined Percentage Transmission vs Wavelength',
              legend_label='Percentage Transmission - Spectrum 3')

plot_spectrum(df_combined['Wavelength'], df_combined['Percentage_Transmission_4'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Combined Percentage Transmission vs Wavelength',
              legend_label='Percentage Transmission - Spectrum 4')

plot_spectrum(df_combined['Wavelength'], df_combined['Percentage_Transmission_5'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Combined Percentage Transmission vs Wavelength',
              legend_label='Percentage Transmission - Spectrum 5')

plot_spectrum(df_combined['Wavelength'], df_combined['Percentage_Transmission_6'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Combined Percentage Transmission vs Wavelength',
              legend_label='Percentage Transmission - Spectrum 6')

plot_spectrum(df_combined['Wavelength'], df_combined['Percentage_Transmission_7'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Combined Percentage Transmission vs Wavelength',
              legend_label='Percentage Transmission - Spectrum 7')

plot_spectrum(df_combined['Wavelength'], df_combined['Percentage_Transmission_8'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Combined Percentage Transmission vs Wavelength',
              legend_label='Percentage Transmission - Spectrum 8')

plot_spectrum(df_combined['Wavelength'], df_combined['Percentage_Transmission_9'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Combined Percentage Transmission vs Wavelength',
              legend_label='Percentage Transmission - Spectrum 9')

plot_spectrum(df_combined['Wavelength'], df_combined['Percentage_Transmission_10'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Combined Percentage Transmission vs Wavelength',
              legend_label='Percentage Transmission - Spectrum 10')

##############################################################################################################################################################
