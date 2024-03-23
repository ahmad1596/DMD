##############################################################################################################################################################

import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import pandas as pd

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

def plot_spectrum_combined(x_values, intensities_1, intensities_2, intensities_3, intensities_4, intensities_5, x_label, y_label, title, legend_label_1=None, legend_label_2=None, legend_label_3=None, legend_label_4=None, legend_label_5=None):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    ax.plot(x_values, intensities_1, label=legend_label_1)
    ax.plot(x_values, intensities_2, label=legend_label_2)
    ax.plot(x_values, intensities_3, label=legend_label_3)
    ax.plot(x_values, intensities_4, label=legend_label_4)
    ax.plot(x_values, intensities_5, label=legend_label_5)
    ax.set_xlabel(x_label, fontsize=14, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=16, fontweight="bold")
    plt.show()
    
##############################################################################################################################################################
# SPECTRUM WITHOUT FIBER
##############################################################################################################################################################

date = "2024-03-20T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/DELL/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
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

##############################################################################################################################################################
# SPECTRUM WITH FIBER 1
##############################################################################################################################################################

date = "2024-03-23T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/DELL/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
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

##############################################################################################################################################################

df_normalized = pd.DataFrame({
    'Wavelength': data_wl_wavelength_1[:, 0],
    'Intensity_1': data_wl_wavelength_1[:, 1],
    'Intensity_2': data_wl_wavelength_2[:, 1],
    'Normalized_Intensity_1': abs(data_wl_wavelength_2[:, 1] / data_wl_wavelength_1[:, 1]) 
})

input_wavelength = float(input("Enter the wavelength: "))
power_percentage = float(input(f"Enter the power percentage at {input_wavelength} nm: "))
closest_wavelength_index = np.argmin(np.abs(df_normalized['Wavelength'] - input_wavelength))
closest_wavelength = df_normalized.loc[closest_wavelength_index, 'Wavelength']
scaled_factor = power_percentage / df_normalized.loc[closest_wavelength_index, 'Normalized_Intensity_1']
df_normalized['Percentage_Transmission_1'] = df_normalized['Normalized_Intensity_1'] * scaled_factor
df_filtered_1 = df_normalized[(df_normalized['Wavelength'] > 400)] #& (df_normalized['Wavelength'] < 1000)]
plot_spectrum(df_filtered_1['Wavelength'], df_filtered_1['Percentage_Transmission_1'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')

##############################################################################################################################################################
# SPECTRUM WITH FIBER 2
##############################################################################################################################################################

date = "2024-03-20T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/DELL/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
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

##############################################################################################################################################################

date = "2024-03-23T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/DELL/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
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

##############################################################################################################################################################

df_normalized = pd.DataFrame({
    'Wavelength': data_wl_wavelength_1[:, 0],
    'Intensity_1': data_wl_wavelength_1[:, 1],
    'Intensity_2': data_wl_wavelength_2[:, 1],
    'Normalized_Intensity_2': abs(data_wl_wavelength_2[:, 1] / data_wl_wavelength_1[:, 1]) 
})

input_wavelength = float(input("Enter the wavelength: "))
power_percentage = float(input(f"Enter the power percentage at {input_wavelength} nm: "))
closest_wavelength_index = np.argmin(np.abs(df_normalized['Wavelength'] - input_wavelength))
closest_wavelength = df_normalized.loc[closest_wavelength_index, 'Wavelength']
scaled_factor = power_percentage / df_normalized.loc[closest_wavelength_index, 'Normalized_Intensity_2']
df_normalized['Percentage_Transmission_2'] = df_normalized['Normalized_Intensity_2'] * scaled_factor
df_filtered_2 = df_normalized[(df_normalized['Wavelength'] > 400)] #& (df_normalized['Wavelength'] < 1000)]
plot_spectrum(df_filtered_2['Wavelength'], df_filtered_2['Percentage_Transmission_2'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')

##############################################################################################################################################################
# SPECTRUM WITH FIBER 3
##############################################################################################################################################################

date = "2024-03-20T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/DELL/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
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

##############################################################################################################################################################

date = "2024-03-23T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/DELL/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
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

##############################################################################################################################################################

df_normalized = pd.DataFrame({
    'Wavelength': data_wl_wavelength_1[:, 0],
    'Intensity_1': data_wl_wavelength_1[:, 1],
    'Intensity_2': data_wl_wavelength_2[:, 1],
    'Normalized_Intensity_3': abs(data_wl_wavelength_2[:, 1] / data_wl_wavelength_1[:, 1]) 
})

input_wavelength = float(input("Enter the wavelength: "))
power_percentage = float(input(f"Enter the power percentage at {input_wavelength} nm: "))
closest_wavelength_index = np.argmin(np.abs(df_normalized['Wavelength'] - input_wavelength))
closest_wavelength = df_normalized.loc[closest_wavelength_index, 'Wavelength']
scaled_factor = power_percentage / df_normalized.loc[closest_wavelength_index, 'Normalized_Intensity_3']
df_normalized['Percentage_Transmission_3'] = df_normalized['Normalized_Intensity_3'] * scaled_factor
df_filtered_3 = df_normalized[(df_normalized['Wavelength'] > 400)] #& (df_normalized['Wavelength'] < 1000)]
plot_spectrum(df_filtered_3['Wavelength'], df_filtered_3['Percentage_Transmission_3'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')

##############################################################################################################################################################
# SPECTRUM WITH FIBER 4
##############################################################################################################################################################

date = "2024-03-20T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/DELL/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
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

##############################################################################################################################################################

date = "2024-03-23T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/DELL/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
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

##############################################################################################################################################################

df_normalized = pd.DataFrame({
    'Wavelength': data_wl_wavelength_1[:, 0],
    'Intensity_1': data_wl_wavelength_1[:, 1],
    'Intensity_2': data_wl_wavelength_2[:, 1],
    'Normalized_Intensity_4': abs(data_wl_wavelength_2[:, 1] / data_wl_wavelength_1[:, 1]) 
})

input_wavelength = float(input("Enter the wavelength: "))
power_percentage = float(input(f"Enter the power percentage at {input_wavelength} nm: "))
closest_wavelength_index = np.argmin(np.abs(df_normalized['Wavelength'] - input_wavelength))
closest_wavelength = df_normalized.loc[closest_wavelength_index, 'Wavelength']
scaled_factor = power_percentage / df_normalized.loc[closest_wavelength_index, 'Normalized_Intensity_4']
df_normalized['Percentage_Transmission_4'] = df_normalized['Normalized_Intensity_4'] * scaled_factor
df_filtered_4 = df_normalized[(df_normalized['Wavelength'] > 400)] #& (df_normalized['Wavelength'] < 1000)]
plot_spectrum(df_filtered_4['Wavelength'], df_filtered_4['Percentage_Transmission_4'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')

##############################################################################################################################################################
# SPECTRUM WITH FIBER 5
##############################################################################################################################################################

date = "2024-03-20T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/DELL/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
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

##############################################################################################################################################################

date = "2024-03-23T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/DELL/Documents/optofluidics-master/optofluidics-master/Python/ahmad_thesis'))
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

##############################################################################################################################################################

df_normalized = pd.DataFrame({
    'Wavelength': data_wl_wavelength_1[:, 0],
    'Intensity_1': data_wl_wavelength_1[:, 1],
    'Intensity_2': data_wl_wavelength_2[:, 1],
    'Normalized_Intensity_5': abs(data_wl_wavelength_2[:, 1] / data_wl_wavelength_1[:, 1]) 
})

input_wavelength = float(input("Enter the wavelength: "))
power_percentage = float(input(f"Enter the power percentage at {input_wavelength} nm: "))
closest_wavelength_index = np.argmin(np.abs(df_normalized['Wavelength'] - input_wavelength))
closest_wavelength = df_normalized.loc[closest_wavelength_index, 'Wavelength']
scaled_factor = power_percentage / df_normalized.loc[closest_wavelength_index, 'Normalized_Intensity_5']
df_normalized['Percentage_Transmission_5'] = df_normalized['Normalized_Intensity_5'] * scaled_factor
df_filtered_5 = df_normalized[(df_normalized['Wavelength'] > 400)] #& (df_normalized['Wavelength'] < 1000)]
plot_spectrum(df_filtered_5['Wavelength'], df_filtered_5['Percentage_Transmission_5'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Normalized Intensity vs Wavelength',
              legend_label='Normalized Intensity')

##############################################################################################################################################################
# COMBINED NORMALISED SPECTRUM WITH FIBER 
##############################################################################################################################################################

plot_spectrum_combined(df_filtered_1['Wavelength'], df_filtered_1['Percentage_Transmission_1'], 
              df_filtered_2['Percentage_Transmission_2'], df_filtered_3['Percentage_Transmission_3'],
              df_filtered_4['Percentage_Transmission_4'], df_filtered_5['Percentage_Transmission_5'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='Percentage Transmission vs Wavelength',
              legend_label_1='Percentage Transmission - Spectrum 1',
              legend_label_2='Percentage Transmission - Spectrum 2',
              legend_label_3='Percentage Transmission - Spectrum 3',
              legend_label_4='Percentage Transmission - Spectrum 4',
              legend_label_5='Percentage Transmission - Spectrum 5')

##############################################################################################################################################################
