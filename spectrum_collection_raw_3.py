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

def plot_spectrum_combined(x_values, intensities_1, intensities_2, intensities_3, x_label, y_label, title, legend_label_1=None, legend_label_2=None, legend_label_3=None, legend_order=None):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    ax.plot(x_values, intensities_1, label=legend_label_1)
    ax.plot(x_values, intensities_2, label=legend_label_2)
    ax.plot(x_values, intensities_3, label=legend_label_3)
    ax.set_xlabel(x_label, fontsize=14, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=16, fontweight="bold")
    plt.show()

    if legend_order:
        handles, labels = ax.get_legend_handles_labels()
        reordered_labels = [labels[i] for i in legend_order]
        ax.legend(handles, reordered_labels, loc="upper right", fontsize=8)
    else:
        ax.legend(loc="upper right", fontsize=8)

    plt.show()
##############################################################################################################################################################
# SPECTRUM WITH FIBER 1
##############################################################################################################################################################

date = "2024-03-27T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/DELL/Documents/Python'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 39
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
data_wl_wavelength_2 = np.zeros((len(data_average), 2), dtype=float)
data_wl_wavelength_2[:, 0] = wavelengths
data_wl_wavelength_2[:, 1] = data_average
plot_spectrum(data_wl_wavelength_2[:, 0], data_wl_wavelength_2[:, 1],
              x_label='Wavelength (nm)', y_label='Intensity (a.u.)',
              title='Average Spectrum',
              legend_label='Intensity at Wavelength')

##############################################################################################################################################################

df_measured = pd.DataFrame({
    'Wavelength': data_wl_wavelength_2[:, 0],
    'Intensity_1': data_wl_wavelength_2[:, 1]
})

df_measured['Counts_1'] = df_measured['Intensity_1']
df_filtered_1 =  df_measured[(df_measured['Wavelength'] > 500) & (df_measured['Wavelength'] < 750)]

##############################################################################################################################################################
# SPECTRUM WITH FIBER 2
##############################################################################################################################################################

date = "2024-03-27T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/DELL/Documents/Python'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 39
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    print("\nKeys: %s" % f.keys())
    keys = list(f.keys())
key = keys[1]
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

##############################################################################################################################################################

df_measured = pd.DataFrame({
    'Wavelength': data_wl_wavelength_2[:, 0],
    'Intensity_2': data_wl_wavelength_2[:, 1]
})

df_measured['Counts_2'] = df_measured['Intensity_2']
df_filtered_2 =  df_measured[(df_measured['Wavelength'] > 500) & (df_measured['Wavelength'] < 750)]


##############################################################################################################################################################
# SPECTRUM WITH FIBER 3
##############################################################################################################################################################

date = "2024-03-27T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/DELL/Documents/Python'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 39
with h5py.File(DIRPATH + '\\' + FILENAMES[file], 'r') as f:
    print("\nKeys: %s" % f.keys())
    keys = list(f.keys())
key = keys[2]
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

##############################################################################################################################################################

df_measured = pd.DataFrame({
    'Wavelength': data_wl_wavelength_2[:, 0],
    'Intensity_3': data_wl_wavelength_2[:, 1]
})

df_measured['Counts_3'] = df_measured['Intensity_3'] 
df_filtered_3 =  df_measured[(df_measured['Wavelength'] > 500) & (df_measured['Wavelength'] < 750)]
plot_spectrum(df_filtered_3['Wavelength'], df_filtered_3['Counts_3'],
              x_label='Wavelength (nm)', y_label='Percentage Transmission (%)',
              title='measured Intensity vs Wavelength',
              legend_label='measured Intensity')


##############################################################################################################################################################
# COMBINED SPECTRUM WITH FIBER 
##############################################################################################################################################################

plot_spectrum_combined(df_filtered_1['Wavelength'], df_filtered_1['Counts_1'], 
              df_filtered_2['Counts_2'], df_filtered_3['Counts_3'],
              x_label='Wavelength (nm)', y_label='Raw Counts (a.u.)',
              title='450nm Lamp Excitation on 1 nM evaporated RhB',
              legend_label_1='Fiber All',
              legend_label_2='Fiber Core',
              legend_label_3='Fiber Ring')

##############################################################################################################################################################  