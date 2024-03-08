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

date = "2022-03-06T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/DELL/Downloads'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 3
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
    laser_wavelength = 633  
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
data_wl_wavelength = np.zeros((len(data_average), 2), dtype=float)
data_wl_wavelength[:, 0] = wavelengths
data_wl_wavelength[:, 1] = data_average
plot_spectrum(data_wl_wavelength[:, 0], data_wl_wavelength[:, 1],
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
csv_file_path1 = os.path.join(folder_path, "Ref_QE65000_Spectrometer.csv")
pd.DataFrame(data_wl_wavelength).to_csv(csv_file_path1, header=None, index=None)

##############################################################################################################################################################

date = "2022-03-06T" 
DIRPATH = os.path.normpath(os.path.abspath('C:/Users/DELL/Downloads'))
FILENAMES = os.listdir(DIRPATH)
print(DIRPATH)
for i, f in enumerate(FILENAMES):
    print(i, "::", f)
file = 3
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
    laser_wavelength = 633  
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
data_wl_wavelength = np.zeros((len(data_average), 2), dtype=float)
data_wl_wavelength[:, 0] = wavelengths
data_wl_wavelength[:, 1] = data_average
plot_spectrum(data_wl_wavelength[:, 0], data_wl_wavelength[:, 1],
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
csv_file_path2 = os.path.join(folder_path, "Ref_USB2000_Spectrometer.csv")
pd.DataFrame(data_wl_wavelength).to_csv(csv_file_path2, header=None, index=None)

##############################################################################################################################################################
