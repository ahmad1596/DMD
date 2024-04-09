import pandas as pd
import matplotlib.pyplot as plt

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

data = pd.read_excel("C:/Users/DELL/Downloads/0.1mgmL_BSA_Tz_RawData.xlsx", skiprows=55)

wavelength = data.iloc[:, 0].values
absorbance = data.iloc[:, 1].values

min_absorbance = min(absorbance)

absorbance_shifted = absorbance + abs(min_absorbance)
plot_spectrum(wavelength, absorbance_shifted, "Wavelength (nm)", "Absorbance (a.u.)", "Absorbance vs Wavelength", legend_label="1.5uM_BSA_TR_RawData")
