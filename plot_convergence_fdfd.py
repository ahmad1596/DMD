import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

mode_1_data = pd.read_csv('C:/Users/DELL/Documents/optofluidics-master/optofluidics-master/Python/mode_1_data_fdfd.csv', header=None, names=['Nx', 'lambda', 'dx', 'lambda/dx', 'neff_real', 'neff_imag'], index_col='Nx')
mode_2_data = pd.read_csv('C:/Users/DELL/Documents/optofluidics-master/optofluidics-master/Python/mode_2_data_fdfd.csv', header=None, names=['Nx', 'lambda', 'dx', 'lambda/dx', 'neff_real', 'neff_imag'], index_col='Nx')

# Use seaborn for styling
sns.set(style="whitegrid")

# Function to add text annotation
def add_final_value_annotation(ax, final_value, title):
    ax.text(0.95, 0.90, f'Final Value: {final_value}', transform=ax.transAxes, fontsize=12, fontweight='bold', verticalalignment='top', horizontalalignment='right')
    ax.set_title(title, fontsize=16, fontweight='bold')

# Plotting for Mode 1
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 12), dpi=600)

# Real Eff. Index of Mode 1
axes[0].plot(mode_1_data['lambda/dx'], mode_1_data['neff_real'], 'o-', label='Real Eff. Index of Mode 1', linewidth=2)
axes[0].set_ylabel('Effective Index (Real)', fontsize=14, fontweight="bold")
axes[0].tick_params(axis="both", which="major", labelsize=12, direction="in")
axes[0].grid(color="gray", linestyle="--", linewidth=0.5)
axes[0].legend(loc="center right", fontsize=8)
add_final_value_annotation(axes[0], f'{mode_1_data["neff_real"].iloc[-1]:.7f}', 'Mode 1 - Real Effective Index')
axes[0].xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
axes[0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.5f'))

# Imag Eff. Index of Mode 1
axes[1].plot(mode_1_data['lambda/dx'], mode_1_data['neff_imag'], 'o-', label='Imag Eff. Index of Mode 1', linewidth=2)
axes[1].set_ylabel('Effective Index (Imaginary)', fontsize=14, fontweight="bold")
axes[1].set_xlabel(r'$\lambda/\Delta x$', fontsize=14, fontweight="bold")
axes[1].tick_params(axis="both", which="major", labelsize=12, direction="in")
axes[1].grid(color="gray", linestyle="--", linewidth=0.5)
axes[1].legend(loc="center right", fontsize=8)
add_final_value_annotation(axes[1], f'{mode_1_data["neff_imag"].iloc[-1]:.6e}', 'Mode 1 - Imaginary Effective Index')
axes[1].xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: format(x, '.2e')))

plt.tight_layout()
plt.show()

# Plotting for Mode 2
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 12), dpi=600)

# Real Eff. Index of Mode 2
axes[0].plot(mode_2_data['lambda/dx'], mode_2_data['neff_real'], 'o-', label='Real Eff. Index of Mode 2', linewidth=2)
axes[0].set_ylabel('Effective Index (Real)', fontsize=14, fontweight="bold")
axes[0].tick_params(axis="both", which="major", labelsize=12, direction="in")
axes[0].grid(color="gray", linestyle="--", linewidth=0.5)
axes[0].legend(loc="center right", fontsize=8)
add_final_value_annotation(axes[0], f'{mode_2_data["neff_real"].iloc[-1]:.7f}', 'Mode 2 - Real Effective Index')
axes[0].xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
axes[0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.5f'))

# Imag Eff. Index of Mode 2
axes[1].plot(mode_2_data['lambda/dx'], mode_2_data['neff_imag'], 'o-', label='Imag Eff. Index of Mode 2', linewidth=2)
axes[1].set_ylabel('Effective Index (Imaginary)', fontsize=14, fontweight="bold")
axes[1].set_xlabel(r'$\lambda/\Delta x$', fontsize=14, fontweight="bold")
axes[1].tick_params(axis="both", which="major", labelsize=12, direction="in")
axes[1].grid(color="gray", linestyle="--", linewidth=0.5)
axes[1].legend(loc="center right", fontsize=8)
add_final_value_annotation(axes[1], f'{mode_2_data["neff_imag"].iloc[-1]:.6e}', 'Mode 2 - Imaginary Effective Index')
axes[1].xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: format(x, '.2e')))

plt.tight_layout()
plt.show()
