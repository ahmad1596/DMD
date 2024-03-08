import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from datetime import datetime
import seaborn as sns
from refractivesqlite import dboperations as DB

sns.set_context('notebook', font_scale=2)
sns.set_style('darkgrid')
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif', size=18)
plt.rc('text.latex', preamble=r'\setlength{\parindent}{0em}')
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Accent_r.colors)

print('\nLoading refractiveindex.info database')
dbpath = 'C:\\Users\\DELL\\Documents\\optofluidics-master\\optofluidics-master\\Python\\ARROW model\\refractive.db'
db = DB.Database(dbpath)

RESOLUTION = int(1e4)
MAX_RESONANCE_ORDER = 10
STRUT_THICKNESS_PLOT_RANGE_START_NM = 100
STRUT_THICKNESS_PLOT_RANGE_END_NM = 700
STRUT_THICKNESS_PLOT_RANGE_GUESS_NM = 300

FIBRE_NAME = 'Silica glass AR fibre'
EXPECTED_D_MIN = False
WAVELENGTH_OF_INTEREST_NM = None
SOLVENT = "Water"
GLASS = "Silica"

def lambda_antires(d_nm, n_glass, n_solvent, m, approximate_wavelength_nm=None):
    if type(approximate_wavelength_nm) is not np.ndarray:
        guess_nm = STRUT_THICKNESS_PLOT_RANGE_GUESS_NM * np.ones_like(d_nm)
        best_guess = lambda_antires(d_nm, n_glass, n_solvent, m, guess_nm)
        result = lambda_antires(d_nm, n_glass, n_solvent, m, best_guess)
        return result
    ret_list = []
    for d_nm, approximate_wavelength_nm in zip(d_nm, approximate_wavelength_nm):
        try:
            n_g = n_glass(approximate_wavelength_nm)
            n_s = n_solvent(approximate_wavelength_nm)
        except:
            ret_list.append(None)
            continue
        ret = 4 * d_nm / (2 * m + 1) * np.sqrt(np.square(n_g) - np.square(n_s))
        ret_list.append(ret)
    return np.array(ret_list)

material_water = db.get_material(2707)
n_water = lambda wavelength_nm: material_water.get_refractiveindex(wavelength_nm)

material_silica = db.get_material(409)
n_silica = lambda wavelength_nm: material_silica.get_refractiveindex(wavelength_nm)

material_methanol = db.get_material(722)
n_methanol = lambda wavelength_nm: material_methanol.get_refractiveindex(wavelength_nm)

material_isopropanol = db.get_material(731)
n_isopropanol = lambda wavelength_nm: material_isopropanol.get_refractiveindex(wavelength_nm)

material_air = db.get_material(2513)
n_air = lambda wavelength_nm: material_air.get_refractiveindex(wavelength_nm)

material_glycerol = db.get_material(747)
n_glycerol = lambda wavelength_nm: material_glycerol.get_refractiveindex(wavelength_nm)

material_sf6 = db.get_material(983)
n_sf6 = lambda wavelength_nm: material_sf6.get_refractiveindex(wavelength_nm)

V_1 = 0.20
V_2 = 1 - V_1
U_1 = V_1
U_2 = V_2
mixtureLabel = f"glycerol {100*V_1:.0f}percent(v/v pre-mix) in water"
n_mixture = lambda wavelength_nm: U_1 * material_glycerol.get_refractiveindex(wavelength_nm) + U_2 * material_water.get_refractiveindex(wavelength_nm)

refractiveindices_solvents = {
    "Air": n_air,
    "Water": n_water,
    "Methanol": n_methanol,
    "Isopropanol": n_isopropanol,
    "Glycerol": n_glycerol,
    "mixture": n_mixture
}

dArray_nm = np.linspace(STRUT_THICKNESS_PLOT_RANGE_START_NM, STRUT_THICKNESS_PLOT_RANGE_END_NM, RESOLUTION)
mArray = np.arange(0, MAX_RESONANCE_ORDER + 1, 1)
n_solvent = refractiveindices_solvents[SOLVENT]

if GLASS == 'SF6':
    n_glass = n_sf6
elif GLASS == 'Silica':
    n_glass = n_silica

fig = plt.figure(figsize=(14, 8))
ax1 = plt.gca()

for m in mArray:
    plt.plot(dArray_nm, lambda_antires(dArray_nm, n_glass, n_solvent, m), label=f"m = {m}, AR", linestyle='--',  linewidth=4.0)
    plt.plot(dArray_nm, lambda_antires(dArray_nm, n_glass, n_solvent, m + 0.5), label=f"m = {m+1}, R", linestyle='-',  linewidth=4.0,
             color=plt.gca().lines[-1].get_color())

EXPECTED_D_MIN = 180
EXPECTED_D_MAX = 220

if EXPECTED_D_MIN != False:
    plt.axvspan(EXPECTED_D_MIN, EXPECTED_D_MAX, alpha=0.5, color='grey')
if WAVELENGTH_OF_INTEREST_NM != None:
    plt.axhspan(*WAVELENGTH_OF_INTEREST_NM, alpha=0.5, color='green')

plt.xlabel('Wall thickness $d$ / nm')
plt.ylabel('$m^{th}$ (anti-) resonance / nm')
     

if SOLVENT == 'mixture':
    solventName = mixtureLabel
else:
    solventName = SOLVENT.lower()
solventName = solventName.replace('percent', '\%')

plt.title(f"\\textbf{{Antiresonant guidance chart}}\n\\underline{{{FIBRE_NAME}}}, filled with \\underline{{{solventName}}}\nAntiresonances (-\hspace{{1pt}}-\hspace{{1pt}}-) and resonances (---) of order $m$ for a given wall thickness\n Wall thickness range for given fibre indicated by grey band")

ax1.grid(True, which='minor', linestyle='--', color='w', linewidth=0.6)
ax1.xaxis.set_minor_locator(plt.MultipleLocator(50))
ax1.yaxis.set_minor_locator(plt.MultipleLocator(25))

print("\nConfigurations:")
print(f"Material for Glass: {GLASS}")
print(f"Material for Solvent: {SOLVENT}")
print(f"Fiber Name: {FIBRE_NAME}")
print(f"Expected Minimum Wall Thickness: {EXPECTED_D_MIN} nm")
print(f"Expected Maximum Wall Thickness: {EXPECTED_D_MAX} nm" if EXPECTED_D_MAX else "No Maximum Wall Thickness specified")
print(f"Wavelength of Interest: {WAVELENGTH_OF_INTEREST_NM} nm" if WAVELENGTH_OF_INTEREST_NM else "No Wavelength of Interest specified")
print(f"Resolution: {RESOLUTION}")
print(f"Maximum Resonance Order: {MAX_RESONANCE_ORDER}")
print(f"Strut Thickness Plot Range Start: {STRUT_THICKNESS_PLOT_RANGE_START_NM} nm")
print(f"Strut Thickness Plot Range End: {STRUT_THICKNESS_PLOT_RANGE_END_NM} nm")
print(f"Strut Thickness Plot Range Guess: {STRUT_THICKNESS_PLOT_RANGE_GUESS_NM} nm")

plt.tight_layout()
if not os.path.exists('./plots/'):
    os.mkdir('./plots/')

print("Saving .png")
plt.savefig('./plots/plot.png', dpi=600)
print("Saving .pdf")
plt.savefig('./plots/plot.pdf', dpi=600)

timestamped_filename = "./plots/plot-{}".format(datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))
shutil.copy2('./plots/plot.pdf', timestamped_filename+'.pdf')
shutil.copy2('./plots/plot.png', timestamped_filename+'.png')
shutil.copy2('./arrow.py', timestamped_filename+'-script.py')