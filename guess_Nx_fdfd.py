import numpy as np

um = 1e-6  
lam = 0.65 * um  
lam_dx_ratio_target = 10 
dx = lam / lam_dx_ratio_target  
Nx = int(np.ceil((2 * 26 * um) / dx)) + 1 

x = np.linspace(-(26) * um, (26) * um, Nx) 

print("lam/dx target: {:.2f}".format(lam_dx_ratio_target))
print("Actual lam/dx: {:.2f}".format(lam/dx))
print("Nx: {}".format(Nx))
