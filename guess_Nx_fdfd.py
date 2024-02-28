import numpy as np

um = 1e-6  
lam = 0.65 * um  
lam_dx_ratio_target = 15
dx = lam / lam_dx_ratio_target  
Nx = int(np.ceil((2 * 30 * um) / dx)) + 1 

print("lam/dx: {:.2f}".format(lam_dx_ratio_target))
print("lam: {:.2f} nm".format(lam/1e-9))
print("dx: {:.2f} nm".format(dx/1e-9))
print("Nx: {}".format(Nx))
