import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt
import time

def check_errors(n, lam, dx):
    if n.shape[1] != n.shape[0]:
        print('Expecting square problem space...\n')
    if lam / dx < 10:
        print('lam/dx < 10: this will likely cause discretization errors...\n')

def ColumnToMatrix(C, Nx, Ny):
    C = np.reshape(C, (Nx, Ny))
    M = np.transpose(C)
    return M

def MatrixToColumn(M): 
    M = np.transpose(M)
    C = M.reshape(-1,1)
    return C

def initialize_parameters(n, lam, dx):
    eps0 = 8.85e-12
    mu0 = 4 * np.pi * 10**-7
    c = 3e8
    Nx = n.shape[0]
    f = c / lam
    w = 2 * np.pi * f
    k0 = 2 * np.pi / lam
    PML_Depth = 10
    PML_TargetLoss = 1e-5
    PML_PolyDegree = 3
    PML_SigmaMax = (PML_PolyDegree + 1) / 2 * eps0 * c / PML_Depth / dx * np.log(1 / PML_TargetLoss)
    Epsr = n**2
    Epsr = MatrixToColumn(Epsr)
    return eps0, mu0, c, Nx, f, w, k0, PML_Depth, PML_TargetLoss, PML_PolyDegree, PML_SigmaMax, Epsr

def calculate_Ux_Uy_Vx_Vy(Nx):
    print('Calculating Ux, Uy, Vx, Vy...\n') 
    I = sparse.eye(Nx * Nx)
    I = sparse.csr_matrix(I)
    idx_x = 0
    idx_y = 1
    Epsx = np.zeros((1, Nx * Nx))
    Epsy = np.zeros((1, Nx * Nx))
    Epsz = np.zeros((1, Nx * Nx))
    Ax_idxi = np.zeros((1, 2 * Nx * Nx + 1))
    Ax_idxj = np.zeros((1, 2 * Nx * Nx + 1))
    Ax_vals = np.zeros((1, 2 * Nx * Nx + 1), dtype=complex)
    Ay_idxi = np.zeros((1, 2 * Nx * Nx + 1))
    Ay_idxj = np.zeros((1, 2 * Nx * Nx + 1))
    Ay_vals = np.zeros((1, 2 * Nx * Nx + 1), dtype=complex)
    Bx_idxi = np.zeros((1, 2 * Nx * Nx + 1))
    Bx_idxj = np.zeros((1, 2 * Nx * Nx + 1))
    Bx_vals = np.zeros((1, 2 * Nx * Nx + 1), dtype=complex)
    By_idxi = np.zeros((1, 2 * Nx * Nx + 1))
    By_idxj = np.zeros((1, 2 * Nx * Nx + 1))
    By_vals = np.zeros((1, 2 * Nx * Nx + 1), dtype=complex)
    Cx_idxi = np.zeros((1, 2 * Nx * Nx + 1))
    Cx_idxj = np.zeros((1, 2 * Nx * Nx + 1))
    Cx_vals = np.zeros((1, 2 * Nx * Nx + 1), dtype=complex)
    Cy_idxi = np.zeros((1, 2 * Nx * Nx + 1))
    Cy_idxj = np.zeros((1, 2 * Nx * Nx + 1))
    Cy_vals = np.zeros((1, 2 * Nx * Nx + 1), dtype=complex)
    Dx_idxi = np.zeros((1, 2 * Nx * Nx + 1))
    Dx_idxj = np.zeros((1, 2 * Nx * Nx + 1))
    Dx_vals = np.zeros((1, 2 * Nx * Nx + 1), dtype=complex)
    Dy_idxi = np.zeros((1, 2 * Nx * Nx + 1))
    Dy_idxj = np.zeros((1, 2 * Nx * Nx + 1))
    Dy_vals = np.zeros((1, 2 * Nx * Nx + 1), dtype=complex)
    return I, idx_x, idx_y, Epsx, Epsy, Epsz, Ax_idxi, Ax_idxj, Ax_vals, Ay_idxi, Ay_idxj, Ay_vals, Bx_idxi, Bx_idxj, Bx_vals, By_idxi, By_idxj, By_vals, Cx_idxi, Cx_idxj, Cx_vals, Cy_idxi, Cy_idxj, Cy_vals, Dx_idxi, Dx_idxj, Dx_vals, Dy_idxi, Dy_idxj, Dy_vals

def calculate_index_distances(Nx, idx_x, idx_y):
    idx_x += 1
    if idx_x > Nx:
        idx_y += 1
        idx_x = 1
    West_Dist = idx_x - 1
    North_Dist = idx_y - 1
    East_Dist = Nx - idx_x
    South_Dist = Nx - idx_y
    return idx_x, idx_y, West_Dist, North_Dist, East_Dist, South_Dist

def calculate_Eps_values(i, Nx, Epsr):
    if i - 1 - Nx > 0:
        Epsx_i = (Epsr[i - 1] + Epsr[i - 1 - Nx]) / 2
    else:
        Epsx_i = Epsr[i - 1]
    if i - 1 - 1 > 0:
        Epsy_i = (Epsr[i - 1] + Epsr[i - 2]) / 2
    else:
        Epsy_i = Epsr[i - 1]
    if i - 1 - 1 - Nx > 0:
        Epsz_i = (Epsr[i - 1] + Epsr[i - 2] + Epsr[i - Nx - 1] + Epsr[i - 1 - Nx - 1]) / 4
    else:
        Epsz_i = Epsr[i - 1]
    return Epsx_i, Epsy_i, Epsz_i

def ModeSolverFD(dx, n, lam, beta, NoModes):
    check_errors(n, lam, dx)
    eps0, mu0, c, Nx, f, w, k0, PML_Depth, PML_TargetLoss, PML_PolyDegree, PML_SigmaMax, Epsr = initialize_parameters(n, lam, dx)
    I, idx_x, idx_y, Epsx, Epsy, Epsz, Ax_idxi, Ax_idxj, Ax_vals, Ay_idxi, Ay_idxj, Ay_vals, Bx_idxi, Bx_idxj, Bx_vals, By_idxi, By_idxj, By_vals, Cx_idxi, Cx_idxj, Cx_vals, Cy_idxi, Cy_idxj, Cy_vals, Dx_idxi, Dx_idxj, Dx_vals, Dy_idxi, Dy_idxj, Dy_vals = calculate_Ux_Uy_Vx_Vy(Nx)
  
    for i in range(1,Nx*Nx+1):
        idx_x, idx_y, West_Dist, North_Dist, East_Dist, South_Dist = calculate_index_distances(Nx, idx_x, idx_y)
        Epsx[:, i - 1], Epsy[:, i - 1], Epsz[:, i - 1] = calculate_Eps_values(i, Nx, Epsr)

        # Sx, Sy
        if West_Dist <= PML_Depth:
            Sx_Ey = 1-PML_SigmaMax*(1-West_Dist/PML_Depth)**PML_PolyDegree*1j/w/eps0/np.sqrt(Epsy[:,i-1])
            Sx_Ez = 1-PML_SigmaMax*(1-West_Dist/PML_Depth)**PML_PolyDegree*1j/w/eps0/np.sqrt(Epsz[:,i-1])
            Sx_Hy = 1-PML_SigmaMax*(1-(West_Dist-0.5)/PML_Depth)**PML_PolyDegree*1j/w/eps0/np.sqrt(Epsx[:,i-1])
            Sx_Hz = 1-PML_SigmaMax*(1-(West_Dist-0.5)/PML_Depth)**PML_PolyDegree*1j/w/eps0/np.sqrt(Epsr[i-1])
        elif East_Dist <= PML_Depth:
            Sx_Ey = 1-PML_SigmaMax*(1-(East_Dist-0.5)/PML_Depth)**PML_PolyDegree*1j/w/eps0/np.sqrt(Epsy[:,i-1])
            Sx_Ez = 1-PML_SigmaMax*(1-(East_Dist-0.5)/PML_Depth)**PML_PolyDegree*1j/w/eps0/np.sqrt(Epsz[:,i-1])
            Sx_Hy = 1-PML_SigmaMax*(1-East_Dist/PML_Depth)**PML_PolyDegree*1j/w/eps0/np.sqrt(Epsx[:,i-1])
            Sx_Hz = 1-PML_SigmaMax*(1-East_Dist/PML_Depth)**PML_PolyDegree*1j/w/eps0/np.sqrt(Epsr[i-1]) 
        if North_Dist <= PML_Depth:
            Sy_Ex = 1-PML_SigmaMax*(1-North_Dist/PML_Depth)**PML_PolyDegree*1j/w/eps0/np.sqrt(Epsx[:,i-1])
            Sy_Ez = 1-PML_SigmaMax*(1-North_Dist/PML_Depth)**PML_PolyDegree*1j/w/eps0/np.sqrt(Epsz[:,i-1])
            Sy_Hx = 1-PML_SigmaMax*(1-(North_Dist-0.5)/PML_Depth)**PML_PolyDegree*1j/w/eps0/np.sqrt(Epsy[:,i-1])
            Sy_Hz = 1-PML_SigmaMax*(1-(North_Dist-0.5)/PML_Depth)**PML_PolyDegree*1j/w/eps0/np.sqrt(Epsr[i-1])
        elif South_Dist <= PML_Depth:
            Sy_Ex = 1-PML_SigmaMax*(1-(South_Dist-0.5)/PML_Depth)**PML_PolyDegree*1j/w/eps0/np.sqrt(Epsx[:,i-1])
            Sy_Ez = 1-PML_SigmaMax*(1-(South_Dist-0.5)/PML_Depth)**PML_PolyDegree*1j/w/eps0/np.sqrt(Epsz[:,i-1])
            Sy_Hx = 1-PML_SigmaMax*(1-South_Dist/PML_Depth)**PML_PolyDegree*1j/w/eps0/np.sqrt(Epsy[:,i-1])
            Sy_Hz = 1-PML_SigmaMax*(1-South_Dist/PML_Depth)**PML_PolyDegree*1j/w/eps0/np.sqrt(Epsr[i-1])
        #Ax
        Ax_idxi[:,2*i-1] = i
        Ax_idxj[:,2*i-1] = i
        Ax_vals[:,2*i-1] = -1/Sx_Ez
        if i <= Nx*Nx-1:
            Ax_idxi[:,2*i] = i
            Ax_idxj[:,2*i] = i+1
            Ax_vals[:,2*i] = 1/Sx_Ez          
        #Bx
        Bx_idxi[:,2*i-1] = i
        Bx_idxj[:,2*i-1] = i
        Bx_vals[:,2*i-1] = -1/Sx_Ey
        if i <= Nx*Nx-1:
            Bx_idxi[:,2*i] = i
            Bx_idxj[:,2*i] = i+1
            Bx_vals[:,2*i] = 1/Sx_Ey
        #Ay
        Ay_idxi[:,2*i-1] = i
        Ay_idxj[:,2*i-1] = i
        Ay_vals[:,2*i-1] = -1/Sy_Ez   
        if i+Nx-1 <= Nx*Nx-1:
            Ay_idxi[:,2*i] = i
            Ay_idxj[:,2*i] = i+Nx
            Ay_vals[:,2*i] = 1/Sy_Ez
        #By
        By_idxi[:,2*i-1] = i
        By_idxj[:,2*i-1] = i
        By_vals[:,2*i-1] = -1/Sy_Ex
        if i+Nx-1 <= Nx*Nx-1:
            By_idxi[:,2*i] = i
            By_idxj[:,2*i] = i + Nx
            By_vals[:,2*i] = 1/Sy_Ex
        #Cx
        Cx_idxi[:,2*i-1] = i
        Cx_idxj[:,2*i-1] = i
        Cx_vals[:,2*i-1] = 1/Sx_Hz 
           
        if i-2 >= 0:
            Cx_idxi[:,2*i] = i       
            Cx_idxj[:,2*i] = i-1
            Cx_vals[:,2*i] = -1/Sx_Hz
        #Cy
        Cy_idxi[:,2*i-1] = i
        Cy_idxj[:,2*i-1] = i
        Cy_vals[:,2*i-1] = 1/Sy_Hz
        if i-Nx-1 >= 0:
            Cy_idxi[:,2*i] = i
            Cy_idxj[:,2*i] = i-Nx
            Cy_vals[:,2*i] = -1/Sy_Hz
        #Dx
        Dx_idxi[:,2*i-1] = i
        Dx_idxj[:,2*i-1] = i
        Dx_vals[:,2*i-1] = 1/Sx_Hy
        if i-2 >= 0:
            Dx_idxi[:,2*i] = i
            Dx_idxj[:,2*i] = i-1
            Dx_vals[:,2*i] = -1/Sx_Hy
        #Dy
        Dy_idxi[:,2*i-1] = i
        Dy_idxj[:,2*i-1] = i
        Dy_vals[:,2*i-1] = 1/Sy_Hx
        if i-Nx-1 >= 0:
            Dy_idxi[:,2*i] = i
            Dy_idxj[:,2*i] = i-Nx
            Dy_vals[:,2*i] = -1/Sy_Hx
    Ax_idxi[Ax_idxi == -1] = []
    Ax_idxj[Ax_idxi == -1] = []
    Ax_vals[Ax_idxi == -1] = []
    flatten_Ax_idxi = Ax_idxi.ravel()
    flatten_Ax_idxj = Ax_idxj.ravel()                       
    flatten_Ax_vals = Ax_vals.ravel()           
    Ax = csr_matrix((flatten_Ax_vals, (flatten_Ax_idxi, flatten_Ax_idxj)), shape=(Nx*Nx+1,Nx*Nx+1))
    Ax = Ax[1:, 1:]
    Ay_idxi[Ay_idxi == -1] = []
    Ay_idxj[Ay_idxi == -1] = []
    Ay_vals[Ay_idxi == -1] = []
    flatten_Ay_idxi = Ay_idxi.ravel()
    flatten_Ay_idxj = Ay_idxj.ravel()                       
    flatten_Ay_vals = Ay_vals.ravel()           
    Ay = csr_matrix((flatten_Ay_vals, (flatten_Ay_idxi, flatten_Ay_idxj)), shape=(Nx*Nx+1,Nx*Nx+1))
    Ay = Ay[1:, 1:]
    Bx_idxi[Bx_idxi == -1] = []
    Bx_idxj[Bx_idxi == -1] = []
    Bx_vals[Bx_idxi == -1] = []
    flatten_Bx_idxi = Bx_idxi.ravel()
    flatten_Bx_idxj = Bx_idxj.ravel()                       
    flatten_Bx_vals = Bx_vals.ravel()           
    Bx = csr_matrix((flatten_Bx_vals, (flatten_Bx_idxi, flatten_Bx_idxj)), shape=(Nx*Nx+1,Nx*Nx+1))
    Bx = Bx[1:, 1:]  
    By_idxi[By_idxi == -1] = []
    By_idxj[By_idxi == -1] = []
    By_vals[By_idxi == -1] = []
    flatten_By_idxi = By_idxi.ravel()
    flatten_By_idxj = By_idxj.ravel()                       
    flatten_By_vals = By_vals.ravel()           
    By = csr_matrix((flatten_By_vals, (flatten_By_idxi, flatten_By_idxj)), shape=(Nx*Nx+1,Nx*Nx+1))
    By = By[1:, 1:]
    Cx_idxi[Cx_idxi == -1] = []
    Cx_idxj[Cx_idxi == -1] = []
    Cx_vals[Cx_idxi == -1] = []
    flatten_Cx_idxi = Cx_idxi.ravel()
    flatten_Cx_idxj = Cx_idxj.ravel()                       
    flatten_Cx_vals = Cx_vals.ravel()           
    Cx = csr_matrix((flatten_Cx_vals, (flatten_Cx_idxi, flatten_Cx_idxj)), shape=(Nx*Nx+1,Nx*Nx+1))
    Cx = Cx[1:, 1:]  
    Cy_idxi[Cy_idxi == -1] = []
    Cy_idxj[Cy_idxi == -1] = []
    Cy_vals[Cy_idxi == -1] = []
    flatten_Cy_idxi = Cy_idxi.ravel()
    flatten_Cy_idxj = Cy_idxj.ravel()                       
    flatten_Cy_vals = Cy_vals.ravel()           
    Cy = csr_matrix((flatten_Cy_vals, (flatten_Cy_idxi, flatten_Cy_idxj)), shape=(Nx*Nx+1,Nx*Nx+1))
    Cy = Cy[1:, 1:]
    Dx_idxi[Dx_idxi == -1] = []
    Dx_idxj[Dx_idxi == -1] = []
    Dx_vals[Dx_idxi == -1] = []
    flatten_Dx_idxi = Dx_idxi.ravel()
    flatten_Dx_idxj = Dx_idxj.ravel()                       
    flatten_Dx_vals = Dx_vals.ravel()           
    Dx = csr_matrix((flatten_Dx_vals, (flatten_Dx_idxi, flatten_Dx_idxj)), shape=(Nx*Nx+1,Nx*Nx+1))
    Dx = Dx[1:, 1:]
    Dy_idxi[Dy_idxi == -1] = []
    Dy_idxj[Dy_idxi == -1] = []
    Dy_vals[Dy_idxi == -1] = []
    flatten_Dy_idxi = Dy_idxi.ravel()
    flatten_Dy_idxj = Dy_idxj.ravel()                       
    flatten_Dy_vals = Dy_vals.ravel()           
    Dy = csr_matrix((flatten_Dy_vals, (flatten_Dy_idxi, flatten_Dy_idxj)), shape=(Nx*Nx+1,Nx*Nx+1))
    Dy = Dy[1:, 1:]
    flatten_Epsx = Epsx.ravel()
    flatten_Epsy = Epsy.ravel()
    flatten_Epsz = Epsz.ravel()
    Epsx = csr_matrix((flatten_Epsx, (np.arange(1,Nx*Nx+1), np.arange(1,Nx*Nx+1))))
    Epsy = csr_matrix((flatten_Epsy, (np.arange(1,Nx*Nx+1), np.arange(1,Nx*Nx+1))))
    invEpsz = csr_matrix((1/flatten_Epsz, (np.arange(1,Nx*Nx+1), np.arange(1,Nx*Nx+1))))
    Epsx = Epsx[1:, 1:]
    Epsy = Epsy[1:, 1:]
    invEpsz = invEpsz[1:, 1:]
    Ax = Ax/dx 
    Bx = Bx/dx
    Cx = Cx/dx 
    Dx = Dx/dx
    Ay = Ay/dx 
    By = By/dx
    Cy = Cy/dx 
    Dy = Dy/dx
    ## Qxx, Qyy, Qxy, Qyx
    print('Calculating Qs...\n')
    Qxx = -k0**(-2)*Ax*Dy*Cx*invEpsz*By + (Epsy + k0**(-2)*Ax*Dx)*(k0**2*I+Cy*invEpsz*By)
    Qyy = -k0**(-2)*Ay*Dx*Cy*invEpsz*Bx + (Epsx + k0**(-2)*Ay*Dy)*(k0**2*I+Cx*invEpsz*Bx)
    Qxy = k0**(-2)*Ax*Dy*(k0**2*I + Cx*invEpsz*Bx) - (Epsy + k0**(-2)*Ax*Dx)*Cy*invEpsz*Bx
    Qyx = k0**(-2)*Ay*Dx*(k0**2*I + Cy*invEpsz*By) - (Epsx + k0**(-2)*Ay*Dy)*Cx*invEpsz*By
    QxxQxy = sparse.hstack([Qxx, Qxy])
    QyxQyy = sparse.hstack([Qyx, Qyy])
    Q = sparse.vstack([QxxQxy, QyxQyy])
    ## Diagonalisation
    print('Taking Eigenvalues and Eigenvectors...\n')
    import scipy.sparse.linalg as sla
    eigvalues, eigvectors = sla.eigs(Q, k = NoModes, sigma = beta**2)
    beta = np.sqrt(np.diag(eigvalues))
    # Ex, Ey, Ez
    print('Calculating Ex, Ey, Ez, Hx, Hy, Hz...\n')
    Ex = np.zeros((Nx*Nx, NoModes), dtype=complex)
    Ey = np.zeros((Nx*Nx, NoModes), dtype=complex)
    Ez = np.zeros((Nx*Nx, NoModes), dtype=complex)
    Hx = np.zeros((Nx*Nx, NoModes), dtype=complex)
    Hy = np.zeros((Nx*Nx, NoModes), dtype=complex)
    Hz = np.zeros((Nx*Nx, NoModes), dtype=complex)     
    for i in range(0,NoModes):
        Hx[:,i] = eigvectors[np.arange(0, Nx*Nx), i]
        Hy[:,i] = eigvectors[np.arange(Nx*Nx,2*Nx*Nx),i]
        Ez[:,i] = invEpsz*(-Dy*Hx[:,i]+ Dx*Hy[:,i])/1j/w/eps0   
        Ey[:,i] = (-1j*w*mu0*Hx[:,i] - Ay*Ez[:,i])/1j/beta[i][i]
        Ex[:,i] = (1j*w*mu0*Hy[:,i] - Ax*Ez[:,i])/1j/beta[i][i]
        Hz[:,i] = -(-By*Ex[:,i] + Bx*Ey[:,i])/1j/w/mu0    
    ## Results
    RetVal = {}
    RetVal_Ex = {}
    RetVal_Ey = {}
    RetVal_Ez = {}
    RetVal_Hx = {}
    RetVal_Hy = {}
    RetVal_Hz = {}
    RetVal_Eabs = {}
    RetVal_Habs = {}
    for i in range(0,NoModes):
        RetVal_Ex[i] = ColumnToMatrix(Ex[:,i], Nx, Nx)
        RetVal_Ey[i] = ColumnToMatrix(Ey[:,i], Nx, Nx)
        RetVal_Ez[i] = ColumnToMatrix(Ez[:,i], Nx, Nx)
        RetVal_Hx[i] = ColumnToMatrix(Hx[:,i], Nx, Nx)
        RetVal_Hy[i] = ColumnToMatrix(Hy[:,i], Nx, Nx)
        RetVal_Hz[i] = ColumnToMatrix(Hz[:,i], Nx, Nx)
        RetVal_Eabs[i] = np.sqrt(abs(RetVal_Ex[i])**2 + 
                                      abs(RetVal_Ey[i])**2 + 
                                      abs(RetVal_Ez[i])**2)
        RetVal_Habs[i] = np.sqrt(abs(RetVal_Hx[i])**2 + 
                                      abs(RetVal_Hy[i])**2 + 
                                      abs(RetVal_Hz[i])**2) 
    RetVal['beta'] = beta    
    RetVal['n'] = n
    RetVal['dx'] = dx
    RetVal['lam'] = lam
    RetVal['k0'] = k0
    RetVal['Nx'] = Nx
    RetVal['PML_Depth'] = PML_Depth
    RetVal['PML_TargetLoss'] = PML_TargetLoss
    RetVal['PML_PolyDegree'] = PML_PolyDegree
    RetVal['PML_SigmaMax'] = PML_SigmaMax
    return RetVal, RetVal_Ex, RetVal_Ey, RetVal_Ez, \
    RetVal_Hx, RetVal_Hy, RetVal_Hz, RetVal_Eabs, RetVal_Habs

def main():
    # Set up problem
    um = 1e-6
    lam = 0.65*um
    k0 = 2*np.pi/lam
    beta = k0 # Propagation constant will be close to that of free space.
    Nx = 200
    NoModes = 2
    n_silica = 1.45654 # Schott N-SF6 1.79883
    r_core = 56.3*um
    r_clad = 63.7*um # treat cladding 2.6 times larger; true value = 24.7 um
    r_total = r_core + r_clad
    x = np.linspace(-81*um,81*um,Nx) # true whole fiber diameter = 162 um
    y = x.copy()
    x_mesh, y_mesh = np.meshgrid(x, y)
    r_mesh = np.sqrt(x_mesh**2 + y_mesh**2)
    n = np.ones([Nx, Nx], dtype=int)
    n = n*n_silica
    n[r_mesh < r_total] = n_silica 
    n[r_mesh < (r_total - r_clad)] = 1
    n[(x_mesh > -12.5*um) & 
      (x_mesh < 12.5*um) & 
      (y_mesh > 21*um) & 
      (y_mesh < 22.3*um)] = n_silica
    n[(x_mesh > -12.5*um) & 
      (x_mesh < 12.5*um) & 
      (y_mesh > -22.3*um) & 
      (y_mesh < -21*um)] = n_silica
    n[(18.25*um + (x_mesh-18.25*um)*np.cos(np.pi/3) - (y_mesh-11.05*um)*np.sin(np.pi/3) > 5.6*um) & 
      (18.25*um + (x_mesh-18.25*um)*np.cos(np.pi/3) - (y_mesh-11.05*um)*np.sin(np.pi/3) < 30.9*um) & 
      (11.05*um + (x_mesh-18.25*um)*np.sin(np.pi/3) + (y_mesh-11.05*um)*np.cos(np.pi/3) > 10.4*um) & 
      (11.05*um + (x_mesh-18.25*um)*np.sin(np.pi/3) + (y_mesh-11.05*um)*np.cos(np.pi/3) < 11.7*um) ] = n_silica
    
    n[(18.25*um + (x_mesh-18.25*um)*np.cos(-np.pi/3) - (y_mesh-(-11.05*um))*np.sin(-np.pi/3) > 5.6*um) & 
      (18.25*um + (x_mesh-18.25*um)*np.cos(-np.pi/3) - (y_mesh-(-11.05*um))*np.sin(-np.pi/3) < 30.9*um) & 
      (-11.05*um + (x_mesh-18.25*um)*np.sin(-np.pi/3) + (y_mesh-(-11.05*um))*np.cos(-np.pi/3) > -11.7*um) & 
      (-11.05*um + (x_mesh-18.25*um)*np.sin(-np.pi/3) + (y_mesh-(-11.05*um))*np.cos(-np.pi/3) < -10.4*um) ] = n_silica
    
    n[(-18.25*um + (x_mesh-(-18.25*um))*np.cos(-np.pi/3) - (y_mesh-11.05*um)*np.sin(-np.pi/3) > -30.9*um) & 
      (-18.25*um + (x_mesh-(-18.25*um))*np.cos(-np.pi/3) - (y_mesh-11.05*um)*np.sin(-np.pi/3) < -5.6*um) & 
      (11.05*um + (x_mesh-(-18.25*um))*np.sin(-np.pi/3) + (y_mesh-11.05*um)*np.cos(-np.pi/3) > 10.4*um) & 
      (11.05*um + (x_mesh-(-18.25*um))*np.sin(-np.pi/3) + (y_mesh-11.05*um)*np.cos(-np.pi/3) < 11.7*um) ] = n_silica
    
    n[(-18.25*um + (x_mesh-(-18.25*um))*np.cos(np.pi/3) - (y_mesh-(-11.05*um))*np.sin(np.pi/3) > -30.9*um) & 
      (-18.25*um + (x_mesh-(-18.25*um))*np.cos(np.pi/3) - (y_mesh-(-11.05*um))*np.sin(np.pi/3) < -5.6*um) & 
      (-11.05*um + (x_mesh-(-18.25*um))*np.sin(np.pi/3) + (y_mesh-(-11.05*um))*np.cos(np.pi/3) > -11.7*um) & 
      (-11.05*um + (x_mesh-(-18.25*um))*np.sin(np.pi/3) + (y_mesh-(-11.05*um))*np.cos(np.pi/3) < -10.4*um) ] = n_silica 
    n[(x_mesh > 23.9*um) & 
      (x_mesh < 57.5*um) & 
      (y_mesh > -0.5*um) & 
      (y_mesh < 0.5*um)] = n_silica
    n[(x_mesh > -57.5*um) & 
      (x_mesh < -23.9*um) & 
      (y_mesh > -0.5*um) & 
      (y_mesh < 0.5*um)] = n_silica    
    n[(20.2*um + (x_mesh-20.2*um)*np.cos(-np.pi/3) - (y_mesh-35.6*um)*np.sin(-np.pi/3) > 3.4*um) & 
      (20.2*um + (x_mesh-20.2*um)*np.cos(-np.pi/3) - (y_mesh-35.6*um)*np.sin(-np.pi/3) < 37*um) & 
      (35.6*um + (x_mesh-20.2*um)*np.sin(-np.pi/3) + (y_mesh-35.6*um)*np.cos(-np.pi/3) > 35.1*um) & 
      (35.6*um + (x_mesh-20.2*um)*np.sin(-np.pi/3) + (y_mesh-35.6*um)*np.cos(-np.pi/3) < 36.1*um) ] = n_silica
    n[(-20.2*um + (x_mesh-(-20.2*um))*np.cos(np.pi/3) - (y_mesh-35.6*um)*np.sin(np.pi/3) > -37*um) & 
      (-20.2*um + (x_mesh-(-20.2*um))*np.cos(np.pi/3) - (y_mesh-35.6*um)*np.sin(np.pi/3) < -3.4*um) & 
      (35.6*um + (x_mesh-(-20.2*um))*np.sin(np.pi/3) + (y_mesh-35.6*um)*np.cos(np.pi/3) > 35.1*um) & 
      (35.6*um + (x_mesh-(-20.2*um))*np.sin(np.pi/3) + (y_mesh-35.6*um)*np.cos(np.pi/3) < 36.1*um) ] = n_silica
    n[(20.2*um + (x_mesh-20.2*um)*np.cos(np.pi/3) - (y_mesh-(-35.6*um))*np.sin(np.pi/3) > 3.4*um) & 
      (20.2*um + (x_mesh-20.2*um)*np.cos(np.pi/3) - (y_mesh-(-35.6*um))*np.sin(np.pi/3) < 37*um) & 
      (-35.6*um + (x_mesh-20.2*um)*np.sin(np.pi/3) + (y_mesh-(-35.6*um))*np.cos(np.pi/3) > -36.1*um) & 
      (-35.6*um + (x_mesh-20.2*um)*np.sin(np.pi/3) + (y_mesh-(-35.6*um))*np.cos(np.pi/3) < -35.1*um) ] = n_silica
    n[(-20.2*um + (x_mesh-(-20.2*um))*np.cos(-np.pi/3) - (y_mesh-(-35.6*um))*np.sin(-np.pi/3) > -37*um) & 
      (-20.2*um + (x_mesh-(-20.2*um))*np.cos(-np.pi/3) - (y_mesh-(-35.6*um))*np.sin(-np.pi/3) < -3.4*um) & 
      (-35.6*um + (x_mesh-(-20.2*um))*np.sin(-np.pi/3) + (y_mesh-(-35.6*um))*np.cos(-np.pi/3) > -36.1*um) & 
      (-35.6*um + (x_mesh-(-20.2*um))*np.sin(-np.pi/3) + (y_mesh-(-35.6*um))*np.cos(-np.pi/3) < -35.1*um) ] = n_silica
    # Show refractive index profile
    fig, fillplot = plt.subplots(1, 1)
    fig.set_size_inches(8, 6)
    fig.set_dpi(640)
    contourf_ = fillplot.contourf(x/um, x/um, n, 100)
    fig.colorbar(contourf_).set_label(label='Refractive Index', labelpad = 12,
                                     fontsize=14, weight='bold')
    plt.axis('square')
    fillplot.set_xlabel('\u03bcm', fontsize=14, fontweight="bold")
    fillplot.set_ylabel('\u03bcm', fontsize=14, fontweight="bold")
    plt.show()
    
    dx = x[2]-x[1]
    # Call FD solver
    t = time.time()
    RetVal, RetVal_Ex, RetVal_Ey, RetVal_Ez, RetVal_Hx, RetVal_Hy, \
    RetVal_Hz, RetVal_Eabs, RetVal_Habs = ModeSolverFD(dx, n, lam, beta, NoModes)
    elapsed = time.time()-t
    print(elapsed)
    # Plot modes
    RetVal['beta'] = np.diag(RetVal['beta'])
    for i in range(0, NoModes):      
        fig, fillplot = plt.subplots(1, 1)
        fig.set_size_inches(8, 6)
        fig.set_dpi(600)
        contourf_= fillplot.contourf(x/um, x/um, RetVal_Eabs[i], 100)
        fig.colorbar(contourf_).set_label(label='E_abs', labelpad = 12,
                                         fontsize=14, weight='bold')
        plt.axis('square')
        fillplot.set_title('\u03B2 =' + str(RetVal['beta'][i]).strip('()'), 
                         pad = 20, fontsize=14, fontweight="bold")
        fillplot.set_xlabel('\u03bcm', fontsize=14, fontweight="bold")
        fillplot.set_ylabel('\u03bcm', fontsize=14, fontweight="bold")
        #plt.savefig("try")
        plt.show()

if __name__ == "__main__":
    main() 

