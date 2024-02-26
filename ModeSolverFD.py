import numpy as np
import scipy.sparse 
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs, inv


def ModeSolverFD(dx, n, lambda_, beta, NoModes):
    
    if n.shape[1] != n.shape[0]:
        raise ValueError('Expecting square problem space...')
    if lambda_ / dx < 10:
        print('lambda_/dx < 10: This will likely cause discretization errors...')
    
    eps0 = 8.85e-12
    mu0 = 4 * np.pi * 10**-7
    c = 3e8
    
    if 'NoModes' not in locals():
        NoModes = 1

    Nx = n.shape[0]
    f = c / lambda_
    w = 2 * np.pi * f
    k0 = 2 * np.pi / lambda_
    
    PML_Depth = 10
    PML_TargetLoss = 1e-5
    PML_PolyDegree = 3
    
    PML_SigmaMax = (PML_PolyDegree + 1) / 2 * eps0 * c / PML_Depth / dx * np.log(1 / PML_TargetLoss)
    Epsr = np.square(n)
    Epsr = MatrixToColumn(Epsr)
    
    print('Calculating Ux, Uy, Vx, Vy...\n') 
    I = scipy.sparse.eye(Nx * Nx)
    idx_x = 0
    idx_y = 1
    Epsx = np.zeros(Nx * Nx)
    Epsy = np.zeros(Nx * Nx)
    Epsz = np.zeros(Nx * Nx)
    Ax_idxi = np.zeros(2 * Nx * Nx, dtype=int)
    Ax_idxj = np.zeros(2 * Nx * Nx, dtype=int)
    Ax_vals = np.zeros(2 * Nx * Nx, dtype=complex)
    Ay_idxi = np.zeros(2 * Nx * Nx, dtype=int)
    Ay_idxj = np.zeros(2 * Nx * Nx, dtype=int)
    Ay_vals = np.zeros(2 * Nx * Nx, dtype=complex)
    Bx_idxi = np.zeros(2 * Nx * Nx, dtype=int)
    Bx_idxj = np.zeros(2 * Nx * Nx, dtype=int)
    Bx_vals = np.zeros(2 * Nx * Nx, dtype=complex)
    By_idxi = np.zeros(2 * Nx * Nx, dtype=int)
    By_idxj = np.zeros(2 * Nx * Nx, dtype=int)
    By_vals = np.zeros(2 * Nx * Nx, dtype=complex)
    Cx_idxi = np.zeros(2 * Nx * Nx, dtype=int)
    Cx_idxj = np.zeros(2 * Nx * Nx, dtype=int)
    Cx_vals = np.zeros(2 * Nx * Nx, dtype=complex)
    Cy_idxi = np.zeros(2 * Nx * Nx, dtype=int)
    Cy_idxj = np.zeros(2 * Nx * Nx, dtype=int)
    Cy_vals = np.zeros(2 * Nx * Nx, dtype=complex)
    Dx_idxi = np.zeros(2 * Nx * Nx, dtype=int)
    Dx_idxj = np.zeros(2 * Nx * Nx, dtype=int)
    Dx_vals = np.zeros(2 * Nx * Nx, dtype=complex)
    Dy_idxi = np.zeros(2 * Nx * Nx, dtype=int)
    Dy_idxj = np.zeros(2 * Nx * Nx, dtype=int)
    Dy_vals = np.zeros(2 * Nx * Nx, dtype=complex)

    for i in range(Nx * Nx):
        idx_x += 1
        if idx_x > Nx:
            idx_y += 1
            idx_x = 1
    
        West_Dist = idx_x - 1
        North_Dist = idx_y - 1
        East_Dist = Nx - idx_x
        South_Dist = Nx - idx_y
    
        # Epsx, Epsy, Epsz
        if i - Nx >= 0:
            Epsx[i] = (Epsr[i] + Epsr[i - Nx]) / 2
        else:
            Epsx[i] = Epsr[i]
    
        if i - 1 >= 0:
            Epsy[i] = (Epsr[i] + Epsr[i - 1]) / 2
        else:
            Epsy[i] = Epsr[i]
    
        if i - 1 - Nx >= 0:
            Epsz[i] = (Epsr[i] + Epsr[i - 1] + Epsr[i - Nx] + Epsr[i - 1 - Nx]) / 4
        else:
            Epsz[i] = Epsr[i]


        # Sx, Sy
        if West_Dist <= PML_Depth:
            Sx_Ey = 1 - PML_SigmaMax * (1 - West_Dist / PML_Depth) ** PML_PolyDegree * 1j / w / eps0 / np.sqrt(Epsy[i])
            Sx_Ez = 1 - PML_SigmaMax * (1 - West_Dist / PML_Depth) ** PML_PolyDegree * 1j / w / eps0 / np.sqrt(Epsz[i])
            Sx_Hy = 1 - PML_SigmaMax * (1 - (West_Dist - 0.5) / PML_Depth) ** PML_PolyDegree * 1j / w / eps0 / np.sqrt(Epsx[i])
            Sx_Hz = 1 - PML_SigmaMax * (1 - (West_Dist - 0.5) / PML_Depth) ** PML_PolyDegree * 1j / w / eps0 / np.sqrt(Epsr[i])
    
        elif East_Dist <= PML_Depth:
            Sx_Ey = 1 - PML_SigmaMax * (1 - (East_Dist - 0.5) / PML_Depth) ** PML_PolyDegree * 1j / w / eps0 / np.sqrt(Epsy[i])
            Sx_Ez = 1 - PML_SigmaMax * (1 - (East_Dist - 0.5) / PML_Depth) ** PML_PolyDegree * 1j / w / eps0 / np.sqrt(Epsz[i])
            Sx_Hy = 1 - PML_SigmaMax * (1 - East_Dist / PML_Depth) ** PML_PolyDegree * 1j / w / eps0 / np.sqrt(Epsx[i])
            Sx_Hz = 1 - PML_SigmaMax * (1 - East_Dist / PML_Depth) ** PML_PolyDegree * 1j / w / eps0 / np.sqrt(Epsr[i])
    
        if North_Dist <= PML_Depth:
            Sy_Ex = 1 - PML_SigmaMax * (1 - North_Dist / PML_Depth) ** PML_PolyDegree * 1j / w / eps0 / np.sqrt(Epsx[i])
            Sy_Ez = 1 - PML_SigmaMax * (1 - North_Dist / PML_Depth) ** PML_PolyDegree * 1j / w / eps0 / np.sqrt(Epsz[i])
            Sy_Hx = 1 - PML_SigmaMax * (1 - (North_Dist - 0.5) / PML_Depth) ** PML_PolyDegree * 1j / w / eps0 / np.sqrt(Epsy[i])
            Sy_Hz = 1 - PML_SigmaMax * (1 - (North_Dist - 0.5) / PML_Depth) ** PML_PolyDegree * 1j / w / eps0 / np.sqrt(Epsr[i])
    
        elif South_Dist <= PML_Depth:
            Sy_Ex = 1 - PML_SigmaMax * (1 - (South_Dist - 0.5) / PML_Depth) ** PML_PolyDegree * 1j / w / eps0 / np.sqrt(Epsx[i])
            Sy_Ez = 1 - PML_SigmaMax * (1 - (South_Dist - 0.5) / PML_Depth) ** PML_PolyDegree * 1j / w / eps0 / np.sqrt(Epsz[i])
            Sy_Hx = 1 - PML_SigmaMax * (1 - South_Dist / PML_Depth) ** PML_PolyDegree * 1j / w / eps0 / np.sqrt(Epsy[i])
            Sy_Hz = 1 - PML_SigmaMax * (1 - South_Dist / PML_Depth) ** PML_PolyDegree * 1j / w / eps0 / np.sqrt(Epsr[i])

        # Ax
        Ax_idxi[2 * i] = i
        Ax_idxj[2 * i] = i
        Ax_vals[2 * i] = -1 / Sx_Ez
        if i + 1 < Nx * Nx:
            Ax_idxi[2 * i + 1] = i
            Ax_idxj[2 * i + 1] = i + 1
            Ax_vals[2 * i + 1] = 1 / Sx_Ez
    
        # Bx
        Bx_idxi[2 * i] = i
        Bx_idxj[2 * i] = i
        Bx_vals[2 * i] = -1 / Sx_Ey
        if i + 1 < Nx * Nx:
            Bx_idxi[2 * i + 1] = i
            Bx_idxj[2 * i + 1] = i + 1
            Bx_vals[2 * i + 1] = 1 / Sx_Ey
    
        # Ay
        Ay_idxi[2 * i] = i
        Ay_idxj[2 * i] = i
        Ay_vals[2 * i] = -1 / Sy_Ez
        if i + Nx < Nx * Nx:
            Ay_idxi[2 * i + 1] = i
            Ay_idxj[2 * i + 1] = i + Nx
            Ay_vals[2 * i + 1] = 1 / Sy_Ez
    
        # By
        By_idxi[2 * i] = i
        By_idxj[2 * i] = i
        By_vals[2 * i] = -1 / Sy_Ex
        if i + Nx < Nx * Nx:
            By_idxi[2 * i + 1] = i
            By_idxj[2 * i + 1] = i + Nx
            By_vals[2 * i + 1] = 1 / Sy_Ex
    
        # Cx
        Cx_idxi[2 * i] = i
        Cx_idxj[2 * i] = i
        Cx_vals[2 * i] = 1 / Sx_Hz
        if i - 1 >= 0:
            Cx_idxi[2 * i + 1] = i
            Cx_idxj[2 * i + 1] = i - 1
            Cx_vals[2 * i + 1] = -1 / Sx_Hz
    
        # Dx
        Dx_idxi[2 * i] = i
        Dx_idxj[2 * i] = i
        Dx_vals[2 * i] = 1 / Sx_Hy
        if i - 1 >= 0:
            Dx_idxi[2 * i + 1] = i
            Dx_idxj[2 * i + 1] = i - 1
            Dx_vals[2 * i + 1] = -1 / Sx_Hy
    
        # Cy
        Cy_idxi[2 * i] = i
        Cy_idxj[2 * i] = i
        Cy_vals[2 * i] = 1 / Sy_Hz
        if i - Nx >= 0:
            Cy_idxi[2 * i + 1] = i
            Cy_idxj[2 * i + 1] = i - Nx
            Cy_vals[2 * i + 1] = -1 / Sy_Hz
    
        # Dy
        Dy_idxi[2 * i] = i
        Dy_idxj[2 * i] = i
        Dy_vals[2 * i] = 1 / Sy_Hx
        if i - Nx >= 0:
            Dy_idxi[2 * i + 1] = i
            Dy_idxj[2 * i + 1] = i - Nx
            Dy_vals[2 * i + 1] = -1 / Sy_Hx                     
            
    Ax_vals = Ax_vals[Ax_idxi != 0]
    Ax_idxj = Ax_idxj[Ax_idxi != 0]
    Ax_idxi = Ax_idxi[Ax_idxi != 0]
    Ax = csr_matrix((Ax_vals, (Ax_idxi, Ax_idxj)), shape=(Nx*Nx, Nx*Nx))
    
    Ay_vals = Ay_vals[Ay_idxi != 0]
    Ay_idxj = Ay_idxj[Ay_idxi != 0]
    Ay_idxi = Ay_idxi[Ay_idxi != 0]
    Ay = csr_matrix((Ay_vals, (Ay_idxi, Ay_idxj)), shape=(Nx*Nx, Nx*Nx))
    
    Bx_vals = Bx_vals[Bx_idxi != 0]
    Bx_idxj = Bx_idxj[Bx_idxi != 0]
    Bx_idxi = Bx_idxi[Bx_idxi != 0]
    Bx = csr_matrix((Bx_vals, (Bx_idxi, Bx_idxj)), shape=(Nx*Nx, Nx*Nx))
    
    By_vals = By_vals[By_idxi != 0]
    By_idxj = By_idxj[By_idxi != 0]
    By_idxi = By_idxi[By_idxi != 0]
    By = csr_matrix((By_vals, (By_idxi, By_idxj)), shape=(Nx*Nx, Nx*Nx))
    
    Cx_vals = Cx_vals[Cx_idxi != 0]
    Cx_idxj = Cx_idxj[Cx_idxi != 0]
    Cx_idxi = Cx_idxi[Cx_idxi != 0]
    Cx = csr_matrix((Cx_vals, (Cx_idxi, Cx_idxj)), shape=(Nx*Nx, Nx*Nx))
    
    Cy_vals = Cy_vals[Cy_idxi != 0]
    Cy_idxj = Cy_idxj[Cy_idxi != 0]
    Cy_idxi = Cy_idxi[Cy_idxi != 0]
    Cy = csr_matrix((Cy_vals, (Cy_idxi, Cy_idxj)), shape=(Nx*Nx, Nx*Nx))
    
    Dx_vals = Dx_vals[Dx_idxi != 0]
    Dx_idxj = Dx_idxj[Dx_idxi != 0]
    Dx_idxi = Dx_idxi[Dx_idxi != 0]
    Dx = csr_matrix((Dx_vals, (Dx_idxi, Dx_idxj)), shape=(Nx*Nx, Nx*Nx))
    
    Dy_vals = Dy_vals[Dy_idxi != 0]
    Dy_idxj = Dy_idxj[Dy_idxi != 0]
    Dy_idxi = Dy_idxi[Dy_idxi != 0]
    Dy = csr_matrix((Dy_vals, (Dy_idxi, Dy_idxj)), shape=(Nx*Nx, Nx*Nx))
                        
    invEpsz = csr_matrix((1.0 / Epsz, (range(Nx * Nx), range(Nx * Nx))), shape=(Nx * Nx, Nx * Nx))
    Epsx = csr_matrix((Epsx, (range(Nx * Nx), range(Nx * Nx))), shape=(Nx * Nx, Nx * Nx))
    Epsy = csr_matrix((Epsy, (range(Nx * Nx), range(Nx * Nx))), shape=(Nx * Nx, Nx * Nx))
    
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
    Qxx = -k0**(-2) * Ax @ Dy @ Cx @ invEpsz @ By + (Epsy + k0**(-2) * Ax @ Dx) @ (k0**2 * np.eye(Nx * Nx) + Cy @ invEpsz @ By)
    Qyy = -k0**(-2) * Ay @ Dx @ Cy @ invEpsz @ Bx + (Epsx + k0**(-2) * Ay @ Dy) @ (k0**2 * np.eye(Nx * Nx) + Cx @ invEpsz @ Bx)
    Qxy = k0**(-2) * Ax @ Dy @ (k0**2 * np.eye(Nx * Nx) + Cx @ invEpsz @ Bx) - (Epsy + k0**(-2) * Ax @ Dx) @ Cy @ invEpsz @ Bx
    Qyx = k0**(-2) * Ay @ Dx @ (k0**2 * np.eye(Nx * Nx) + Cy @ invEpsz @ By) - (Epsx + k0**(-2) * Ay @ Dy) @ Cx @ invEpsz @ By
    
    Q = np.block([[Qxx, Qxy], [Qyx, Qyy]])
    ## Diagonalisation
    print('Taking Eigenvalues and Eigenvectors...\n')

    eigvalues, eigvectors = eigs(Q, k=NoModes, sigma=np.square(beta))
    beta = np.sqrt(np.diag(eigvalues))
    # Ex, Ey, Ez
    print('Calculating Ex, Ey, Ez, Hx, Hy, Hz...\n')
    Ex = np.zeros((Nx*Nx, NoModes), dtype=complex)
    Ey = np.zeros((Nx*Nx, NoModes), dtype=complex)
    Ez = np.zeros((Nx*Nx, NoModes), dtype=complex)
    Hx = np.zeros((Nx*Nx, NoModes), dtype=complex)
    Hy = np.zeros((Nx*Nx, NoModes), dtype=complex)
    Hz = np.zeros((Nx*Nx, NoModes), dtype=complex)     
    # Inside the loop for i in range(NoModes):
    for i in range(NoModes):
        Hx[:, i] = eigvectors[0:Nx * Nx, i]
        Hy[:, i] = eigvectors[Nx * Nx:2 * Nx * Nx, i]
        Ez[:, i] = inv(invEpsz) @ (-Dy @ Hx[:, i] + Dx @ Hy[:, i]) / (1j * w * eps0)        
        Ey[:, i] = (-1j * w * mu0 * Hx[:, i] - Ay @ Ez[:, i]) / (1j * beta[i,i])   
        Ex[:, i] = (1j * w * mu0 * Hy[:, i] - Ax @ Ez[:, i]) / (1j * beta[i,i])
        Hz[:, i] = -(-By @ Ex[:, i] + Bx @ Ey[:, i]) / (1j * w * mu0)

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
    RetVal['lam'] = lambda_
    RetVal['k0'] = k0
    RetVal['Nx'] = Nx
    RetVal['PML_Depth'] = PML_Depth
    RetVal['PML_TargetLoss'] = PML_TargetLoss
    RetVal['PML_PolyDegree'] = PML_PolyDegree
    RetVal['PML_SigmaMax'] = PML_SigmaMax
    return RetVal, RetVal_Ex, RetVal_Ey, RetVal_Ez, \
    RetVal_Hx, RetVal_Hy, RetVal_Hz, RetVal_Eabs, RetVal_Habs
    
def ColumnToMatrix(C, Nx, Ny):
    M = np.reshape(C, (Nx, Ny)).T
    return M

def MatrixToColumn(M):
    M = M.T
    C = M.flatten()
    return C