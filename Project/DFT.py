import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import electron_mass,elementary_charge,hbar,epsilon_0
from scipy.integrate import trapz
from KS import KS
me = electron_mass
e = elementary_charge
e0 = epsilon_0

def grid(N):
    r = np.logspace(np.log10(1e-8), np.log10(100), 1000) 
    #np.linspace(1e-8,100,N)
    return r

def Vexternal(Z,x):
    y = -Z/x
    return y

# def Hartree(n,x):
#     y = []
#     for i,m in x,n:
#         y.append(np.trapz(lambda j: m/np.absolute(i-j),x))
#     return y

def Hartree(density, positions):
    """
    Calculate Hartree potential from electron density using the trapezoidal rule for numerical integration.

    Parameters:
        density (np.array): 1D array of electron density values.
        positions (np.array): 1D array of radial position values (r) for each point.

    Returns:
        hartree_potential (np.array): 1D array of Hartree potential values.
    """
    N = len(density)
    hartree_potential = np.zeros(N)
    for i in range(N):
        r_i = positions[i]
        for j in range(N):
            if i != j:
                r_j = positions[j]
                r_ij = np.abs(r_i - r_j)
                trapezoid = (density[i] + density[j]) / 2 * r_ij
                hartree_potential[i] += trapezoid
    return hartree_potential

# Example usage:
def lda(rho):
    """
    Returns the exchange-correlation potential Vxc(x) for a given density rho(x)
    using the local density approximation (LDA).

    Arguments:
    rho -- an array of density values corresponding to x

    Returns:
    Vxc -- an array of Vxc(x) values corresponding to x
    """
    
    # Calculate the exchange energy density ex(rho)
    alpha = (3 / np.pi)**(1/3)
    rs = (4 * np.pi * rho / 3)**(-1/3)
    ex = -3 / (4 * np.pi * rho) * alpha * rs**(3/2)

    # Calculate the correlation energy density ec(rho)
    A = 0.0311
    B = -0.048
    rs0 = 3 / (4 * np.pi * rho)
    ec = A * np.log(rs0) + B

    # Calculate the exchange-correlation potential Vxc(x) = ex(rho(x)) + ec(rho(x))
    Vxc = ex + ec

    return Vxc

def excfunction(rho, np):
        """Compute the exchange(-correlation) energy density."""
        clda = (3 / 4) * (3.0 / np.pi)**(1 / 3)
        return -clda * rho**(4 / 3)


# def xcfunctional(rho: np.ndarray, excfunction) -> Tuple[np.ndarray, np.ndarray]:
#     """Compute the exchange-(correlation) energy density and potential."""
#     exc = excfunction(rho, np)
#     # pylint: disable=no-value-for-parameter
#     vxc = np.elementwise_grad(excfunction)(rho, agnp)
#     return exc, vxc

# def SolveKS(V,N,n,l):
#     h = 1
#     x = grid(N)
#     d2 = np.diag(-2*np.ones(N)) + np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1)
#     d2[0, 0] = -2
#     d2[0, 1] = 2
#     d2[-1, -1] = -2
#     d2[-1, -2] = 2
#     d2 *= -1/h**2  #minus
#     V = np.diagflat(V)
#     A = np.diagflat(l(l+1)/(2*x**2))
#     H = V - (1/2)*d2 + A
#     E, phi =np.linalg.eigh(H)
#     sort_indices = np.argsort(E.real)  # Sort eigenvalues in ascending order
#     E = E[sort_indices]
#     phi = phi[:, sort_indices]
#     return E, phi

def edensity(x,N,l,phi,O):
    n = np.array(N)
    while O > 0:
        num = len(l)
        for i in range(num):
            for j in range(2*l[i]+1):
                n = n + np.square(phi[i])/(4*np.pi*x**2)
                O -= 1
                if O == 0:
                    break
            if O == 0:
                break
    return n

def NotConsistent(nold,n):
    tolerance = 1e-8
    diff = np.max(np.abs(nold - n))
    if diff < tolerance:
        return False
    else:
        return True 

def TinyDFT(rhoi,Z,N,O):
    rho = rhoi
    rhoold = 0
    x = grid(N)
    counter = 0
    while(NotConsistent(rhoold,rho)):
        print(counter)
        counter += 1
        V = lda(rho) + Vexternal(Z,x) + Hartree(rho,x) 
        idx, E, phi = KS(V,x)
        print(idx)
        print(E)
        rho0ld = rho
        rho = edensity(x,N,idx,phi,O)
        
    
    return idx,E,phi,rho

N = 1000
rhoi = np.repeat(1/N,N)
Z = 19
O = Z

TinyDFT(rhoi,Z,N,O)






