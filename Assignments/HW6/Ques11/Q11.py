import numpy as np
from scipy.constants import hbar,m_e,eV
from matplotlib import pyplot as plt


def RK4(ode_func, y0, t):
    """
    Implementation of the Fourth Order Runge-Kutta (RK4) method.
    Args:
        ode_func (function): ODE function f(t, y) to be solved.
        y0 (float): Initial value of the dependent variable.
        t0 (float): Initial value of the independent variable.
        t_end (float): End value of the independent variable.
        h (float): Step size.
    Returns:
        tuple: Tuple containing two arrays: t (array of time steps) and y (array of solutions).
    """
    h = t[1]-t[0]
    n = len(t)
    y = np.zeros([len(y0),n+1])
    y[:,0] = y0

    for i in range(n):
        k1 = h * ode_func(t[i], y[:,i])
        k2 = h * ode_func(t[i] + h/2, y[:,i] + k1[:,0]/2)
        k3 = h * ode_func(t[i] + h/2, y[:,i] + k2[:,0]/2)
        k4 = h * ode_func(t[i] + h, y[:,i] + k3[:,0])

        y[:,i + 1] = y[:,i] + (k1[:,0] + 2*k2[:,0] + 2*k3[:,0] + k4[:,0]) / 6

    return y[:,:n]



V0 = 50*eV
a = 1e-11


def solve_schrodinger_eq(E0,potential,tol = 1e-8,xi = -10*a,xf = 10*a,N = 1000):

    x = np.linspace(xi,xf,N)
    x0 = [.001,0]

    def f(x,psi,V,E): 
        w, dwdt = psi
        ddwdt = (2*m_e/hbar**2)*(V(x)-E)*w
        return np.array([[dwdt,ddwdt]]).T

    prevE = E0
    E = E0 + 10*eV
    dE = 0.1*E0
    prevPhi = RK4(lambda x,psi: f(x,psi,potential,prevE),x0,x)
    Phi = prevPhi
    
    while (np.abs(((E-prevE)/eV)/(prevE/eV))>=tol):
        prevE = E
        prevPhi = np.copy(Phi)
        E = prevE + dE
        Phi = RK4(lambda x,psi: f(x,psi,potential,E),x0,x)
        if (np.abs(Phi[0,-1])>np.abs(prevPhi[0,-1])):
            dE = -dE
        if (Phi[0,-1]*prevPhi[0,-1]<0):
            dE = dE/2
    
    return (x,prevE,prevPhi)

print('Harmonic Oscillator:')
V = lambda x: V0*(x/a)**2
x, Energy, phi = solve_schrodinger_eq(V0,V)
print('Energy of ground state is {} eV'.format(Energy/eV))
x, Energy, phi = solve_schrodinger_eq(7*V0,V)
print('Energy of 1st excited state is {} eV'.format(Energy/eV))
x, prevE, phi = solve_schrodinger_eq(20*V0,V)
print('Energy of 2nd excited state is {} eV'.format(Energy/eV))

print('\nAnharmonic Oscillator:')
V = lambda x: V0*(x/a)**4
x, Energy, phi1 = solve_schrodinger_eq(V0,V)
print('Energy of ground state is {} eV'.format(Energy/eV))
x, Energy, phi2 = solve_schrodinger_eq(10*V0,V)
print('Energy of 1st excited state is {} eV'.format(Energy/eV))
x, Energy, phi3 = solve_schrodinger_eq(20*V0,V)
print('Energy of 2nd excited state is {} eV'.format(Energy/eV))

'''
Harmonic Oscillator:
Energy of ground state is 138.0214118957519 eV
Energy of 1st excited state is 414.0642380714417 eV
Energy of 2nd excited state is 414.0642380714417 eV

Anharmonic Oscillator:
Energy of ground state is 205.3018283843994 eV
Energy of 1st excited state is 735.6730675697319 eV
Energy of 2nd excited state is 1443.5337638854967 eV
'''

A1 = 2*np.trapz(x[0:500],phi1[0][0:500])
A2 = 2*np.trapz(x[0:500],phi2[0][0:500])
A3 = 2*np.trapz(x[0:500],phi3[0][0:500])
phi1 = np.array(phi1[0])/A1
phi2 = np.array(phi2[0])/A2
phi3 = np.array(phi3[0])/A3

plt.figure(figsize=(10,10))
plt.plot(x,phi1)
plt.plot(x,phi2)
plt.plot(x,phi3)
plt.ylim((-1e11,1e11))

plt.grid()
plt.ylabel(r'$phi$')
plt.xlabel(r'$x$')
plt.title(r'Anharmonic Oscillator')
plt.legend(loc='best')
plt.savefig('Ques11/11.png')
plt.show()

