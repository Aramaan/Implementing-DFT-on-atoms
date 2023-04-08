import numpy as np
from scipy.constants import hbar,m_e,eV

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

#Relation of code units with SI units
hbar = 1.0545718e-34 #Js
UNIT_ENERGY = hbar #nm -> m
UNIT_TIME = 1 #s
UNIT_LENGTH = 1e-10 # A -> m
UNIT_MASS = UNIT_ENERGY*UNIT_TIME**2/UNIT_LENGTH**2

hbar = 1.0
m = 9.10938356e-31/UNIT_MASS #electron mass

def f(x,psi,V,E): 
    psi, x1 = psi
    psidot = x1
    psiddot = (2*m/hbar**2)*(V(x)-E)*psi
    return np.array([[psidot,psiddot]]).T

V0 = 50*1.609e-19/UNIT_ENERGY
a = 1e-11/UNIT_LENGTH

x = np.linspace(-10*a,10*a,100) #Time in code units
x0 = [.001,0]

def eigen_energy(guess,potential):
    Enew = guess
    Eold = guess
    dE = 0.1*guess
    tol = 1e-8
    soln_old = RK4.RK4(f,x0,x,potential,Eold)
    soln_new = np.copy(soln_old)
    counter = 0
    
    while (np.abs((Enew-Eold)/Eold)>=tol or counter==0):
        counter += 1
        Eold = Enew
        soln_old = np.copy(soln_new)
        Enew = Eold + dE
        soln_new = RK4(f,x0,x,potential,Enew)
        if (np.abs(soln_new[0,-1])>np.abs(soln_old[0,-1])): dE = -dE
        if (soln_new[0,-1]*soln_old[0,-1]<0): dE = dE/2
    
    return (Eold,soln_old)
 
def find_states(n,potential):
        counter = -1
        Enew = 0
        Eold = 0
        factor = 1
        while(counter<n):
            if(Enew!=Eold): counter += 1
            Eold = Enew
            Enew, wave_func = eigen_energy(factor*V0,potential)            
            factor += 0.5
        return Enew

potential = lambda x: V0*(x/a)**2
print('Harmonic Oscillator:')
E = find_states(0,potential)
print('Ground State: %.1f eV'%(E*UNIT_ENERGY/1.602e-19))
E = find_states(0,potential)
print('1st Excited State: %.1f eV'%(Eold*UNIT_ENERGY/1.602e-19))
Eold, wave_func = eigen_energy(10*V0,potential)
print('2nd Excited State: %.1f eV'%(Eold*UNIT_ENERGY/1.602e-19))
potential = lambda x: V0*(x/a)**4
print('\nAnharmonic Oscillator:')
Eold, wave_func = eigen_energy(2*V0,potential)
print('Ground State: %.1f eV'%(Eold*UNIT_ENERGY/1.602e-19))
Eold, wave_func = eigen_energy(10*V0,potential)
print('1st Excited State: %.1f eV'%(Eold*UNIT_ENERGY/1.602e-19))
Eold, wave_func = eigen_energy(18*V0,potential)
print('2nd Excited State: %.1f eV'%(Eold*UNIT_ENERGY/1.602e-19))