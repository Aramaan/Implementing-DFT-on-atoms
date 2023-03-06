import numpy as np
from scipy.constants import electron_mass,Planck,electron_volt

me, h, eV = electron_mass,Planck,electron_volt
hbar = h/(2*np.pi)

w = 1e-9
V = 20
z0 = w*np.sqrt(2*me*V*eV)/(2*hbar)
L =1

def fpositive(func,a,b,e):
    c = (a+b)/2
    while(np.abs(func(c))>=e):
        c = (a*func(b)-b*func(a))/(func(b)-func(a))
        if (func(a)*func(c)>=0):
            a=c
        else: 
            b=c
    return c

def sym(z):
    return  -z*np.sin(z) + np.sqrt(z0**2-z**2)* np.cos(z)
      

def asym(z):
    return z*np.cos(z) + np.sqrt(z0**2-z**2)* np.sin(z)

def Z(E):
    print(E)
    return np.sqrt(2*me*E*eV)*w/2

def E(z):
    #print('{}cc'.format(z   ))
    return (2*z*hbar/(w))**2/(2*me*eV)

print('Symmetric wavefunction energies')
for n in range(0,4):
    a = (1 - 0.3*3 + 3*n)
    b = (1 + 0.3*3 + 3*n)
    E = ((1/(2*me))*(2*hbar*fpositive(sym,a,b,1e-6)/(L*1e-9))**2)/eV
    if (E<V): print("%.6g eV "%E)
print('Symmetric wavefunction energies')
for n in range(0,4):
    a = (2.7+3*n)-0.3*3
    b = (2.7+3*n)+0.3*3
    E = ((1/(2*me))*(2*hbar*fpositive(asym,a,b,1e-6)/(L*1e-9))**2)/eV
    if (E<V): print("%.6g eV "%E)

    

