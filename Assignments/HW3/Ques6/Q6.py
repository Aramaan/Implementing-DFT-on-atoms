import numpy as np
from scipy.constants import electron_mass,Planck,electron_volt

me, h, eV = electron_mass,Planck,electron_volt
hbar = h/(2*np.pi)

w = 1e-9
V = 20
z0 = w*np.sqrt(2*me*V*eV)/(2*hbar)

def fpositive(func,a,b,e):
    c = (a+b)/2
    while(np.abs(func(c))>=e):
        c = (a*func(b)-b*func(a))/(func(b)-func(a))
        if (func(a)*func(c)>=0): a=c
        else: b=c
    return c

def sym(z):
    return  -z*np.sin(z) + np.sqrt(z0-z**2)* np.cos(z)
      

def asym(z):
    return z*np.cos(z) + np.sqrt(z0-z**2)* np.sin(z)

def Z(E):
    print(E)
    return np.sqrt(2*me*E*eV)*w/2

def E(z):
    #print('{}cc'.format(z   ))
    return (2*z*hbar/(w))**2/(2*me*eV)

L = []
K = []
u = 0.01
i,j = u,np.pi/2 -u
b = E(j)
SE = lambda E: sym(Z(E))
AE = lambda E: asym(Z(E))
while (b<=V):
    a = E(i)
    b = E(j)
    e = round(E(fpositive(SE,a,b,1e-3)),3)
    if e not in L:
        L.append(e)
    e = round(E(fpositive(AE,a,b,1e-3)),3)
    if e not in K:
        K.append(e)
    i += np.pi
    j += np.pi
print(L,K)

    

