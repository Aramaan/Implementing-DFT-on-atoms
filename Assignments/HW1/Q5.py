'''
Quantum potential step
'''
import numpy as np
from scipy.constants import electron_volt,electron_mass,Planck

def wv(E):
    return np.sqrt(2*electron_mass*(E)*electron_volt)*2*np.pi/Planck

def StepProb(E,V):
    k1 = wv(E)
    k2 = wv(E-V)
    T = 4*k1*k2/(k1+k2)**2
    R = ((k1-k2)/(k1+k2))**2
    return T,R

P = StepProb(10,9)
print('The Reflection and Transmission Probability of the electron is %0.4f and %0.4f respectively' %(P[0],P[1]))
