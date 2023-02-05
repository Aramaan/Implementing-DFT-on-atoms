import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import speed_of_light,Planck,Boltzmann,Stefan_Boltzmann
import sys, os
sys.path.append(os.path.abspath("."))
from Packages.Integration import GaussQuad

'''
The Stefan-Boltzmann constant
'''

c = speed_of_light
hbar = Planck/(2*np.pi)
k = Boltzmann

def int1(x):
    return x**3/(np.exp(x)-1)

def int2(t):
    return (np.tan(t)**3*np.cos(t)**(-2))/(np.exp(np.tan(t))-1)

Integral = GaussQuad(0,np.pi/2,100,int2)
Integral2 = GaussQuad(0,np.pi/2,200,int2)
error = 4*np.abs(Integral - Integral2)/3
print('The value of the integral is {}. \n The method used is Gaussian Quadrature with 100 points \n the error is around {} '.format(Integral,error))
SBconstant = k**4*Integral/(4*np.pi**2*c**2*hbar**3)
print('The calculated value of Stefan-Boltzmann constant is {} whereas that online is {}'.format(SBconstant,Stefan_Boltzmann))