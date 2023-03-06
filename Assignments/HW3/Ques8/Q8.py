import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import gravitational_constant
G = gravitational_constant
M = 5.974*1e24
m = 7.384*1e22
R = 3.844*1e8
w = 2.662*1e-6

def Force(r):
    return (G*M/(r**2) - G*m/(R-r)**2 - w**2*r)

def Secant(x0,e):
    x = x0
    w = x0/2
    while(np.abs(x-w)>e):
        temp = x
        x = x- Force(x)*((x-w)/(Force(x)-Force(w)))
        w = temp
    return x

print("{}} x 10^6 m is the L1 Lagrange point".format(Secant(R/2,1e-4)/1e6))

'''
325.9577268905441 x 10^6 m is the L1 Lagrange point
'''