import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import gravitational_constant
import sys, os
sys.path.append(os.path.abspath("."))
from Packages.Integration import GaussQuad2d

G = gravitational_constant

'''
Gravitational pull of a uniform sheet
'''
M = 1000
Area = 10*10
sigma = M/Area

def Fz(z):
    Integrand = lambda x,y : G*sigma*z/(x**2 + y**2 + z**2)**(3/2)
    Integral = GaussQuad2d(-5,5,100,Integrand)
    return np.nan_to_num(Integral)


Z = np.linspace(0,10,30)
F = np.array([Fz(z)for z in Z])
plt.plot(Z,F)
plt.xlabel(r'$z$')
plt.ylabel(r'$F_z$')
plt.title('Gravitational pull')
plt.grid()
plt.savefig('Ques14/Q14.png')
plt.show()

