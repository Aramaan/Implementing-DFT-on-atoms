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


I = lambda x,y,z: np.nan_to_num(z/(x**2+y**2+z**2)**1.5) #To remove singularity at (0,0,0)

L = 10
N = 100


Z = np.linspace(0.,10,30)
I2 = lambda 
Fz = np.array([GaussQuad2d(I,-L/2,L/2,100,) for z in Z])

plt.plot(z,Fz,c='r')
plt.xlabel(r'$z$')
plt.ylabel(r'$F_z$')
plt.title('Gravitational pull')
plt.grid()
plt.savefig('14_1.png')
plt.show()