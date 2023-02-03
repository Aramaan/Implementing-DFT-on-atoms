import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import gravitational_constant
import sys, os
sys.path.append(os.path.abspath(".."))
from Packages.Integration import GaussQuad2d

G = gravitational_constant.value

'''
Gravitational pull of a uniform sheet
'''


I = lambda x,y,z: np.nan_to_num(z/(x**2+y**2+z**2)**1.5) #To remove singularity at (0,0,0)

L = 10
N = 100


z = np.linspace(0.,10,30)
Fz = np.array([GaussQuad2d(-L/2,L/2,100,I(Z)) for Z in z])