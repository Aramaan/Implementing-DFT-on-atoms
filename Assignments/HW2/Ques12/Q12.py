import numpy as np
from matplotlib import pyplot as plt
from scipy.special import roots_legendre
from scipy.constants import speed_of_light,Planck,Boltzmann

'''
The Stefan-Boltzmann constant
'''

c = speed_of_light.value
hbar = Planck.value/(2*np.pi)
k = Boltzmann.value

def int():
    return 1


plt.plot()