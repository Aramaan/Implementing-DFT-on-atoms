import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import Avogadro,Boltzmann
import sys, os.path
sys.path.append(os.path.abspath('.'))
from Packages.Integration import GaussQuad,mapper

def integrand(x):
    return x**4*np.exp(x)/(np.exp(x)-1)**2

def cv(T):
    V = 10**(-3) # 10^(-3) cumbic meters
    ro = Avogadro
    DebT = 428
    integral = GaussQuad(0,DebT/T,50,integrand)
    C = 9*V*ro*Boltzmann*(T/DebT)**3*integral
    return C

T = np.linspace(5,500,100)
Cv = cv(T)
plt.plot(T,Cv)
plt.show()

    
