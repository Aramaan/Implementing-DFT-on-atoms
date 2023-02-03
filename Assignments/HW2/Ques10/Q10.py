import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import Avogadro,Boltzmann
import sys, os.path
sys.path.append(os.path.abspath('.'))
from Packages.Integration import GaussQuad,mapper

V = lambda x: x**4
m = 1
def period(a,V):
    N =20
    integrand = lambda x: 4/np.sqrt((2/m)*(V(a)-V(x)))
    integral = GaussQuad(a,0,N,integrand)
    return integral

A = np.linspace(0,2,100)
P = [period(a,V) for a in A]
plt.plot(A,P)
plt.title(r'Time period of anharmonic oscillator')
plt.xlabel(r'amplitude',size=22)
plt.ylabel(r'period',size=22)
plt.savefig('Ques10/Q10.png')
plt.show()




