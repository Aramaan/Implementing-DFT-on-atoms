import numpy as np
import sys, os.path
sys.path.append(os.path.abspath('.'))
from Packages.Integration import trapezoidal,romberg

def f(x):
    return (np.sin(np.sqrt(100*x)))**2

maxE = 10**(-6)

n = 1
Ii = 0
while(np.abs(If -Ii)> maxE):

    print('Slices: {}, Integral Estimate: {}, Error Estimate: {}'.format())