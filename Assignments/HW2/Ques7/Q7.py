import numpy as np
import sys, os.path
sys.path.append(os.path.abspath('.'))
from Packages.Integration import trapezoidal,rombergTRAP

def f(x):
    return (np.sin(np.sqrt(100*x)))**2

maxE = 10**(-6)
If = maxE
Ii = 0
n = 1

print('Trapezoidal Integration')
while(np.abs(If -Ii)>= maxE):
    Ii = trapezoidal(0,1,n,f)
    If = trapezoidal(0,1,2*n,f)
    error = np.abs(If - Ii)*(4/3)
    print('Slices: {}, Integral Estimate: {}, Error Estimate: {}'.format(n,Ii,error))
    n *= 2

maxE = 10**(-6)
If = maxE
Ii = 0.
n = 1

print('romberg Integration')
while(np.abs(If -Ii)>= maxE):
    Ii = rombergTRAP(0,1,n,f,trapezoidal)
    If = rombergTRAP(0,1,2*n,f,trapezoidal)
    error = np.abs(If - Ii)
    print('Slices: {}, Integral Estimate: {}, Error Estimate: {}'.format(n,Ii,error))
    n *= 2  
 

