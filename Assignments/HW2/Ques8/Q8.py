import numpy as np
import sys, os.path
sys.path.append(os.path.abspath('.'))
from  Packages.Integration import simpson

def f(x):
    return (np.sin(np.sqrt(100*x)))**2

maxE = 10**(-6)
If = maxE
Ii = 0
n = 2
print('Simpson Integration')
while(np.abs(If -Ii)>= maxE):
    Ii = simpson(0,1,n,f)
    If = simpson(0,1,2*n,f)
    error = np.abs(If - Ii)*(4/3)
    print('Slices: {}, Integral Estimate: {}, Error Estimate: {}'.format(n,Ii,error))
    n *= 2
