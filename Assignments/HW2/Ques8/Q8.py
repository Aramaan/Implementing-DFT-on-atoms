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

'''
Simpson Integration
Slices: 2, Integral Estimate: 0.38431604889308213, Error Estimate: 0.2537561548478241
Slices: 4, Integral Estimate: 0.5746331650289502, Error Estimate: 0.27741891195430624
Slices: 8, Integral Estimate: 0.36656898106322056, Error Estimate: 0.0967595935604791
Slices: 16, Integral Estimate: 0.4391386762335799, Error Estimate: 0.02050634006861925
Slices: 32, Integral Estimate: 0.4545184312850443, Error Estimate: 0.0016363400972888944
Slices: 64, Integral Estimate: 0.455745686358011, Error Estimate: 0.00010845653413316114
Slices: 128, Integral Estimate: 0.45582702875861086, Error Estimate: 6.877850813324926e-06
Slices: 256, Integral Estimate: 0.45583218714672086, Error Estimate: 4.314281130124442e-07
'''