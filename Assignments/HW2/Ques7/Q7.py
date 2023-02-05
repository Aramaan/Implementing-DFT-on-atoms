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
 
'''
Trapezoidal Integration
Slices: 1, Integral Estimate: 0.147979484546652, Error Estimate: 0.23633656434643013
Slices: 2, Integral Estimate: 0.3252319078064746, Error Estimate: 0.2494012572224758
Slices: 4, Integral Estimate: 0.5122828507233315, Error Estimate: 0.14571386966011093
Slices: 8, Integral Estimate: 0.40299744847824825, Error Estimate: 0.03614122775533161
Slices: 16, Integral Estimate: 0.43010336929474696, Error Estimate: 0.024415061990297117
Slices: 32, Integral Estimate: 0.4484146657874698, Error Estimate: 0.0073310205705415426
Slices: 64, Integral Estimate: 0.45391293121537596, Error Estimate: 0.001914097543234794
Slices: 128, Integral Estimate: 0.45534850437280205, Error Estimate: 0.00048368277391852565
Slices: 256, Integral Estimate: 0.45571126645324095, Error Estimate: 0.00012124426456446476
Slices: 512, Integral Estimate: 0.4558021996516643, Error Estimate: 3.033130767415848e-05
Slices: 1024, Integral Estimate: 0.4558249481324199, Error Estimate: 7.584092302096815e-06
Slices: 2048, Integral Estimate: 0.4558306362016465, Error Estimate: 1.8961021654254986e-06
Slices: 4096, Integral Estimate: 0.45583205827827056, Error Estimate: 4.7403048442085094e-07
romberg Integration
Slices: 1, Integral Estimate: 0.147979484546652, Error Estimate: 0.23633656434643013
Slices: 2, Integral Estimate: 0.38431604889308213, Error Estimate: 0.0353421870356061
Slices: 4, Integral Estimate: 0.34897386185747603, Error Estimate: 0.10685867044753744
Slices: 8, Integral Estimate: 0.45583253230501347, Error Estimate: 4.066580405748255e-12
'''
