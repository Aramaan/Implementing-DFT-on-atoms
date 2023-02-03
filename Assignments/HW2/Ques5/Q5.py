import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys, os.path
sys.path.append(os.path.abspath('.'))
from Packages.Integration import simpson,trapezoidal

def f(x):
    return (x**4-2*x+1)

N1 = 10
N2 = 20

I1 = trapezoidal(0,2,N1,f)
I2 = trapezoidal(0,2,N2,f)
dI = np.abs(I2-I1)
eI2 = dI/3
#I2 - I1 = c(h1)^2 + c(h2)^2
#h1 = 2h2 as N2 = 2N1
#I2 -I1 = 3*c*(h2)^2


print('Error calculated in this manner is {}'.format(eI2))
print('Direct Error Computation gives {}'.format(np.abs(4.4-I2)))


'''
Error calculated in this manner is 0.026633333333333137
Direct Error Computation gives 0.026660000000000572

The Two values don't agree perfectly because there are errors
 with higher degree in h (i.e. O(h3), O(h4)) that we have not taken
 into consideration while calculating this
'''

