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

I1 = simpson(0,2,N1,f)
I2 = simpson(0,2,N2,f)
dI = np.abs(I2-I1)
eI2 = dI/15
#I2 - I1 = c(h1)^4 - c(h2)^4
#h1 = 2h2 as N2 = 2N1
#I2 -I1 = 15*c*(h2)^4


print('Error calculated in this manner is {}'.format(eI2))
print('Direct Error Computation gives {}'.format(np.abs(4.4-I2)))


'''
Error calculated in this manner is 2.666666666672294e-05
Direct Error Computation gives 2.666666666417683e-05

The Two values don't agree perfectly because there are errors
 with higher degree in h (i.e. O(h5), O(h6)) that we have not taken
 into consideration while calculating this
'''

