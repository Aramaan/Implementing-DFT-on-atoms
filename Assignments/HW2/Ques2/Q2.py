
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys, os.path
sys.path.append(os.path.abspath('.'))
from Packages.Integration import simpson,trapezoidal

def f(x):
    return (x**4-2*x+1)

a = 0
b = 2
N = 10
N = [10,100,1000]
for n in N:
    I = simpson(a,b,n,f)
    absERROR = I - 4.4
    fracERROR = absERROR/4.4
    print('Simpson Integration results in fractional error of {} for {} slices'.format(fracERROR,n))


for n in N:
    I =  trapezoidal(a,b,n,f)
    absERROR = I - 4.4
    fracERROR = absERROR/4.4
    print('Trapezoidal Integration results in fractional error of {} for {} slices'.format(fracERROR,n))
