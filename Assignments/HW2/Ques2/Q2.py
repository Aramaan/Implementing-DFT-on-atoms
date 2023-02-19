
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

'''
Simpson Integration results in fractional error of 9.696969696932294e-05 for 10 slices
Simpson Integration results in fractional error of 9.696971912311688e-09 for 100 slices
Simpson Integration results in fractional error of 9.735646632303872e-13 for 1000 slices
Trapezoidal Integration results in fractional error of 0.024218181818181812 for 10 slices
Trapezoidal Integration results in fractional error of 0.00024242181818179273 for 100 slices
Trapezoidal Integration results in fractional error of 2.4242421817452255e-06 for 1000 slices
'''