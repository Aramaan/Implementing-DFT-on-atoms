
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys, os.path
sys.path.append(os.path.abspath('.'))
from Packages.Integration import Simpson

def f(x):
    return (x**4-2*x+1)

a = 0
b = 2
N = 10
x = np.linspace(a,b,N)
y = f(x)
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function and its integral')
plt.legend(['function','Integral'])
I = Simpson(a,b,N,f)
print(x.shape)
plt.plot(x[::2],I)
plt.show()

