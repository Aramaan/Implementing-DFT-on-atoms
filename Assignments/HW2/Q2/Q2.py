#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# %%
def f(x):
    return (x**4-2*x+1)

x = np.linspace(0,2,10)
y = f(x)

def simp(s,j,x):
    area =(s/2)*(d+c)
    pass

def sIntegration(x,y):
    integral = [0]
    y0 = y[0]
    y1 = y[1]
    y2 = y[2]
    s = y1 -y0
    for i in range(2,len(x)):
        if i%2 != 0:
            continue
        area = (s)*(y[i-2]+4*y[i-1]+y[i])/3
        integral.append( integral[i-2] + area)
        y0 = y1
        y1 = y2
        y2 = y[i]

    return integral 

plt.plot(x,y)
I = sIntegration(x,y)
x
plt.plot(x,I)
# %%
