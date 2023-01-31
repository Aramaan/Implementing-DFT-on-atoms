#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# %%
def f(x):
    return (x**4-2*x+1)



def sIntegration(x,y):
    integral = [0]
    s = x[1] -x[0]
    for i in range(2,len(x)):
        
        if i%2 != 0:
            continue
        
        area = (s)*(y[i-2]+4*y[i-1]+y[i])/3
        integral.append(integral[i//2-1] + area)
        print(i/2-1)
        print(integral)

    return integral 

def Simpson(a,b,n,f):
    x = np.linspace(a,b,n)
    y = f(x)
    return x,y

# %%

x,y = Simpson(0,2,11,f)
plt.plot(x,y)
I = sIntegration(x,y)
print(x.shape)
plt.plot(x[::2],I)
# %%
