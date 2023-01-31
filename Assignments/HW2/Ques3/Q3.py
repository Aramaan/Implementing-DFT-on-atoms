
#%%
from matplotlib import pyplot as plt
import numpy as np
from Q2 import sIntegration

def f(t):
    return np.exp(-t**2)

def step(a,b,s,f):
    x = np.arange(a,b,s)
    y = f(x)
    return x,y



x,y = step(0,3,0.1,f)
plt.plot(x,y)
I = sIntegration(x,y)
plt.plot(x[::2],I)
plt.show()
 

# %%
